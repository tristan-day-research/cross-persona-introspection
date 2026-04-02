"""Patchscope activation-interpretation experiment.

This module contains only the experiment orchestration logic: setup, the
run loop, incremental checkpointing, and result saving.  All lower-level
concerns live in sibling modules:

- **patchscope_patching** — activation extract / inject / logits / relevancy.
- **patchscope_helpers** — config loading, prompt formatting, answer parsing,
  placeholder resolution, interpretation-prompt construction.
- **patchscope_evaluation** — aggregate metric computation.
- **patchscope_source_overrides** — Phase 3 raw-text override trials.
- **patchscope_reporting** — companion .txt log writer.

The experiment implements the Patchscopes protocol (Ghandeharioun et al.,
ICML 2024, arXiv:2401.06102): extract a hidden-state activation from a
*source* forward pass and inject it into a separate *interpretation* forward
pass to decode what the model's internal representation encodes.

Three phases
------------
**Phase 1 — Source extraction.**
  For every (source_persona, question, layer) triple, run one forward pass
  through the source prompt and capture the hidden-state activation.  Also
  record each source persona's direct answer (via constrained logits) so we
  can later check whether the reporter recovers it.

**Phase 2 — Interpretation matrix.**
  The full experiment matrix is::

      source_persona x question x source_layer x injection_layer
          x reporter_persona x template x condition

  where *condition* in {real, text_only_baseline, shuffled}.
  For each cell: build an interpretation prompt with placeholder tokens,
  inject the source activation at the placeholder positions, and decode the
  reporter's response (constrained logits or free-form generation).

  Conditions:
    - **real** — inject the actual source activation.
    - **text_only_baseline** — no activation injected; measures what the
      reporter can infer from the prompt text alone.
    - **shuffled** — inject a mismatched activation (from a different
      question) as a specificity control.

**Phase 3 — Source overrides (optional).**
  Sanity-check trials using raw text instead of the normal question/persona
  pipeline.  Each override specifies a literal input string and a target word
  (e.g. "The CEO resigned" / "CEO").  The experiment extracts the activation
  at that word's token position, injects it into interpretation templates,
  and checks whether the model can decode the word back.  This validates that
  the patching machinery actually carries token-level information before you
  trust the persona-level results from Phase 2.
"""

import json
import logging
import random
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import torch
from tqdm import tqdm

from cross_persona_introspection.experiments.base import BaseExperiment
from cross_persona_introspection.experiments.patchscope import (
    patchscope_evaluation,
    patchscope_helpers,
    patchscope_patching,
    patchscope_reporting,
    patchscope_source_overrides,
)
from cross_persona_introspection.schemas import (
    PatchscopeRecord,
    PersonaConfig,
    RunConfig,
)

logger = logging.getLogger(__name__)


class PatchscopeExperiment(BaseExperiment):
    """Orchestrates the full Patchscope activation-interpretation experiment.

    Lifecycle: ``setup()`` -> ``run()`` -> ``evaluate()`` -> ``save_results()``.
    """

    def __init__(self, config: RunConfig, all_personas: dict[str, PersonaConfig]):
        super().__init__(config)
        self.all_personas = all_personas
        self.ps_config: dict = {}
        self.backend = None
        self.questions: list[dict] = []
        self.records: list[PatchscopeRecord] = []
        self._sample_prompts: dict[str, dict] = {}
        self._no_persona_chat_per_layer: dict[str, dict] = {}
        self._errors: list[str] = []
        self._run_elapsed: float = 0.0
        self._jsonl_path: Optional[Path] = None
        self._log_path: Optional[Path] = None
        self._base_name: str = ""
        self._last_flush: int = 0
        self._use_chat_template: bool = True

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Load model, tokenizer, patchscope config, and questions."""
        import transformers
        import torch as _torch
        from cross_persona_introspection.backends.hf_backend import HFBackend

        transformers.logging.set_verbosity_error()

        if not self.config.patchscope_config:
            raise ValueError("patchscope experiment requires 'patchscope_config' in experiment config")
        self.ps_config = patchscope_helpers._load_patchscope_config(self.config.patchscope_config)

        dtype_map = {
            "bfloat16": _torch.bfloat16,
            "float16": _torch.float16,
            "float32": _torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.model_dtype) if self.config.model_dtype else None
        logger.info(f"Loading model: {self.config.model_name}  dtype={self.config.model_dtype or 'auto'}")
        self.backend = HFBackend(self.config.model_name, device="auto", torch_dtype=torch_dtype)

        self._source_mode = (self.ps_config.get("source_mode") or "questions").strip().lower()
        self._manual_prompts: list[dict] = []

        if self._source_mode == "manual":
            raw_manual = self.ps_config.get("manual_prompts") or []
            if not raw_manual:
                raise ValueError(
                    "source_mode is 'manual' but no manual_prompts defined in patchscope config"
                )
            self._manual_prompts = raw_manual
            # Build synthetic question dicts so Phase 2 can iterate unchanged
            self.questions = [
                {
                    "question_id": p["id"],
                    "question_text": p["text"],
                    "options": {},
                    "_manual_target_word": p["target_word"],
                    "_manual_target_strategy": p.get("target_strategy", "last"),
                }
                for p in raw_manual
            ]
            logger.info(
                "Manual source mode: %d prompts loaded from patchscope config",
                len(self.questions),
            )
        else:
            task_file = self.config.task_file
            if not task_file:
                raise ValueError("patchscope experiment requires 'task_file' in experiment config")
            self.questions = patchscope_helpers._load_questions(
                task_file,
                sample_size=self.config.sample_size,
                seed=self.config.seed,
                categories=self.ps_config.get("categories") or None,
                samples_per_category=self.ps_config.get("samples_per_category"),
            )
            logger.info(f"Loaded {len(self.questions)} questions from {task_file}")

    # ── Output / checkpointing ────────────────────────────────────────────

    def _run_filename_suffix(self) -> str:
        """Filesystem-safe slug: prompt style, templates, layer mode, placeholder."""
        ps = self.ps_config

        def slug(s: str, max_len: int = 40) -> str:
            raw = str(s).lower().replace("_", "u")
            out = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
            out = out[:max_len].rstrip("-")
            return out or "na"

        style = (
            ps.get("prompt_style")
            or ps.get("injection", {}).get("placeholder_style")
            or "patchscopes"
        )
        style_s = slug(style, 20)

        enabled = ps.get("enabled_templates") or []
        all_templates = ps.get("interpretation_templates") or {}
        if enabled:
            names = [t for t in sorted(enabled) if t in all_templates]
            joined = "+".join(slug(t, 22) for t in names)
            tmpl_s = joined if joined else "notmpl"
            if len(tmpl_s) > 50:
                tmpl_s = f"{len(names)}tmpl"
        else:
            tmpl_s = f"all{len(all_templates)}tmpl"

        if ps.get("layer_sweep", {}).get("enabled"):
            sw = ps["layer_sweep"]
            layer_s = f"sw{len(sw['source_layers'])}x{len(sw['injection_layers'])}"
        elif ps.get("layer_pairs"):
            layer_s = f"pairs{len(ps['layer_pairs'])}"
        else:
            ex = ps.get("extraction", {}).get("layers", "middle")
            inj_layer = int(ps.get("injection", {}).get("layer", 0))
            ex_s = (
                slug(str(ex), 12)
                if isinstance(ex, str)
                else f"L{len(ex)}"
            )
            layer_s = f"ex{ex_s}-inj{inj_layer}"

        ph = (ps.get("injection") or {}).get("placeholder_token") or ""
        tail = f"-ph{slug(ph, 8)}" if ph else ""

        out = f"{style_s}_{tmpl_s}_{layer_s}{tail}"
        if len(out) > 120:
            out = out[:120].rstrip("-_")
        return out

    def _init_output_paths(self) -> None:
        if self._jsonl_path is not None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = patchscope_helpers._model_short_name(self.config.model_name)
        detail = self._run_filename_suffix()
        self._base_name = f"ps_{model_short}_{timestamp}_{detail}"
        logger.info(f"Run output basename: {self._base_name}")
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = out_dir / f"{self._base_name}.jsonl"
        self._log_path = out_dir / f"{self._base_name}.txt"

    def _flush_results(self) -> None:
        """Overwrite JSONL with all records so far and refresh the log."""
        if self._jsonl_path is None:
            self._init_output_paths()
        with open(self._jsonl_path, "w") as f:
            for record in self.records:
                f.write(json.dumps(asdict(record), default=str) + "\n")
        self._write_log(self._log_path, self._base_name)
        self._last_flush = len(self.records)
        logger.info(f"  Checkpoint: flushed {len(self.records)} records to {self._jsonl_path.name}")

    # ── Run ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full experiment: Phase 1 (extract) -> Phase 2 (interpret) -> Phase 3 (overrides)."""
        assert self.backend is not None
        model = self.backend.model
        tokenizer = self.backend.tokenizer
        device = self.backend.input_device
        ps_config = self.ps_config

        self._init_output_paths()
        run_start = time.monotonic()

        # ── Unpack config ─────────────────────────────────────────────────
        extraction_cfg = ps_config["extraction"]
        injection_cfg = ps_config["injection"]
        gen_cfg = ps_config["generation"]
        relevancy_cfg = ps_config["relevancy"]
        controls_cfg = ps_config["controls"]
        templates = ps_config["interpretation_templates"]
        save_logprobs = gen_cfg.get("save_logprobs", False)

        # Filter templates
        enabled = ps_config.get("enabled_templates", [])
        if enabled:
            templates = {k: v for k, v in templates.items() if k in enabled}
            logger.info(f"Template filter: running only {list(templates.keys())}")
        else:
            logger.info(f"Running all templates: {list(templates.keys())}")

        base_prompt = ps_config["interpretation_base_prompt"]
        self._use_chat_template = bool(
            ps_config.get("use_chat_template", True)
        )
        source_persona_names = ps_config["source_personas"]
        reporter_persona_names = ps_config["reporter_personas"]

        # Resolve layer pairs
        source_layers, injection_layers, pair_map = patchscope_helpers._resolve_layer_config(
            ps_config, extraction_cfg, injection_cfg, model.config.num_hidden_layers
        )

        injection_mode = injection_cfg["mode"]
        injection_alpha = float(injection_cfg["alpha"])

        # Resolve prompt style
        style = ps_config["prompt_style"]
        configured_placeholder = injection_cfg["placeholder_token"]
        num_placeholders = int(injection_cfg.get("num_placeholders", 1))

        placeholder_token_id = patchscope_helpers._get_placeholder_token_id(tokenizer, configured_placeholder)
        placeholder_token = tokenizer.decode([placeholder_token_id])

        control_sample_rate = float(controls_cfg.get("control_sample_rate", 1.0))
        control_rng = random.Random(self.config.seed + 7)

        # Pre-resolve choice token IDs for logits-mode templates
        all_choice_token_ids: dict[str, dict[str, int]] = {}
        for tmpl_name, tmpl_cfg in templates.items():
            if tmpl_cfg.get("decode_mode", "generate") == "logits" and tmpl_name in patchscope_helpers.TEMPLATE_CHOICES:
                all_choice_token_ids[tmpl_name] = patchscope_helpers._resolve_choice_token_ids(
                    tokenizer, patchscope_helpers.TEMPLATE_CHOICES[tmpl_name]
                )
                logger.info(f"Logits mode — {tmpl_name} choices: {all_choice_token_ids[tmpl_name]}")
        logger.info(
            f"Placeholder token: id={placeholder_token_id} repr={repr(placeholder_token)} "
            f"x {num_placeholders}"
        )

        # ── Phase 1: extract source activations & direct answers ──────────

        logger.info("=== Phase 1: Extracting source activations ===")
        activations, direct_answers, shuffle_maps, extraction_meta = self._extract_sources(
            model, tokenizer, device,
            source_persona_names, source_layers,
            extraction_cfg, save_logprobs,
        )

        # ── Phase 2: activation transfer sweep ────────────────────────────

        logger.info("=== Phase 2: Running activation transfer sweep ===")

        conditions = ["real"]
        if controls_cfg.get("text_only_baseline", False):
            conditions.append("text_only_baseline")
        if controls_cfg.get("shuffled_activation", False):
            conditions.append("shuffled")

        _use_both = patchscope_helpers.extraction_uses_both_modes(extraction_cfg)
        matrix_dims = {
            "source_personas": len(source_persona_names),
            "questions": len(self.questions),
            "extraction_modes": 2 if _use_both else 1,
            "source_layers": len(source_layers),
            "injection_layers": len(injection_layers),
            "reporter_personas": len(reporter_persona_names),
            "templates": len(templates),
            "conditions": len(conditions),
        }
        total_cells = 1
        for v in matrix_dims.values():
            total_cells *= v
        matrix_breakdown = " x ".join(f"{name}={value}" for name, value in matrix_dims.items())
        matrix_msg = f"Total matrix cells: {matrix_breakdown} = {total_cells}"
        print(matrix_msg)
        logger.info(matrix_msg)

        baseline_cache: dict[tuple, dict] = {}
        cell_count = 0

        # Phase 2 core: iterate every cell in the matrix and run one
        # interpretation trial per cell.
        #
        # _iter_matrix_cells() handles the 7-level nested loop (personas x
        # questions x layers x reporters x templates x conditions) and yields
        # one dict per cell containing:
        #   - "activation": the tensor to inject (real, shuffled, or None)
        #   - "interp_text" / "interp_messages" / "placeholder_positions":
        #     the fully-built interpretation prompt
        #   - "record": a pre-populated PatchscopeRecord
        #   - metadata: condition, template name/cfg, layer indices, etc.
        #
        # _run_single_cell() does the actual injection + decode for that cell.

        for cell in self._iter_matrix_cells(
            source_persona_names=source_persona_names,
            source_layers=source_layers,
            injection_layers=injection_layers,
            pair_map=pair_map,
            reporter_persona_names=reporter_persona_names,
            templates=templates,
            conditions=conditions,
            activations=activations,
            direct_answers=direct_answers,
            shuffle_maps=shuffle_maps,
            extraction_meta=extraction_meta,
            control_sample_rate=control_sample_rate,
            control_rng=control_rng,
            style=style,
            tokenizer=tokenizer,
            base_prompt=base_prompt,
            placeholder_token=placeholder_token,
            num_placeholders=num_placeholders,
            injection_mode=injection_mode,
        ):
            cell_count += 1
            record = cell["record"]

            try:
                self._run_single_cell(
                    record=record,
                    condition=cell["condition"],
                    activation=cell["activation"],
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    interp_text=cell["interp_text"],
                    interp_messages=cell["interp_messages"],
                    placeholder_positions=cell["placeholder_positions"],
                    prompt_style=cell["prompt_style"],
                    tmpl_name=cell["tmpl_name"],
                    tmpl_cfg=cell["tmpl_cfg"],
                    inj_layer=cell["inj_layer"],
                    injection_mode=injection_mode,
                    injection_alpha=injection_alpha,
                    gen_cfg=gen_cfg,
                    relevancy_cfg=relevancy_cfg,
                    all_choice_token_ids=all_choice_token_ids,
                    save_logprobs=save_logprobs,
                    shuffle_remap=cell["shuffle_remap"],
                    baseline_cache=baseline_cache,
                    interp_question=cell["interp_question"],
                    base_prompt=base_prompt,
                    placeholder_token=placeholder_token,
                    num_placeholders=num_placeholders,
                )
            except Exception as e:
                msg = (
                    f"Interpret error: {record.source_persona}->{record.reporter_persona} "
                    f"{record.template_name} {record.condition} q={record.question_id}: {e}"
                )
                logger.error(msg)
                self._errors.append(msg)
                record.error = str(e)

            record.timestamp = datetime.now(timezone.utc).isoformat()
            self.records.append(record)

            if cell_count % 50 == 0:
                logger.info(f"  Progress: {cell_count}/{total_cells} cells")
            if len(self.records) - self._last_flush >= 20:
                self._run_elapsed = time.monotonic() - run_start
                self._flush_results()

        # ── Phase 3: source overrides (optional) ──────────────────────────

        source_override_list = ps_config.get("source_overrides", [])
        if source_override_list:
            patchscope_source_overrides.run_source_overrides(
                source_overrides=source_override_list,
                model=model,
                tokenizer=tokenizer,
                device=device,
                source_layers=source_layers,
                injection_layers=injection_layers,
                pair_map=pair_map,
                templates=templates,
                prompt_style=style,
                base_prompt=base_prompt,
                placeholder_token=placeholder_token,
                num_placeholders=num_placeholders,
                injection_mode=injection_mode,
                injection_alpha=injection_alpha,
                gen_cfg=gen_cfg,
                all_choice_token_ids=all_choice_token_ids,
                save_logprobs=save_logprobs,
                model_name=self.config.model_name,
                records=self.records,
                sample_prompts=self._sample_prompts,
                errors=self._errors,
            )
            self._run_elapsed = time.monotonic() - run_start
            self._flush_results()

        self._run_elapsed = time.monotonic() - run_start
        self._flush_results()
        logger.info(f"Completed {len(self.records)} records in {self._run_elapsed:.1f}s")

    # ── Phase 1: source extraction ────────────────────────────────────────

    def _extract_sources(
        self, model, tokenizer, device,
        source_persona_names, source_layers,
        extraction_cfg, save_logprobs,
    ) -> tuple[dict, dict, dict, dict]:
        """Extract activations and direct answers for every (persona, question, layer).

        Returns:
            (activations, direct_answers, shuffle_maps, extraction_meta)

        When ``readout: "both"``, activations are keyed as::

            activations[persona][question_idx] = {
                "prefill": {layer: tensor, ...},
                "during_gen": {layer: tensor, ...},
            }

        Otherwise (single mode), the existing flat structure is preserved::

            activations[persona][question_idx] = {layer: tensor, ...}

        extraction_meta maps ``(persona, question_idx)`` to a dict with keys:
            extraction_mode, extraction_token_index, extraction_token_id,
            extraction_token_text.  For "both" mode, maps to a dict of
            ``{"prefill": {...}, "during_gen": {...}}``.
        """
        activations: dict[str, dict[int, dict]] = {}
        direct_answers: dict[str, dict[int, dict]] = {}
        shuffle_maps: dict[tuple[str, str], tuple[dict, dict[str, str]]] = {}
        extraction_meta: dict[tuple[str, int], dict] = {}
        self._sample_source_prompts: dict[str, dict] = {}

        for sp_name in source_persona_names:
            activations[sp_name] = {}
            direct_answers[sp_name] = {}
            source_persona = self.all_personas[sp_name]

            for question_idx, question in enumerate(tqdm(
                self.questions, desc=f"  Extract [{sp_name}]", leave=True
            )):
                try:
                    question_id = question.get("question_id", f"q{question_idx}")
                    _is_manual = self._source_mode == "manual"

                    if _is_manual:
                        # ── Manual prompt mode: free-form text, extract at target_word ──
                        _target_word = question["_manual_target_word"]
                        _target_strategy = question.get("_manual_target_strategy", "last")
                        prompt_text = question["question_text"]

                        # Build raw source text with persona system prompt
                        sys_text = (source_persona.system_prompt or "").strip()
                        source_raw_text = (sys_text + "\n\n" + prompt_text) if sys_text else prompt_text

                        # Resolve target word to token index in the full source text
                        _tok_pos = patchscope_helpers.find_token_position(
                            tokenizer, source_raw_text, _target_word, strategy=_target_strategy,
                        )

                        # Always use prefill extraction for manual prompts
                        activations[sp_name][question_idx] = (
                            patchscope_patching.extract_activations_multi_layer(
                                model, tokenizer, device,
                                messages=[],  # unused when raw_text is provided
                                layer_indices=source_layers,
                                token_position=_tok_pos,
                                raw_text=source_raw_text,
                            )
                        )

                        # No MCQ direct answer for manual prompts
                        direct_answers[sp_name][question_idx] = {
                            "answer": None, "probs": {}, "logits": {},
                            "logprobs": None, "total_choice_prob": None,
                            "raw": f"[manual] target_word={_target_word!r}",
                            "shuffled_answer": None,
                            "option_order": [],
                        }
                        # Identity remap (no shuffling)
                        remap = {}
                        shuffle_maps[(question_id, sp_name)] = (question, remap)

                        # Extraction metadata for manual mode
                        token_ids = tokenizer.encode(source_raw_text, add_special_tokens=False)
                        _tok_id = int(token_ids[_tok_pos]) if 0 <= _tok_pos < len(token_ids) else None
                        _tok_text = tokenizer.decode([_tok_id]).strip() if _tok_id is not None else None
                        extraction_meta[(sp_name, question_idx)] = {
                            "extraction_mode": f"prefill_manual_{_target_word}",
                            "extraction_token_index": _tok_pos,
                            "extraction_token_id": _tok_id,
                            "extraction_token_text": _tok_text,
                        }

                        # Sample prompt for log
                        if sp_name not in self._sample_source_prompts:
                            token_ids = tokenizer.encode(source_raw_text, add_special_tokens=False)
                            tokens = [tokenizer.decode([tid]) for tid in token_ids]
                            _site = {
                                "readout": "prefill",
                                "capture_mode": "manual_target_word",
                                "target_word": _target_word,
                                "target_strategy": _target_strategy,
                                "token_position_spec": f"find_token_position({_target_word!r}, strategy={_target_strategy!r})",
                                "token_index": _tok_pos,
                                "n_tokens": len(tokens),
                                "token_id": int(token_ids[_tok_pos]) if 0 <= _tok_pos < len(token_ids) else None,
                                "token_decoded_repr": repr(tokens[_tok_pos]) if 0 <= _tok_pos < len(tokens) else None,
                                "token_decoded_strip": tokens[_tok_pos].strip() if 0 <= _tok_pos < len(tokens) else None,
                            }
                            self._sample_source_prompts[sp_name] = {
                                "persona": sp_name,
                                "question_id": question_id,
                                "prompt_text": source_raw_text,
                                "extraction_site": _site,
                                "answer": None,
                                "shuffled_answer": None,
                                "option_order": [],
                            }

                    else:
                        # ── Questions mode: existing MCQ flow ──
                        shuffled_question, remap = patchscope_helpers._shuffle_options(
                            question, self.config.seed, question_id, sp_name
                        )
                        shuffle_maps[(question_id, sp_name)] = (shuffled_question, remap)

                        _src_tpl = self.ps_config["source_pass"]["user_message_template"]
                        user_msg = patchscope_helpers.format_source_pass_user_message(
                            shuffled_question, _src_tpl
                        )
                        messages = []
                        if source_persona.system_prompt:
                            messages.append({"role": "system", "content": source_persona.system_prompt})
                        messages.append({"role": "user", "content": user_msg})

                        # Build raw text for source pass when chat template is disabled.
                        source_raw_text = None
                        if not self._use_chat_template:
                            sys_text = (source_persona.system_prompt or "").strip()
                            source_raw_text = (sys_text + "\n\n" + user_msg) if sys_text else user_msg

                        _ex_tok = extraction_cfg.get("token_position", "last")
                        _ex_boundary = extraction_cfg.get("assistant_boundary_marker")
                        _readout = (extraction_cfg.get("readout") or "prefill").strip().lower()
                        _ar = extraction_cfg.get("autoregressive") or {}
                        _raw_steps = _ar.get("decode_steps", 1)
                        _ar_stop_tokens: list[str] | None = None
                        if isinstance(_raw_steps, str) and _raw_steps.strip().lower() == "until_answer":
                            _ar_stop_tokens = _ar.get("answer_tokens", ["A", "B", "C", "D"])
                            _ar_steps = int(_ar.get("max_decode_steps", 64))
                        else:
                            _ar_steps = int(_raw_steps)
                        _ar_max = int(_ar.get("max_decode_steps", 64))
                        _ar_temp = float(_ar.get("temperature", 0.0))
                        _ar_sample = bool(_ar.get("do_sample", False))
                        _use_both = patchscope_helpers.extraction_uses_both_modes(extraction_cfg)
                        _wants_gen_capture = _use_both or patchscope_helpers.extraction_uses_generation_time_capture(
                            extraction_cfg
                        )
                        if _wants_gen_capture and _ar_steps < 1:
                            logger.warning(
                                "Generation-time extraction requested but autoregressive.decode_steps < 1; "
                                "using decode_steps=1."
                            )
                            _ar_steps = 1
                        use_ar_extract = _wants_gen_capture and _ar_steps >= 1
                        # In "both" mode, always extract prefill too
                        use_prefill_extract = _use_both or not use_ar_extract

                        # Validate extraction position once (first question of first persona)
                        if not hasattr(self, '_extraction_validated'):
                            if source_raw_text is not None:
                                source_text = source_raw_text
                            else:
                                source_text = tokenizer.apply_chat_template(
                                    messages, tokenize=False, add_generation_prompt=True
                                )
                            if use_ar_extract:
                                if _ar_stop_tokens is not None:
                                    logger.info(
                                        "Generation-time Phase-1 extraction (until_answer, stop_tokens=%s, "
                                        "max_steps=%s): activations captured at the decode step that produces "
                                        "a matching answer token.",
                                        _ar_stop_tokens,
                                        _ar_steps,
                                    )
                                else:
                                    logger.info(
                                        "Generation-time Phase-1 extraction (decode_steps=%s): activations are "
                                        "hidden[:, -1, :] after that many post-prefill decode forwards — from "
                                        "generated tokens, not from extraction.token_position / "
                                        "last_before_assistant.",
                                        _ar_steps,
                                    )
                            if use_prefill_extract:
                                patchscope_helpers.validate_extraction_position(
                                    tokenizer, source_text,
                                    _ex_tok,
                                    tmpl_name=f"source_extraction[{sp_name}]",
                                    boundary_marker=_ex_boundary,
                                )
                            if _use_both:
                                logger.info(
                                    "Dual extraction mode (readout: both): extracting from "
                                    "prefill position '%s' AND during generation.",
                                    _ex_tok,
                                )
                            self._extraction_validated = True

                        # ── Run extraction(s) ──
                        _ar_meta = None
                        _prefill_caps = None
                        _ar_caps = None

                        if use_prefill_extract:
                            _prefill_caps = (
                                patchscope_patching.extract_activations_multi_layer(
                                    model, tokenizer, device, messages,
                                    layer_indices=source_layers,
                                    token_position=_ex_tok,
                                    boundary_marker=_ex_boundary,
                                    raw_text=source_raw_text,
                                )
                            )

                        if use_ar_extract:
                            _ar_caps, _ar_meta = (
                                patchscope_patching.extract_activations_during_decode(
                                    model,
                                    tokenizer,
                                    device,
                                    messages,
                                    layer_indices=source_layers,
                                    decode_steps=_ar_steps,
                                    max_decode_steps=_ar_max,
                                    temperature=_ar_temp,
                                    do_sample=_ar_sample,
                                    raw_text=source_raw_text,
                                    stop_tokens=_ar_stop_tokens,
                                )
                            )

                        # Store activations
                        if _use_both:
                            activations[sp_name][question_idx] = {
                                "prefill": _prefill_caps or {},
                                "during_gen": _ar_caps or {},
                            }
                        elif use_ar_extract:
                            activations[sp_name][question_idx] = _ar_caps
                        else:
                            activations[sp_name][question_idx] = _prefill_caps

                        # ── Build per-question extraction metadata ──
                        _ptext = source_raw_text if source_raw_text is not None else (
                            tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                        )

                        def _prefill_meta() -> dict:
                            site = patchscope_helpers.describe_source_extraction_site(
                                tokenizer, _ptext, _ex_tok,
                                boundary_marker=_ex_boundary,
                            )
                            _tok_spec = str(_ex_tok).strip().lower().replace("-", "_")
                            return {
                                "extraction_mode": f"prefill_{_tok_spec}",
                                "extraction_token_index": site.get("token_index"),
                                "extraction_token_id": site.get("token_id"),
                                "extraction_token_text": site.get("token_decoded_repr") or site.get("token_decoded_strip"),
                            }

                        def _gen_meta() -> dict:
                            m = _ar_meta or {}
                            return {
                                "extraction_mode": "during_generation",
                                "extraction_token_index": None,  # position is dynamic (generated)
                                "extraction_token_id": m.get("activation_token_id"),
                                "extraction_token_text": (
                                    m.get("activation_token_decoded_strip")
                                    or (repr(tokenizer.decode([m["activation_token_id"]])) if m.get("activation_token_id") is not None else None)
                                ),
                            }

                        if _use_both:
                            extraction_meta[(sp_name, question_idx)] = {
                                "prefill": _prefill_meta(),
                                "during_gen": _gen_meta(),
                            }
                        elif use_ar_extract:
                            extraction_meta[(sp_name, question_idx)] = _gen_meta()
                        else:
                            extraction_meta[(sp_name, question_idx)] = _prefill_meta()

                        # Direct answer via logits
                        if source_raw_text is not None:
                            probs, logits_dict = self.backend.get_choice_probs_and_logits_from_text(
                                source_raw_text, ["A", "B", "C", "D"],
                            )
                            logprobs_dict, total_cp = None, None
                        else:
                            source_result = self.backend.get_choice_probs_and_logits(
                                messages, ["A", "B", "C", "D"], save_logprobs=save_logprobs,
                            )
                            if save_logprobs:
                                probs, logits_dict, logprobs_dict, total_cp = source_result
                            else:
                                probs, logits_dict = source_result
                                logprobs_dict, total_cp = None, None

                        shuffled_answer = max(probs, key=probs.get)
                        canonical_answer = remap.get(shuffled_answer, shuffled_answer)
                        canonical_probs = {remap.get(k, k): v for k, v in probs.items()}
                        canonical_logits = {remap.get(k, k): v for k, v in logits_dict.items()}
                        canonical_logprobs = (
                            {remap.get(k, k): v for k, v in logprobs_dict.items()}
                            if logprobs_dict else None
                        )

                        _sorted_cp = sorted(canonical_probs.values(), reverse=True)

                        # Source generated answer: the token actually produced
                        # during autoregressive decode, remapped to canonical.
                        _gen_answer_shuffled = (
                            (_ar_meta or {}).get("activation_token_decoded_strip")
                            if _ar_meta else None
                        )
                        _gen_answer_canonical = (
                            remap.get(_gen_answer_shuffled, _gen_answer_shuffled)
                            if _gen_answer_shuffled else None
                        )

                        direct_answers[sp_name][question_idx] = {
                            "answer": canonical_answer,
                            "probs": canonical_probs,
                            "logits": canonical_logits,
                            "logprobs": canonical_logprobs,
                            "total_choice_prob": total_cp,
                            "raw": f"[logits] {canonical_answer}",
                            "shuffled_answer": shuffled_answer,
                            "option_order": [remap.get(l, l) for l in ["A", "B", "C", "D"]],
                            "source_chosen_prob": _sorted_cp[0] if _sorted_cp else None,
                            "source_margin": (_sorted_cp[0] - _sorted_cp[1]) if len(_sorted_cp) > 1 else None,
                            "source_generated_answer": _gen_answer_canonical,
                        }

                        if sp_name not in self._sample_source_prompts:
                            _ptext = source_raw_text if source_raw_text is not None else (
                                tokenizer.apply_chat_template(
                                    messages, tokenize=False, add_generation_prompt=True
                                )
                            )
                            if use_ar_extract:
                                _meta = _ar_meta or {}
                                _cap_step = _meta.get("capture_at_step", _ar_steps)
                                if _ar_stop_tokens is not None:
                                    _tspec = (
                                        f"until_answer: capture hidden[:, -1, :] on forward that emits "
                                        f"first token in {list(_ar_stop_tokens)!r}"
                                    )
                                    _site = {
                                        "readout": _readout,
                                        "capture_mode": "during_generation",
                                        "autoregressive_mode": "until_answer",
                                        "stop_tokens": list(_ar_stop_tokens),
                                        "max_decode_budget": _ar_steps,
                                        "max_decode_steps": _ar_max,
                                        "temperature": _ar_temp,
                                        "do_sample": _ar_sample,
                                        "token_position_spec": _tspec,
                                        "n_tokens": len(
                                            tokenizer.encode(_ptext, add_special_tokens=False)
                                        ),
                                        **_meta,
                                    }
                                else:
                                    _site = {
                                        "readout": _readout,
                                        "capture_mode": "during_generation",
                                        "autoregressive_mode": "fixed_steps",
                                        "decode_steps": _ar_steps,
                                        "max_decode_steps": _ar_max,
                                        "temperature": _ar_temp,
                                        "do_sample": _ar_sample,
                                        "token_position_spec": (
                                            f"last_seq_position_after_{_cap_step}_post_prefill_decode_forward(s)"
                                        ),
                                        "n_tokens": len(
                                            tokenizer.encode(_ptext, add_special_tokens=False)
                                        ),
                                        **_meta,
                                    }
                            else:
                                _site = patchscope_helpers.describe_source_extraction_site(
                                    tokenizer,
                                    _ptext,
                                    extraction_cfg.get("token_position", "last"),
                                    boundary_marker=extraction_cfg.get(
                                        "assistant_boundary_marker"
                                    ),
                                )
                            self._sample_source_prompts[sp_name] = {
                                "persona": sp_name,
                                "question_id": question_id,
                                "prompt_text": _ptext,
                                "extraction_site": _site,
                                "answer": canonical_answer,
                                "shuffled_answer": shuffled_answer,
                                "option_order": [remap.get(l, l) for l in ["A", "B", "C", "D"]],
                            }

                except Exception as e:
                    msg = f"Extract error: {sp_name} q{question_idx} {question.get('question_id', '?')}: {e}"
                    logger.error(msg)
                    self._errors.append(msg)
                    activations.setdefault(sp_name, {})[question_idx] = {}
                    direct_answers.setdefault(sp_name, {})[question_idx] = {
                        "answer": None, "probs": {}, "logits": {}, "raw": ""
                    }

        return activations, direct_answers, shuffle_maps, extraction_meta

    # ── Phase 2: matrix iteration ─────────────────────────────────────────

    def _iter_matrix_cells(
        self, *,
        source_persona_names, source_layers, injection_layers, pair_map,
        reporter_persona_names, templates, conditions,
        activations, direct_answers, shuffle_maps, extraction_meta,
        control_sample_rate, control_rng,
        style, tokenizer, base_prompt, placeholder_token, num_placeholders,
        injection_mode,
    ) -> Iterator[dict]:
        """Yield one dict per cell in the Phase 2 experiment matrix.

        This generator owns the nested loop so that run() stays flat.
        Each yielded dict contains everything _run_single_cell needs:
          - "record": pre-populated PatchscopeRecord
          - "activation": tensor to inject (real, shuffled, or None for baseline)
          - "condition": which condition this cell is for
          - "interp_text" / "interp_messages" / "placeholder_positions": built prompt
          - "tmpl_name" / "tmpl_cfg" / "prompt_style": template info
          - "inj_layer": injection layer index
          - "shuffle_remap": label remap dict for answer remapping

        When ``readout: "both"`` was used, activations[persona][question] is
        ``{"prefill": {layer: tensor}, "during_gen": {layer: tensor}}``.
        This method detects that structure and loops over both extraction modes,
        yielding separate cells for each.
        """
        for sp_name in source_persona_names:
            for question_idx, question in enumerate(self.questions):
                question_id = question.get("question_id", f"q{question_idx}")

                q_acts = activations.get(sp_name, {}).get(question_idx, {})
                if not q_acts:
                    logger.warning(f"Skipping {sp_name} q={question_id}: no activations")
                    continue

                # Detect "both" mode: activations have "prefill"/"during_gen" sub-dicts
                _is_both = "prefill" in q_acts and "during_gen" in q_acts
                if _is_both:
                    extraction_modes = [("prefill", q_acts["prefill"]), ("during_gen", q_acts["during_gen"])]
                else:
                    extraction_modes = [(None, q_acts)]  # single mode, acts is {layer: tensor}

                for ext_mode_key, mode_acts in extraction_modes:
                    for src_layer in source_layers:
                        real_act = mode_acts.get(src_layer)
                        if real_act is None:
                            continue

                        # Shuffled control: activation from a different question
                        # (mismatched-content control, NOT random noise)
                        shuffled_act = None
                        if "shuffled" in conditions and len(self.questions) > 1:
                            other_question_idx = (question_idx + 1) % len(self.questions)
                            other_acts = activations.get(sp_name, {}).get(other_question_idx, {})
                            if _is_both and ext_mode_key in other_acts:
                                shuffled_act = other_acts[ext_mode_key].get(src_layer)
                            else:
                                shuffled_act = other_acts.get(src_layer)

                        paired_inj_layers = pair_map[src_layer] if pair_map else injection_layers
                        for inj_layer in paired_inj_layers:
                            for reporter_name in reporter_persona_names:
                                reporter_persona = self.all_personas[reporter_name]

                                for tmpl_name, tmpl_cfg in templates.items():
                                    prompt_style = style

                                    # Use shuffled options matching the source pass
                                    shuffle_key = (question_id, sp_name)
                                    shuffled_question = shuffle_maps.get(shuffle_key, (question, {}))[0]
                                    shuffled_options = shuffled_question.get("options", {})
                                    shuffle_remap = shuffle_maps.get(shuffle_key, (None, {}))[1]

                                    # Build the interpretation prompt and find where the
                                    # placeholder tokens landed (token indices).  These
                                    # positions are where the activation will be patched in.
                                    interp_text, interp_messages, placeholder_positions = (
                                        patchscope_helpers.build_interpretation_prompt(
                                            tokenizer=tokenizer,
                                            tmpl_cfg=tmpl_cfg,
                                            prompt_style=prompt_style,
                                            base_prompt=base_prompt,
                                            placeholder_token=placeholder_token,
                                            num_placeholders=num_placeholders,
                                            question=question,
                                            reporter_system_prompt=reporter_persona.system_prompt,
                                            options_override=shuffled_options,
                                            use_chat_template=self._use_chat_template,
                                        )
                                    )

                                    # Validate placeholder positions once per template
                                    # (first time we see this template name)
                                    _validation_key = f"_validated_{tmpl_name}"
                                    if not hasattr(self, _validation_key):
                                        patchscope_helpers.validate_placeholder_positions(
                                            tokenizer, interp_text,
                                            placeholder_positions, placeholder_token,
                                            tmpl_name=tmpl_name,
                                            raise_on_error=True,
                                        )
                                        setattr(self, _validation_key, True)

                                    for condition in conditions:
                                        # Subsample controls if configured
                                        if condition != "real" and control_sample_rate < 1.0:
                                            if control_rng.random() > control_sample_rate:
                                                continue

                                        # Select the activation for this condition:
                                        #   "real"               -> the actual source activation
                                        #   "shuffled"           -> activation from a different question
                                        #   "text_only_baseline" -> None (no injection)
                                        if condition == "text_only_baseline":
                                            activation = None
                                        elif condition == "shuffled":
                                            activation = shuffled_act
                                        else:
                                            activation = real_act

                                        # Resolve extraction metadata for this cell
                                        _emeta_raw = extraction_meta.get((sp_name, question_idx), {})
                                        if ext_mode_key is not None and ext_mode_key in _emeta_raw:
                                            _emeta = _emeta_raw[ext_mode_key]
                                        else:
                                            _emeta = _emeta_raw

                                        source_direct = direct_answers[sp_name][question_idx]
                                        record = PatchscopeRecord(
                                            experiment="patchscope",
                                            template_name=tmpl_name,
                                            model=self.config.model_name,
                                            question_id=question_id,
                                            source_persona=sp_name,
                                            reporter_persona=reporter_name,
                                            condition=condition,
                                            source_layer=src_layer,
                                            injection_layer=inj_layer,
                                            injection_mode=injection_mode,
                                            source_direct_answer=source_direct["answer"],
                                            source_generated_answer=source_direct.get("source_generated_answer"),
                                            source_answer_probs=source_direct["probs"],
                                            source_chosen_prob=source_direct.get("source_chosen_prob"),
                                            source_margin=source_direct.get("source_margin"),
                                            extraction_mode=_emeta.get("extraction_mode"),
                                            extraction_token_index=_emeta.get("extraction_token_index"),
                                            extraction_token_id=_emeta.get("extraction_token_id"),
                                            extraction_token_text=_emeta.get("extraction_token_text"),
                                            question_text=question.get("question_text", ""),
                                            question_options=question.get("options"),
                                            category_id=question.get("category_id"),
                                            category_name=question.get("category_name"),
                                            reporter_system_prompt=(
                                                (reporter_persona.system_prompt or "").strip()
                                            ),
                                        )

                                        yield {
                                            "record": record,
                                            "condition": condition,
                                            "activation": activation,
                                            "interp_text": interp_text,
                                            "interp_messages": interp_messages,
                                            "placeholder_positions": placeholder_positions,
                                            "prompt_style": prompt_style,
                                            "tmpl_name": tmpl_name,
                                            "tmpl_cfg": tmpl_cfg,
                                            "inj_layer": inj_layer,
                                            "shuffle_remap": shuffle_remap,
                                            "interp_question": {
                                                "question_text": question.get("question_text", ""),
                                                "options": dict(shuffled_options),
                                            },
                                        }

    # ── Phase 2: single-cell execution ────────────────────────────────────

    def _run_single_cell(
        self, *,
        record: PatchscopeRecord,
        condition: str,
        activation: torch.Tensor | None,
        model, tokenizer, device,
        interp_text: str,
        interp_messages: list[dict],
        placeholder_positions: list[int],
        prompt_style: str,
        tmpl_name: str,
        tmpl_cfg: dict,
        inj_layer: int,
        injection_mode: str,
        injection_alpha: float,
        gen_cfg: dict,
        relevancy_cfg: dict,
        all_choice_token_ids: dict,
        save_logprobs: bool,
        shuffle_remap: dict,
        baseline_cache: dict,
        interp_question: dict,
        base_prompt: str,
        placeholder_token: str,
        num_placeholders: int,
    ) -> None:

        """Inject an activation and decode the reporter's response for one matrix cell.

        This is the core of Phase 2.  The activation has already been selected
        by _iter_matrix_cells based on the condition:
          - "real":               actual source activation (tensor)
          - "shuffled":           mismatched activation from different question (tensor)
          - "text_only_baseline": None (no injection, prompt-only)

        Steps:
          1. Build the effective prompt (strip placeholder for baseline)
          2. Check baseline cache (avoid redundant forward passes)
          3. Run injection + decode (constrained logits or free-form generation)
          4. Populate the record with results
        """

        # ── 1. Build prompt for this condition ──────────────────
        # identity-style prompts are plain completion strings (no chat template); they must
        # be passed via raw_text so patch_and_decode matches build_interpretation_prompt.
        raw_text = interp_text if (prompt_style == "identity" or not self._use_chat_template) else None

        # For text_only_baseline: strip everything from the first placeholder
        # token onward, so the model predicts the next token from context alone.
        # We locate the placeholder by decoding the token at the first
        # placeholder_position rather than hardcoding "?".
        baseline_prompt = None
        if condition == "text_only_baseline" and placeholder_positions:
            interp_ids = tokenizer.encode(interp_text, add_special_tokens=False)
            first_ph_idx = placeholder_positions[0]
            # Sum decoded token lengths up to the placeholder to find char offset
            char_offset = sum(
                len(tokenizer.decode([interp_ids[i]]))
                for i in range(min(first_ph_idx, len(interp_ids)))
            )
            if char_offset > 0:
                baseline_prompt = interp_text[:char_offset].rstrip()
                if not baseline_prompt.endswith(" "):
                    baseline_prompt += " "

        record.interpretation_prompt = (
            baseline_prompt if (condition == "text_only_baseline" and baseline_prompt)
            else interp_text
        )

        # ── 2. Check baseline cache ──────────────────────────────────────
        # text_only_baseline runs the model with NO activation injected, so the
        # output depends only on (question, reporter, template) — not on which
        # source persona or layer the activation came from.  Without caching,
        # the same no-injection forward pass would run once per source_persona x
        # source_layer combination, producing identical results each time.
        # The cache stores the first result (decode_mode, generated_text,
        # parsed_answer, choice_probs, etc.) and copies it to subsequent records.

        baseline_key = (record.question_id, record.reporter_persona, tmpl_name, record.source_persona)
        if condition == "text_only_baseline" and baseline_key in baseline_cache:
            cached = baseline_cache[baseline_key]
            record.decode_mode = cached["decode_mode"]
            record.generated_text = cached["generated_text"]
            record.reporter_parsed_answer = cached.get("reporter_parsed_answer")
            record.parse_success = cached.get("parse_success", False)
            record.choice_probs = cached.get("choice_probs")
            record.choice_logits = cached.get("choice_logits")
            record.choice_logprobs = cached.get("choice_logprobs")
            record.total_choice_prob = cached.get("total_choice_prob")
            record.predicted = cached.get("predicted")
            if tmpl_name == "answer_extraction" and record.source_direct_answer:
                record.is_correct = record.predicted == record.source_direct_answer
            return

        # ── 3. Patch activation + decode ─────────────────────────────────
        #
        # Validate: activation=None is only valid for text_only_baseline.
        # Any other condition with a missing activation is a silent bug.
        if activation is None and condition != "text_only_baseline":
            raise ValueError(
                f"activation is None for condition={condition!r} "
                f"(question={record.question_id}, source={record.source_persona}). "
                f"Only text_only_baseline should have no activation."
            )

        decode_mode = tmpl_cfg.get("decode_mode", "generate")
        use_logits = decode_mode == "logits" and tmpl_name in all_choice_token_ids
        effective_raw = baseline_prompt if condition == "text_only_baseline" else raw_text

        # Single call to patch_and_decode handles both logits and generate modes
        result = patchscope_patching.patch_and_decode(
            model, tokenizer, device,
            interp_messages, activation,
            injection_layer=inj_layer,
            placeholder_positions=placeholder_positions,
            mode=injection_mode,
            alpha=injection_alpha,
            raw_text=effective_raw,
            decode_mode="logits" if use_logits else "generate",
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            do_sample=gen_cfg.get("do_sample", False),
            use_cache=gen_cfg.get("use_cache", False),
            choice_token_ids=all_choice_token_ids.get(tmpl_name),
            save_logprobs=save_logprobs,
        )

        # Populate the record from the result
        record.decode_mode = "logits" if use_logits else "generate"
        record.generated_text = result["generated_text"]

        if use_logits:
            # Remap shuffled labels back to canonical for answer_extraction
            if shuffle_remap and tmpl_name == "answer_extraction":
                record.choice_probs = {shuffle_remap.get(k, k): v for k, v in result["probs"].items()}
                record.choice_logits = {shuffle_remap.get(k, k): v for k, v in result["logits"].items()}
                record.choice_logprobs = (
                    {shuffle_remap.get(k, k): v for k, v in result["logprobs"].items()}
                    if "logprobs" in result else None
                )
                record.predicted = shuffle_remap.get(result["predicted"], result["predicted"])
            else:
                record.choice_probs = result["probs"]
                record.choice_logits = result["logits"]
                record.choice_logprobs = result.get("logprobs")
                record.predicted = result["predicted"]
            record.total_choice_prob = result.get("total_choice_prob")
            record.reporter_parsed_answer = record.predicted
            record.parse_success = True
            if tmpl_name == "answer_extraction" and record.source_direct_answer:
                record.is_correct = record.predicted == record.source_direct_answer
        else:
            # Free-form generation: parse structured answer from text
            if tmpl_name in patchscope_helpers.TEMPLATE_CHOICES:
                record.reporter_parsed_answer, record.parse_success = (
                    patchscope_helpers._parse_constrained(tmpl_name, result["generated_text"])
                )
            else:
                # Fallback: try to extract a single A/B/C/D letter from generated text.
                # Covers templates like open_summary where the model is asked to
                # repeat the activation content (often a single answer letter).
                _gen = result["generated_text"].strip()
                if _gen and _gen[0] in "ABCD":
                    record.reporter_parsed_answer = _gen[0]
                    record.parse_success = True

            # Remap shuffled letter back to canonical space so it can be
            # compared against source_direct_answer (which is already canonical).
            if shuffle_remap and record.reporter_parsed_answer:
                record.reporter_parsed_answer = shuffle_remap.get(
                    record.reporter_parsed_answer, record.reporter_parsed_answer
                )

            # Relevancy scoring (SelfIE metric, only for real condition)
            if (
                relevancy_cfg.get("enabled", False)
                and condition == "real"
                and activation is not None
            ):
                gen_token_ids = tokenizer.encode(result["generated_text"], add_special_tokens=False)
                if gen_token_ids:
                    rel_scores = patchscope_patching.compute_relevancy_scores(
                        model, tokenizer, device,
                        interp_messages, activation,
                        injection_layer=inj_layer,
                        placeholder_positions=placeholder_positions,
                        mode=injection_mode,
                        alpha=injection_alpha,
                        generated_token_ids=gen_token_ids,
                        max_tokens=relevancy_cfg.get("max_tokens", 64),
                        raw_text=interp_text if (prompt_style == "identity" or not self._use_chat_template) else None,
                    )
                    record.relevancy_scores = rel_scores
                    record.mean_relevancy = (
                        sum(rel_scores) / len(rel_scores) if rel_scores else None
                    )

        # ── 3b. Reporter confidence metrics ──────────────────────────────
        if record.choice_probs:
            _sorted_rp = sorted(record.choice_probs.values(), reverse=True)
            record.reporter_chosen_prob = _sorted_rp[0] if _sorted_rp else None
            record.reporter_margin = (_sorted_rp[0] - _sorted_rp[1]) if len(_sorted_rp) > 1 else None

        # ── 4. Cache baseline result ──────────────────────────────────────

        if condition == "text_only_baseline" and baseline_key not in baseline_cache:
            baseline_cache[baseline_key] = {
                "decode_mode": record.decode_mode,
                "generated_text": record.generated_text,
                "reporter_parsed_answer": record.reporter_parsed_answer,
                "parse_success": record.parse_success,
                "choice_probs": record.choice_probs,
                "choice_logits": record.choice_logits,
                "choice_logprobs": record.choice_logprobs,
                "total_choice_prob": record.total_choice_prob,
                "predicted": record.predicted,
            }

        # .txt log: one match + one oppose sample per (template, condition, layer pair)
        rep_cfg = self.ps_config.get("reporting") or {}
        oppose_pol = rep_cfg.get("opposing_sample_policy", "cross_ideology")
        layer_tag = f"L{record.source_layer}to{inj_layer}"

        def _try_reporter_sample(persona_alignment: str) -> None:
            sample_key = f"{tmpl_name}_{condition}_{layer_tag}_{persona_alignment}"
            if sample_key in self._sample_prompts:
                return
            self._sample_prompts[sample_key] = {
                "template": tmpl_name,
                "condition": condition,
                "persona_alignment": persona_alignment,
                "source_persona": record.source_persona,
                "reporter_persona": record.reporter_persona,
                "source_layer": record.source_layer,
                "injection_layer": inj_layer,
                "interp_prompt_text": interp_text,
                "generated_text": record.generated_text,
                "question_id": record.question_id,
            }
            if rep_cfg.get("include_no_reporter_system_sample", True) and (
                record.reporter_system_prompt or ""
            ).strip():
                try:
                    self._capture_no_system_log_sample(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        record=record,
                        condition=condition,
                        activation=activation,
                        tmpl_name=tmpl_name,
                        tmpl_cfg=tmpl_cfg,
                        inj_layer=inj_layer,
                        injection_mode=injection_mode,
                        injection_alpha=injection_alpha,
                        gen_cfg=gen_cfg,
                        all_choice_token_ids=all_choice_token_ids,
                        save_logprobs=save_logprobs,
                        interp_question=interp_question,
                        base_prompt=base_prompt,
                        placeholder_token=placeholder_token,
                        num_placeholders=num_placeholders,
                        prompt_style=prompt_style,
                        sample_key=sample_key,
                    )
                except Exception as e:
                    logger.warning(
                        f"No-system log sample failed for {sample_key}: {e}"
                    )

        if record.source_persona == record.reporter_persona:
            _try_reporter_sample("match")
        if patchscope_helpers.reporter_sample_opposing_qualifies(
            record.source_persona, record.reporter_persona, oppose_pol
        ):
            _try_reporter_sample("oppose")

        body_ov = (rep_cfg.get("no_persona_layer_log_body") or "").strip()
        if (
            rep_cfg.get("include_no_persona_chat_template_sample_per_layer", False)
            and condition == "real"
            and (prompt_style != "identity" or body_ov)
            and layer_tag not in self._no_persona_chat_per_layer
            and activation is not None
        ):
            try:
                self._capture_no_persona_plain_prompt_per_layer(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    layer_tag=layer_tag,
                    record=record,
                    activation=activation,
                    tmpl_name=tmpl_name,
                    tmpl_cfg=tmpl_cfg,
                    inj_layer=inj_layer,
                    injection_mode=injection_mode,
                    injection_alpha=injection_alpha,
                    gen_cfg=gen_cfg,
                    all_choice_token_ids=all_choice_token_ids,
                    save_logprobs=save_logprobs,
                    interp_question=interp_question,
                    base_prompt=base_prompt,
                    placeholder_token=placeholder_token,
                    num_placeholders=num_placeholders,
                    prompt_style=prompt_style,
                    layer_log_system_prompt=(
                        (rep_cfg.get("no_persona_layer_log_system_prompt") or "").strip() or None
                    ),
                    layer_log_body_override=(
                        (rep_cfg.get("no_persona_layer_log_body") or "").strip() or None
                    ),
                )
            except Exception as e:
                logger.warning(
                    f"No-persona layer log sample failed for {layer_tag}: {e}"
                )

    def _capture_no_persona_plain_prompt_per_layer(
        self,
        *,
        model,
        tokenizer,
        device,
        layer_tag: str,
        record,
        activation,
        tmpl_name: str,
        tmpl_cfg: dict,
        inj_layer: int,
        injection_mode: str,
        injection_alpha: float,
        gen_cfg: dict,
        all_choice_token_ids: dict,
        save_logprobs: bool,
        interp_question: dict,
        base_prompt: str,
        placeholder_token: str,
        num_placeholders: int,
        prompt_style: str,
        layer_log_system_prompt: str | None = None,
        layer_log_body_override: str | None = None,
    ) -> None:
        """One .txt log decode per layer pair: exact plain string sent to the model (no apply_chat_template).

        If ``reporting.no_persona_layer_log_body`` is set, it replaces the normal template body
        for this decode only (still supports {placeholder}, {question_text}, …). Optional
        ``no_persona_layer_log_system_prompt`` prepends plain text; omit both for a bare line.
        """
        tmpl_cfg_use = dict(tmpl_cfg)
        if layer_log_body_override:
            tmpl_cfg_use[prompt_style] = layer_log_body_override

        ns_text, ns_msgs, ns_ph = patchscope_helpers.build_interpretation_prompt(
            tokenizer=tokenizer,
            tmpl_cfg=tmpl_cfg_use,
            prompt_style=prompt_style,
            base_prompt=base_prompt,
            placeholder_token=placeholder_token,
            num_placeholders=num_placeholders,
            question=interp_question,
            reporter_system_prompt=layer_log_system_prompt,
            use_chat_template=False,
        )
        decode_mode = tmpl_cfg.get("decode_mode", "generate")
        use_logits = decode_mode == "logits" and tmpl_name in all_choice_token_ids

        result = patchscope_patching.patch_and_decode(
            model,
            tokenizer,
            device,
            ns_msgs,
            activation,
            injection_layer=inj_layer,
            placeholder_positions=ns_ph,
            mode=injection_mode,
            alpha=injection_alpha,
            raw_text=ns_text,
            decode_mode="logits" if use_logits else "generate",
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            do_sample=gen_cfg.get("do_sample", False),
            use_cache=gen_cfg.get("use_cache", True),
            choice_token_ids=all_choice_token_ids.get(tmpl_name),
            save_logprobs=save_logprobs,
        )
        self._no_persona_chat_per_layer[layer_tag] = {
            "template": tmpl_name,
            "question_id": record.question_id,
            "source_persona": record.source_persona,
            "reporter_persona": record.reporter_persona,
            "source_layer": record.source_layer,
            "injection_layer": inj_layer,
            "layer_log_system_prompt": layer_log_system_prompt or "",
            "layer_log_body_override": layer_log_body_override or "",
            "interp_prompt_text": ns_text,
            "generated_text": result["generated_text"],
        }

    def _capture_no_system_log_sample(
        self,
        *,
        model,
        tokenizer,
        device,
        record,
        condition: str,
        activation,
        tmpl_name: str,
        tmpl_cfg: dict,
        inj_layer: int,
        injection_mode: str,
        injection_alpha: float,
        gen_cfg: dict,
        all_choice_token_ids: dict,
        save_logprobs: bool,
        interp_question: dict,
        base_prompt: str,
        placeholder_token: str,
        num_placeholders: int,
        prompt_style: str,
        sample_key: str,
    ) -> None:
        """One extra patch_and_decode without personas.yaml reporter system — .txt log only.

        When use_chat_template is false, dropping the reporter persona removes the long plain
        preamble; for text_only_baseline the prefix before the placeholder can collapse to
        nearly nothing and the model decodes garbage. We therefore (1) prepend
        reporting.no_persona_layer_log_system_prompt when set (same as per-layer log), and
        (2) if it is unset, force apply_chat_template for this log decode only on
        text_only_baseline so the prompt is not degenerate.
        """
        rep_cfg = self.ps_config.get("reporting") or {}
        synthetic = (rep_cfg.get("no_persona_layer_log_system_prompt") or "").strip() or None
        use_ct = self._use_chat_template
        if (
            synthetic is None
            and not use_ct
            and condition == "text_only_baseline"
            and prompt_style != "identity"
        ):
            use_ct = True

        ns_text, ns_msgs, ns_ph = patchscope_helpers.build_interpretation_prompt(
            tokenizer=tokenizer,
            tmpl_cfg=tmpl_cfg,
            prompt_style=prompt_style,
            base_prompt=base_prompt,
            placeholder_token=placeholder_token,
            num_placeholders=num_placeholders,
            question=interp_question,
            reporter_system_prompt=synthetic,
            use_chat_template=use_ct,
        )
        effective_raw = None
        if condition == "text_only_baseline" and ns_ph:
            interp_ids = tokenizer.encode(ns_text, add_special_tokens=False)
            first_ph_idx = ns_ph[0]
            char_offset = sum(
                len(tokenizer.decode([interp_ids[i]]))
                for i in range(min(first_ph_idx, len(interp_ids)))
            )
            if char_offset > 0:
                bp = ns_text[:char_offset].rstrip()
                if not bp.endswith(" "):
                    bp += " "
                effective_raw = bp
        elif prompt_style == "identity" or not use_ct:
            effective_raw = ns_text

        decode_mode = tmpl_cfg.get("decode_mode", "generate")
        use_logits = decode_mode == "logits" and tmpl_name in all_choice_token_ids

        result = patchscope_patching.patch_and_decode(
            model,
            tokenizer,
            device,
            ns_msgs,
            activation,
            injection_layer=inj_layer,
            placeholder_positions=ns_ph,
            mode=injection_mode,
            alpha=injection_alpha,
            raw_text=effective_raw,
            decode_mode="logits" if use_logits else "generate",
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            do_sample=gen_cfg.get("do_sample", False),
            use_cache=gen_cfg.get("use_cache", True),
            choice_token_ids=all_choice_token_ids.get(tmpl_name),
            save_logprobs=save_logprobs,
        )
        log_prompt = (
            effective_raw
            if (condition == "text_only_baseline" and effective_raw)
            else ns_text
        )
        self._sample_prompts[sample_key]["no_reporter_system"] = {
            "interp_prompt_text": log_prompt,
            "generated_text": result["generated_text"],
        }

    # ── Evaluate & save ───────────────────────────────────────────────────

    def evaluate(self) -> dict:
        """Compute aggregate metrics from collected records."""
        return patchscope_evaluation.compute_metrics(self.records)

    def save_results(self) -> str:
        """Final save — run() already flushes incrementally, this is the last write."""
        self._init_output_paths()
        self._flush_results()
        return str(self._jsonl_path)

    def _write_log(self, log_path: Path, base_name: str) -> None:
        patchscope_reporting.write_run_log(
            log_path=log_path,
            base_name=base_name,
            records=self.records,
            ps_config=self.ps_config,
            run_config=self.config,
            backend=self.backend,
            all_personas=self.all_personas,
            sample_source_prompts=getattr(self, '_sample_source_prompts', {}),
            sample_prompts=self._sample_prompts,
            no_persona_chat_per_layer=self._no_persona_chat_per_layer,
            errors=self._errors,
            elapsed=getattr(self, '_run_elapsed', 0.0),
            n_questions=len(self.questions),
            evaluate_fn=self.evaluate,
        )
