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
        self._errors: list[str] = []
        self._run_elapsed: float = 0.0
        self._jsonl_path: Optional[Path] = None
        self._log_path: Optional[Path] = None
        self._base_name: str = ""
        self._last_flush: int = 0

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
        self._base_name = f"patchscope_{model_short}_{timestamp}_{detail}"
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
        activations, direct_answers, shuffle_maps = self._extract_sources(
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

        matrix_dims = {
            "source_personas": len(source_persona_names),
            "questions": len(self.questions),
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
    ) -> tuple[dict, dict, dict]:
        """Extract activations and direct answers for every (persona, question, layer).

        Returns:
            (activations, direct_answers, shuffle_maps)
        """
        activations: dict[str, dict[int, dict[int, torch.Tensor]]] = {}
        direct_answers: dict[str, dict[int, dict]] = {}
        shuffle_maps: dict[tuple[str, str], tuple[dict, dict[str, str]]] = {}
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

                    _ex_tok = extraction_cfg.get("token_position", "last")
                    _ex_boundary = extraction_cfg.get("assistant_boundary_marker")

                    # Validate extraction position once (first question of first persona)
                    if not hasattr(self, '_extraction_validated'):
                        source_text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        patchscope_helpers.validate_extraction_position(
                            tokenizer, source_text,
                            _ex_tok,
                            tmpl_name=f"source_extraction[{sp_name}]",
                            boundary_marker=_ex_boundary,
                        )
                        if isinstance(_ex_tok, str) and _ex_tok.strip().lower().replace(
                            "-", "_"
                        ) in ("last_before_assistant", "before_assistant"):
                            logger.info(
                                "extraction.token_position is last_before_assistant: hidden state is "
                                "read at the end of the user turn (before assistant scaffolding). "
                                "Source direct-answer logits (A/B/C/D) still use the full templated "
                                "prefix through the last token — comparable to extraction only if you "
                                "interpret them as different readout sites."
                            )
                        self._extraction_validated = True

                    activations[sp_name][question_idx] = patchscope_patching.extract_activations_multi_layer(
                        model, tokenizer, device, messages,
                        layer_indices=source_layers,
                        token_position=_ex_tok,
                        boundary_marker=_ex_boundary,
                    )

                    # Direct answer via logits
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

                    direct_answers[sp_name][question_idx] = {
                        "answer": canonical_answer,
                        "probs": canonical_probs,
                        "logits": canonical_logits,
                        "logprobs": canonical_logprobs,
                        "total_choice_prob": total_cp,
                        "raw": f"[logits] {canonical_answer}",
                        "shuffled_answer": shuffled_answer,
                        "option_order": [remap.get(l, l) for l in ["A", "B", "C", "D"]],
                    }

                    if sp_name not in self._sample_source_prompts:
                        _ptext = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        self._sample_source_prompts[sp_name] = {
                            "persona": sp_name,
                            "question_id": question_id,
                            "prompt_text": _ptext,
                            "extraction_site": patchscope_helpers.describe_source_extraction_site(
                                tokenizer,
                                _ptext,
                                extraction_cfg.get("token_position", "last"),
                                boundary_marker=extraction_cfg.get(
                                    "assistant_boundary_marker"
                                ),
                            ),
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

        return activations, direct_answers, shuffle_maps

    # ── Phase 2: matrix iteration ─────────────────────────────────────────

    def _iter_matrix_cells(
        self, *,
        source_persona_names, source_layers, injection_layers, pair_map,
        reporter_persona_names, templates, conditions,
        activations, direct_answers, shuffle_maps,
        control_sample_rate, control_rng,
        style, tokenizer, base_prompt, placeholder_token, num_placeholders,
        injection_mode,
    ) -> Iterator[dict]:
        """Yield one dict per cell in the Phase 2 experiment matrix.

        This generator owns the 7-level nested loop so that run() stays flat.
        Each yielded dict contains everything _run_single_cell needs:
          - "record": pre-populated PatchscopeRecord
          - "activation": tensor to inject (real, shuffled, or None for baseline)
          - "condition": which condition this cell is for
          - "interp_text" / "interp_messages" / "placeholder_positions": built prompt
          - "tmpl_name" / "tmpl_cfg" / "prompt_style": template info
          - "inj_layer": injection layer index
          - "shuffle_remap": label remap dict for answer remapping
        """
        for sp_name in source_persona_names:
            for question_idx, question in enumerate(self.questions):
                question_id = question.get("question_id", f"q{question_idx}")

                if not activations.get(sp_name, {}).get(question_idx, {}):
                    logger.warning(f"Skipping {sp_name} q={question_id}: no activations")
                    continue

                for src_layer in source_layers:
                    real_act = activations.get(sp_name, {}).get(question_idx, {}).get(src_layer)
                    if real_act is None:
                        continue

                    # Shuffled control: activation from a different question
                    # (mismatched-content control, NOT random noise)
                    shuffled_act = None
                    if "shuffled" in conditions and len(self.questions) > 1:
                        other_question_idx = (question_idx + 1) % len(self.questions)
                        shuffled_act = activations.get(sp_name, {}).get(other_question_idx, {}).get(src_layer)

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
                                        source_answer_probs=source_direct["probs"],
                                        question_text=question.get("question_text", ""),
                                        question_options=question.get("options"),
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
        raw_text = None  # never bypass chat template — system prompt must always be used

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
            record.parsed_answer = cached.get("parsed_answer")
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
            use_cache=gen_cfg.get("use_cache", True),
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
            record.parsed_answer = record.predicted
            record.parse_success = True
            if tmpl_name == "answer_extraction" and record.source_direct_answer:
                record.is_correct = record.predicted == record.source_direct_answer
        else:
            # Free-form generation: optionally parse structured answer from text
            if tmpl_name in patchscope_helpers.TEMPLATE_CHOICES:
                record.parsed_answer, record.parse_success = (
                    patchscope_helpers._parse_constrained(tmpl_name, result["generated_text"])
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
                    )
                    record.relevancy_scores = rel_scores
                    record.mean_relevancy = (
                        sum(rel_scores) / len(rel_scores) if rel_scores else None
                    )

        # ── 4. Cache baseline result ──────────────────────────────────────

        if condition == "text_only_baseline" and baseline_key not in baseline_cache:
            baseline_cache[baseline_key] = {
                "decode_mode": record.decode_mode,
                "generated_text": record.generated_text,
                "parsed_answer": record.parsed_answer,
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
        """One extra patch_and_decode with no reporter system — .txt log only."""
        ns_text, ns_msgs, ns_ph = patchscope_helpers.build_interpretation_prompt(
            tokenizer=tokenizer,
            tmpl_cfg=tmpl_cfg,
            prompt_style=prompt_style,
            base_prompt=base_prompt,
            placeholder_token=placeholder_token,
            num_placeholders=num_placeholders,
            question=interp_question,
            reporter_system_prompt=None,
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
            errors=self._errors,
            elapsed=getattr(self, '_run_elapsed', 0.0),
            n_questions=len(self.questions),
            evaluate_fn=self.evaluate,
        )
