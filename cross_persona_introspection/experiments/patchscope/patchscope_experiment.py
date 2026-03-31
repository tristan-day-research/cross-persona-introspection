"""Patchscope activation-interpretation experiment.

This module contains only the experiment orchestration logic: setup, the
two-phase run loop, incremental checkpointing, and result saving.  All
lower-level concerns live in sibling modules:

- **patchscope_patching** — the four activation-patching primitives
  (extract, inject+generate, inject+logits, relevancy scores).
- **patchscope_helpers** — config/data loading, prompt formatting, answer
  parsing, placeholder resolution, interpretation-prompt construction.
- **patchscope_evaluation** — aggregate metric computation.
- **patchscope_source_overrides** — Phase 3 raw-text override trials.
- **patchscope_reporting** — companion .txt log writer.

The experiment implements the Patchscopes protocol (Ghandeharioun et al.,
ICML 2024, arXiv:2401.06102): extract a hidden-state activation from a
*source* forward pass and inject it into a separate *interpretation* forward
pass to decode what the model's internal representation encodes.

The full experiment matrix is::

    source_persona × question × source_layer × injection_layer
        × reporter_persona × template × condition

where *condition* ∈ {real, text_only_baseline, shuffled}.

Phase 1 pre-computes all source activations and direct answers so that
Phase 2 can sweep reporters/templates/conditions without redundant forward
passes.  Phase 3 (optional) runs source overrides for raw-text sanity checks.
"""

import json
import logging
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from cross_persona_introspection.experiments.base import BaseExperiment
from cross_persona_introspection.experiments.patchscope.patchscope_evaluation import (
    compute_metrics,
)
from cross_persona_introspection.experiments.patchscope.patchscope_helpers import (
    TEMPLATE_CHOICES,
    _format_question_for_source,
    _get_placeholder_token_id,
    _load_patchscope_config,
    _load_questions,
    _model_short_name,
    _parse_constrained,
    _resolve_choice_token_ids,
    _resolve_layers,
    _shuffle_options,
    build_interpretation_prompt,
)
from cross_persona_introspection.experiments.patchscope.patchscope_patching import (
    compute_relevancy_scores,
    extract_activation,
    inject_and_extract_logits,
    inject_and_generate,
)
from cross_persona_introspection.experiments.patchscope.patchscope_reporting import (
    write_run_log,
)
from cross_persona_introspection.experiments.patchscope.patchscope_source_overrides import (
    run_source_overrides,
)
from cross_persona_introspection.schemas import (
    PatchscopeRecord,
    PersonaConfig,
    RunConfig,
)

logger = logging.getLogger(__name__)


class PatchscopeExperiment(BaseExperiment):
    """Orchestrates the full Patchscope activation-interpretation experiment.

    Lifecycle: ``setup()`` → ``run()`` → ``evaluate()`` → ``save_results()``.
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
        self.ps_config = _load_patchscope_config(self.config.patchscope_config)

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
        self.questions = _load_questions(
            task_file,
            sample_size=self.config.sample_size,
            seed=self.config.seed,
            categories=self.ps_config.get("categories") or None,
            samples_per_category=self.ps_config.get("samples_per_category"),
        )
        logger.info(f"Loaded {len(self.questions)} questions from {task_file}")

    # ── Output / checkpointing ────────────────────────────────────────────

    def _init_output_paths(self) -> None:
        if self._jsonl_path is not None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = _model_short_name(self.config.model_name)
        exp_type = (
            "layer_sweep"
            if self.ps_config.get("layer_sweep", {}).get("enabled", False)
            else "matrix"
        )
        self._base_name = f"patchscope_{model_short}_{timestamp}_{exp_type}"
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
        """Execute the full experiment: Phase 1 (extract) → Phase 2 (interpret) → Phase 3 (overrides)."""
        assert self.backend is not None
        model = self.backend.model
        tokenizer = self.backend.tokenizer
        device = self.backend.input_device
        ps = self.ps_config

        self._init_output_paths()
        run_start = time.monotonic()

        # Unpack config sections
        extraction_cfg = ps["extraction"]
        injection_cfg = ps["injection"]
        gen_cfg = ps["generation"]
        relevancy_cfg = ps["relevancy"]
        controls_cfg = ps["controls"]
        templates = ps["interpretation_templates"]
        save_logprobs = gen_cfg.get("save_logprobs", False)

        # Filter templates
        enabled = ps.get("enabled_templates", [])
        if enabled:
            templates = {k: v for k, v in templates.items() if k in enabled}
            logger.info(f"Template filter: running only {list(templates.keys())}")
        else:
            logger.info(f"Running all templates: {list(templates.keys())}")

        base_prompt = ps["interpretation_base_prompt"]
        source_persona_names = ps["source_personas"]
        reporter_persona_names = ps["reporter_personas"]

        # Resolve layer pairs
        source_layers, injection_layers, _pair_map = self._resolve_layer_config(
            ps, extraction_cfg, injection_cfg, model.config.num_hidden_layers
        )

        injection_mode = injection_cfg["mode"]
        injection_alpha = float(injection_cfg["alpha"])

        # Resolve prompt style
        style = ps.get("prompt_style") or injection_cfg.get("placeholder_style", "patchscopes")
        configured_placeholder = injection_cfg.get("placeholder_token", "?")
        if style in ("patchscopes", "identity"):
            num_placeholders = 1
        elif style == "selfie":
            num_placeholders = 5
        else:
            num_placeholders = int(injection_cfg["num_placeholders"])

        placeholder_token_id = _get_placeholder_token_id(tokenizer, configured_placeholder)
        placeholder_token = tokenizer.decode([placeholder_token_id])

        control_sample_rate = float(controls_cfg.get("control_sample_rate", 1.0))
        control_rng = random.Random(self.config.seed + 7)

        # Pre-resolve choice token IDs for logits-mode templates
        all_choice_token_ids: dict[str, dict[str, int]] = {}
        for tmpl_name, tmpl_cfg in templates.items():
            if tmpl_cfg.get("decode_mode", "generate") == "logits" and tmpl_name in TEMPLATE_CHOICES:
                all_choice_token_ids[tmpl_name] = _resolve_choice_token_ids(
                    tokenizer, TEMPLATE_CHOICES[tmpl_name]
                )
                logger.info(f"Logits mode — {tmpl_name} choices: {all_choice_token_ids[tmpl_name]}")
        logger.info(
            f"Placeholder token: id={placeholder_token_id} repr={repr(placeholder_token)} "
            f"× {num_placeholders}"
        )

        # ── Phase 1: extract source activations & direct answers ──────────
        logger.info("=== Phase 1: Extracting source activations ===")
        activations, direct_answers, shuffle_maps = self._extract_sources(
            model, tokenizer, device,
            source_persona_names, source_layers,
            extraction_cfg, save_logprobs,
        )

        # ── Phase 2: interpretation matrix ────────────────────────────────
        logger.info("=== Phase 2: Running interpretation matrix ===")

        conditions = ["real"]
        if controls_cfg.get("text_only_baseline", False):
            conditions.append("text_only_baseline")
        if controls_cfg.get("shuffled_activation", False):
            conditions.append("shuffled")

        total_cells = (
            len(source_persona_names) * len(self.questions) * len(source_layers)
            * len(injection_layers) * len(reporter_persona_names)
            * len(templates) * len(conditions)
        )
        logger.info(f"Total matrix cells: {total_cells}")

        _baseline_cache: dict[tuple, dict] = {}
        cell_count = 0

        for sp_name in source_persona_names:
            for qi, question in enumerate(self.questions):
                qid = question.get("question_id", f"q{qi}")

                if not activations.get(sp_name, {}).get(qi, {}):
                    logger.warning(f"Skipping {sp_name} q={qid}: extraction produced no activations")
                    continue

                for src_layer in source_layers:
                    real_act = activations.get(sp_name, {}).get(qi, {}).get(src_layer)
                    if real_act is None:
                        logger.warning(f"Skipping {sp_name} q={qid} layer={src_layer}: no activation")
                        continue

                    # Shuffled control: activation from a different question
                    shuffled_act = None
                    if "shuffled" in conditions and len(self.questions) > 1:
                        other_qi = (qi + 1) % len(self.questions)
                        shuffled_act = activations.get(sp_name, {}).get(other_qi, {}).get(src_layer)

                    _inj_layers = _pair_map[src_layer] if _pair_map else injection_layers
                    for inj_layer in _inj_layers:
                        for reporter_name in reporter_persona_names:
                            reporter_persona_obj = self.all_personas[reporter_name]

                            for tmpl_name, tmpl_cfg in templates.items():
                                prompt_style = style if style in ("patchscopes", "selfie", "identity") else "selfie"

                                # Use shuffled options matching the source pass
                                _shuffle_key = (qid, sp_name)
                                _q_for_template = shuffle_maps.get(_shuffle_key, (question, {}))[0]
                                _opts = _q_for_template.get("options", {})

                                interp_text, interp_messages, placeholder_positions = (
                                    build_interpretation_prompt(
                                        tokenizer=tokenizer,
                                        tmpl_cfg=tmpl_cfg,
                                        prompt_style=prompt_style,
                                        base_prompt=base_prompt,
                                        placeholder_token=placeholder_token,
                                        num_placeholders=num_placeholders,
                                        question=question,
                                        reporter_system_prompt=reporter_persona_obj.system_prompt,
                                        options_override=_opts,
                                    )
                                )

                                for condition in conditions:
                                    if condition != "real" and control_sample_rate < 1.0:
                                        if control_rng.random() > control_sample_rate:
                                            continue
                                    cell_count += 1

                                    _da = direct_answers[sp_name][qi]
                                    record = PatchscopeRecord(
                                        experiment="patchscope",
                                        template_name=tmpl_name,
                                        model=self.config.model_name,
                                        question_id=qid,
                                        source_persona=sp_name,
                                        reporter_persona=reporter_name,
                                        condition=condition,
                                        source_layer=src_layer,
                                        injection_layer=inj_layer,
                                        injection_mode=injection_mode,
                                        source_direct_answer=_da["answer"],
                                        source_answer_probs=_da["probs"],
                                        question_text=question.get("question_text", ""),
                                        question_options=question.get("options"),
                                    )

                                    try:
                                        self._run_single_cell(
                                            record=record,
                                            condition=condition,
                                            real_act=real_act,
                                            shuffled_act=shuffled_act,
                                            model=model,
                                            tokenizer=tokenizer,
                                            device=device,
                                            interp_text=interp_text,
                                            interp_messages=interp_messages,
                                            placeholder_positions=placeholder_positions,
                                            prompt_style=prompt_style,
                                            tmpl_name=tmpl_name,
                                            tmpl_cfg=tmpl_cfg,
                                            inj_layer=inj_layer,
                                            injection_mode=injection_mode,
                                            injection_alpha=injection_alpha,
                                            gen_cfg=gen_cfg,
                                            relevancy_cfg=relevancy_cfg,
                                            all_choice_token_ids=all_choice_token_ids,
                                            save_logprobs=save_logprobs,
                                            reporter_persona_obj=reporter_persona_obj,
                                            shuffle_maps=shuffle_maps,
                                            sp_name=sp_name,
                                            qid=qid,
                                            _baseline_cache=_baseline_cache,
                                            reporter_name=reporter_name,
                                            src_layer=src_layer,
                                        )
                                    except Exception as e:
                                        msg = (
                                            f"Interpret error: {sp_name}->{reporter_name} "
                                            f"{tmpl_name} {condition} q={qid}: {e}"
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

        # ── Phase 3: source overrides ─────────────────────────────────────
        source_override_list = ps.get("source_overrides", [])
        if source_override_list:
            run_source_overrides(
                source_overrides=source_override_list,
                model=model,
                tokenizer=tokenizer,
                device=device,
                source_layers=source_layers,
                injection_layers=injection_layers,
                pair_map=_pair_map,
                templates=templates,
                prompt_style=style if style in ("patchscopes", "selfie", "identity") else "selfie",
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

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_layer_config(
        ps: dict, extraction_cfg: dict, injection_cfg: dict, num_layers: int,
    ) -> tuple[list[int], list[int], dict[int, list[int]] | None]:
        """Parse layer_pairs / layer_sweep / simple layer config.

        Returns (source_layers, injection_layers, pair_map_or_None).
        """
        explicit_pairs = ps.get("layer_pairs")
        if explicit_pairs:
            source_layers = sorted(set(int(p[0]) for p in explicit_pairs))
            injection_layers = sorted(set(int(p[1]) for p in explicit_pairs))
            pair_map: dict[int, list[int]] = {}
            for p in explicit_pairs:
                pair_map.setdefault(int(p[0]), []).append(int(p[1]))
            logger.info(f"Using explicit layer_pairs: {explicit_pairs}")
            return source_layers, injection_layers, pair_map

        if ps.get("layer_sweep", {}).get("enabled", False):
            sweep = ps["layer_sweep"]
            source_layers = [int(x) for x in sweep["source_layers"]]
            injection_layers = [int(x) for x in sweep["injection_layers"]]
            return source_layers, injection_layers, None

        source_layers = _resolve_layers(extraction_cfg["layers"], num_layers)
        injection_layers = [int(injection_cfg["layer"])]
        return source_layers, injection_layers, None

    def _extract_sources(
        self, model, tokenizer, device,
        source_persona_names, source_layers,
        extraction_cfg, save_logprobs,
    ) -> tuple[dict, dict, dict]:
        """Phase 1: extract activations and direct answers for every (persona, question, layer).

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
            sp = self.all_personas[sp_name]

            for qi, question in enumerate(tqdm(
                self.questions, desc=f"  Extract [{sp_name}]", leave=True
            )):
                try:
                    qid = question.get("question_id", f"q{qi}")

                    shuffled_q, remap = _shuffle_options(question, self.config.seed, qid, sp_name)
                    shuffle_maps[(qid, sp_name)] = (shuffled_q, remap)

                    user_msg = _format_question_for_source(shuffled_q)
                    messages = []
                    if sp.system_prompt:
                        messages.append({"role": "system", "content": sp.system_prompt})
                    messages.append({"role": "user", "content": user_msg})

                    activations[sp_name][qi] = {}
                    for layer in source_layers:
                        act = extract_activation(
                            model, tokenizer, device, messages,
                            layer_idx=layer,
                            token_position=extraction_cfg["token_position"],
                        )
                        activations[sp_name][qi][layer] = act

                    # Direct answer via logits
                    _source_result = self.backend.get_choice_probs_and_logits(
                        messages, ["A", "B", "C", "D"], save_logprobs=save_logprobs,
                    )
                    if save_logprobs:
                        probs, logits_dict, logprobs_dict, total_cp = _source_result
                    else:
                        probs, logits_dict = _source_result
                        logprobs_dict, total_cp = None, None

                    shuffled_answer = max(probs, key=probs.get)
                    canonical_answer = remap.get(shuffled_answer, shuffled_answer)
                    canonical_probs = {remap.get(k, k): v for k, v in probs.items()}
                    canonical_logits = {remap.get(k, k): v for k, v in logits_dict.items()}
                    canonical_logprobs = (
                        {remap.get(k, k): v for k, v in logprobs_dict.items()}
                        if logprobs_dict else None
                    )

                    direct_answers[sp_name][qi] = {
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
                        self._sample_source_prompts[sp_name] = {
                            "persona": sp_name,
                            "question_id": qid,
                            "prompt_text": tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            ),
                            "answer": canonical_answer,
                            "shuffled_answer": shuffled_answer,
                            "option_order": [remap.get(l, l) for l in ["A", "B", "C", "D"]],
                        }

                except Exception as e:
                    msg = f"Extract error: {sp_name} q{qi} {question.get('question_id', '?')}: {e}"
                    logger.error(msg)
                    self._errors.append(msg)
                    activations.setdefault(sp_name, {})[qi] = {}
                    direct_answers.setdefault(sp_name, {})[qi] = {
                        "answer": None, "probs": {}, "logits": {}, "raw": ""
                    }

        return activations, direct_answers, shuffle_maps

    def _run_single_cell(
        self, *,
        record: PatchscopeRecord,
        condition: str,
        real_act: torch.Tensor,
        shuffled_act: torch.Tensor | None,
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
        reporter_persona_obj,
        shuffle_maps: dict,
        sp_name: str,
        qid: str,
        _baseline_cache: dict,
        reporter_name: str,
        src_layer: int,
    ) -> None:
        """Process one (condition × template × reporter × layer) cell of the matrix."""
        # Pick activation
        if condition == "text_only_baseline":
            act_for_condition = None
        elif condition == "shuffled":
            if shuffled_act is None:
                record.error = "no shuffled activation available"
                return
            act_for_condition = shuffled_act
        else:
            act_for_condition = real_act

        record.reporter_system_prompt = reporter_persona_obj.system_prompt or ""

        # Shuffle remap for answer remapping
        _shuffle_key = (qid, sp_name)
        _remap = shuffle_maps.get(_shuffle_key, (None, {}))[1]

        use_logits = (
            tmpl_cfg.get("decode_mode", "generate") == "logits"
            and tmpl_name in all_choice_token_ids
        )

        _raw = interp_text if prompt_style == "identity" else None

        # For text_only_baseline: strip placeholder from prompt
        _raw_baseline = None
        if condition == "text_only_baseline":
            _SENTINEL_POS = interp_text.find("?")
            if _SENTINEL_POS >= 0:
                _raw_baseline = interp_text[:_SENTINEL_POS].rstrip()
                if not _raw_baseline.endswith(" "):
                    _raw_baseline += " "

        if condition == "text_only_baseline" and _raw_baseline:
            record.interpretation_prompt = _raw_baseline
        else:
            record.interpretation_prompt = interp_text

        # Baseline deduplication
        _baseline_key = (qid, reporter_name, tmpl_name, sp_name)
        if condition == "text_only_baseline" and _baseline_key in _baseline_cache:
            cached = _baseline_cache[_baseline_key]
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

        if use_logits:
            self._run_logits_cell(
                record, model, tokenizer, device,
                interp_messages, act_for_condition,
                inj_layer, placeholder_positions,
                all_choice_token_ids[tmpl_name],
                injection_mode, injection_alpha,
                _raw_baseline if condition == "text_only_baseline" else _raw,
                save_logprobs, tmpl_name, _remap,
            )
        else:
            self._run_generate_cell(
                record, model, tokenizer, device,
                interp_messages, act_for_condition,
                inj_layer, placeholder_positions,
                injection_mode, injection_alpha,
                gen_cfg, _raw, tmpl_name,
                relevancy_cfg, real_act, condition,
            )

        # Cache baseline
        if condition == "text_only_baseline" and _baseline_key not in _baseline_cache:
            _baseline_cache[_baseline_key] = {
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

        # Capture sample prompt
        sample_key = f"{tmpl_name}_{condition}"
        if sample_key not in self._sample_prompts:
            self._sample_prompts[sample_key] = {
                "template": tmpl_name,
                "condition": condition,
                "source_persona": sp_name,
                "reporter_persona": reporter_name,
                "source_layer": src_layer,
                "injection_layer": inj_layer,
                "interp_prompt_text": interp_text,
                "generated_text": record.generated_text,
                "question_id": qid,
            }

    def _run_logits_cell(
        self, record, model, tokenizer, device,
        interp_messages, activation, inj_layer,
        placeholder_positions, choice_token_ids,
        injection_mode, injection_alpha, raw_text,
        save_logprobs, tmpl_name, remap,
    ) -> None:
        """Logits-mode cell: single forward pass, constrained next-token probs."""
        record.decode_mode = "logits"
        result = inject_and_extract_logits(
            model, tokenizer, device,
            interp_messages, activation,
            injection_layer=inj_layer,
            placeholder_positions=placeholder_positions,
            choice_token_ids=choice_token_ids,
            mode=injection_mode,
            alpha=injection_alpha,
            raw_text=raw_text,
            save_logprobs=save_logprobs,
        )

        if remap and tmpl_name == "answer_extraction":
            canonical_probs = {remap.get(k, k): v for k, v in result["probs"].items()}
            canonical_logits = {remap.get(k, k): v for k, v in result["logits"].items()}
            canonical_logprobs = (
                {remap.get(k, k): v for k, v in result["logprobs"].items()}
                if "logprobs" in result else None
            )
            canonical_predicted = remap.get(result["predicted"], result["predicted"])
        else:
            canonical_probs = result["probs"]
            canonical_logits = result["logits"]
            canonical_logprobs = result.get("logprobs")
            canonical_predicted = result["predicted"]

        record.choice_probs = canonical_probs
        record.choice_logits = canonical_logits
        record.choice_logprobs = canonical_logprobs
        record.total_choice_prob = result.get("total_choice_prob")
        record.predicted = canonical_predicted
        record.parsed_answer = canonical_predicted
        record.parse_success = True
        record.generated_text = f"[logits] {canonical_predicted}"

        if tmpl_name == "answer_extraction" and record.source_direct_answer:
            record.is_correct = canonical_predicted == record.source_direct_answer

    def _run_generate_cell(
        self, record, model, tokenizer, device,
        interp_messages, activation, inj_layer,
        placeholder_positions, injection_mode, injection_alpha,
        gen_cfg, raw_text, tmpl_name,
        relevancy_cfg, real_act, condition,
    ) -> None:
        """Generate-mode cell: auto-regressive generation + optional relevancy."""
        record.decode_mode = "generate"
        if activation is None:
            gen_text = self.backend.generate(
                interp_messages,
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                do_sample=gen_cfg.get("do_sample", False),
            )
        else:
            gen_text = inject_and_generate(
                model, tokenizer, device,
                interp_messages, activation,
                injection_layer=inj_layer,
                placeholder_positions=placeholder_positions,
                mode=injection_mode,
                alpha=injection_alpha,
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                do_sample=gen_cfg.get("do_sample", False),
                raw_text=raw_text,
            )
        record.generated_text = gen_text

        if tmpl_name in TEMPLATE_CHOICES:
            record.parsed_answer, record.parse_success = _parse_constrained(tmpl_name, gen_text)

        if (
            relevancy_cfg.get("enabled", False)
            and condition == "real"
            and real_act is not None
        ):
            gen_token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
            if gen_token_ids:
                rel_scores = compute_relevancy_scores(
                    model, tokenizer, device,
                    interp_messages, real_act,
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

    # ── Evaluate & save ───────────────────────────────────────────────────

    def evaluate(self) -> dict:
        """Compute aggregate metrics from collected records."""
        return compute_metrics(self.records)

    def save_results(self) -> str:
        """Final save — run() already flushes incrementally, this is the last write."""
        self._init_output_paths()
        self._flush_results()
        return str(self._jsonl_path)

    def _write_log(self, log_path: Path, base_name: str) -> None:
        write_run_log(
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
