"""Confidence vs Entropy experiment for base (non-instruct) models.

Adapted from confidence_entropy.py for base models that don't follow
instructions. Uses few-shot pattern completion instead of chat templates.

Key differences from the instruct version:
- No chat template — raw text prompts only
- Persona induction via text prefix, not system prompts
- Only 2 prompts per trial (no open-ended prompts):
    1. Forced-choice MC answer (few-shot) → logprob extraction
    2. Stated confidence (few-shot) → logprob extraction
- Open-ended prompts are skipped because base models produce unreliable
  free-form text without instruction following.
"""

import logging
import math
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from cross_persona_introspection.core.results_logger import ResultsLogger
from cross_persona_introspection.core.task_loader import sample_tasks
from cross_persona_introspection.experiments.base import BaseExperiment
from cross_persona_introspection.experiments.base_model_prompts import (
    format_confidence_prompt_base,
    format_mc_prompt_base,
    get_persona_context,
)
from cross_persona_introspection.experiments.confidence_entropy import (
    STATED_CONFIDENCE_MIDPOINTS,
    STATED_CONFIDENCE_OPTIONS,
    _format_options_block,
    _parse_confidence_letter,
    _parse_forced_answer,
)
from cross_persona_introspection.schemas import (
    ConfidenceEntropyRecord,
    PersonaConfig,
    RunConfig,
    TaskItem,
)

logger = logging.getLogger(__name__)


def _truncate_to_first_token(raw: str) -> str:
    """Truncate base model output to just the first line/token.

    Base models continue the few-shot pattern after the answer letter,
    producing output like "B\\n\\nQuestion: Which of...". We only care
    about the first non-whitespace character(s) before any newline.
    """
    # Strip leading whitespace, then take everything up to the first newline or space
    stripped = raw.lstrip()
    # Split on whitespace — first chunk is the answer token
    parts = stripped.split(None, 1)
    return parts[0] if parts else stripped


class ConfidenceEntropyBase(BaseExperiment):
    """Confidence vs Entropy experiment for base (non-instruct) models.

    Uses few-shot pattern completion prompts and raw text generation
    instead of chat templates and system prompts.
    """

    def __init__(self, config: RunConfig, personas: dict[str, PersonaConfig]):
        super().__init__(config)
        self.personas = personas
        self.backend = None
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = Path(config.output_dir) / (
            f"confidence_entropy_base_{self._run_timestamp}.jsonl"
        )
        self.manifest_path = Path(config.output_dir) / (
            f"confidence_entropy_base_{self._run_timestamp}_manifest.txt"
        )
        self.tasks: list[TaskItem] = []
        self.results_logger: ResultsLogger | None = None
        self._sample_logged = False
        self._warnings: list[str] = []

    def setup(self) -> None:
        """Load model and dataset."""
        from cross_persona_introspection.backends.hf_backend import HFBackend

        logger.info(f"Loading base model: {self.config.model_name}")
        self.backend = HFBackend(self.config.model_name)

        # Build or load cached dataset (same dataset as instruct version)
        from cross_persona_introspection.core.dataset_builder import build_confidence_entropy_dataset

        cache_path = Path("tasks/confidence_entropy_tasks.json")
        art_path = Path("tasks/art_tasks_template.json")

        self.tasks = build_confidence_entropy_dataset(
            art_path=art_path if art_path.exists() else None,
            cache_path=cache_path,
        )

        if self.config.sample_size is not None:
            self.tasks = sample_tasks(self.tasks, self.config.sample_size, self.config.seed)

        logger.info(f"Loaded {len(self.tasks)} tasks, {len(self.config.personas)} personas")
        self.results_logger = ResultsLogger(self.output_path)

    def run(self) -> None:
        """Execute all trials: each persona answers each question (2 prompts each)."""
        assert self.backend is not None and self.results_logger is not None

        total = len(self.config.personas) * len(self.tasks)
        pbar = tqdm(total=total, desc="Confidence vs Entropy (base model)")

        for persona_name in self.config.personas:
            persona = self.personas[persona_name]
            for task in self.tasks:
                try:
                    record = self._run_one_trial(persona, task)
                except Exception as exc:
                    logger.error(f"Trial failed: {persona_name}/{task.task_id}: {exc}")
                    record = ConfidenceEntropyRecord(
                        experiment="confidence_entropy_base",
                        model=self.config.model_name,
                        persona_name=persona_name,
                        question_id=task.task_id,
                        domain=task.metadata.get("domain", ""),
                        source_dataset=task.metadata.get("source_dataset", ""),
                        error=str(exc),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                self.results_logger.log_trial(record)
                pbar.update(1)

        pbar.close()

    def _run_one_trial(
        self, persona: PersonaConfig, task: TaskItem
    ) -> ConfidenceEntropyRecord:
        """Run one (persona, question) trial with 2 few-shot prompts."""
        assert self.backend is not None

        question_text, formatted_options, choices_dict = _format_options_block(
            task.choices, task.prompt
        )
        option_keys = sorted(choices_dict.keys()) if choices_dict else task.choices

        # ── Prompt 1: Forced-choice MC answer (few-shot) ──────────────
        few_shot_mode = self.config.few_shot_mode
        mc_prompt = format_mc_prompt_base(
            question_text=question_text,
            options=choices_dict,
            persona_name=persona.name,
            mode=few_shot_mode,
            use_suffix=self.config.use_persona_suffixes,
        )

        # Get logprob-based option probabilities BEFORE generating
        option_probs, option_logits = self.backend.get_choice_probs_and_logits_from_text(
            mc_prompt, option_keys
        )

        # Generate the forced-choice response
        forced_raw = self.backend.generate_from_text(
            mc_prompt,
            max_new_tokens=8,
            temperature=0.0,
        )

        # Parse forced-choice answer — truncate first since base models
        # continue the pattern after the letter (e.g. "B\n\nQuestion: ...")
        forced_answer, forced_validity = _parse_forced_answer(
            _truncate_to_first_token(forced_raw), option_keys
        )

        # Fallback: if parsing failed, use the logprob argmax
        if forced_answer is None:
            forced_answer = max(option_probs, key=option_probs.get)
            forced_validity = False
            logger.debug(f"Using logprob argmax fallback: {forced_answer}")

        # Compute entropy metrics
        prob_values = list(option_probs.values())
        entropy = -sum(p * math.log(p + 1e-12) for p in prob_values)
        sorted_probs = sorted(prob_values, reverse=True)
        chosen_prob = option_probs.get(forced_answer) if forced_answer else None
        margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else None
        answer_ranking = sorted(option_probs, key=option_probs.get, reverse=True)

        # ── Prompt 2: Stated confidence (few-shot) ────────────────────
        conf_prompt = format_confidence_prompt_base(
            question_text=question_text,
            options=choices_dict,
            persona_name=persona.name,
            mode=few_shot_mode,
        )
        conf_keys = list(STATED_CONFIDENCE_OPTIONS.keys())

        # Get logprob-based confidence probabilities
        confidence_option_probs, confidence_option_logits = (
            self.backend.get_choice_probs_and_logits_from_text(conf_prompt, conf_keys)
        )

        # Generate confidence response
        conf_raw = self.backend.generate_from_text(
            conf_prompt,
            max_new_tokens=8,
            temperature=0.0,
        )

        # Parse confidence letter — same truncation for base model output
        conf_letter, conf_validity = _parse_confidence_letter(
            _truncate_to_first_token(conf_raw)
        )

        # Fallback: if parsing failed, use the logprob argmax
        if conf_letter is None:
            conf_letter = max(confidence_option_probs, key=confidence_option_probs.get)
            conf_validity = False
            logger.debug(f"Using logprob argmax fallback for confidence: {conf_letter}")

        conf_midpoint = STATED_CONFIDENCE_MIDPOINTS.get(conf_letter) if conf_letter else None

        # ── Log sample prompt for the first trial ─────────────────────
        if not self._sample_logged:
            self._log_sample_prompts(
                persona, task,
                mc_prompt, forced_raw,
                conf_prompt, conf_raw,
                option_probs, option_logits,
                confidence_option_probs, confidence_option_logits,
            )
            self._sample_logged = True

        # ── Build record ──────────────────────────────────────────────
        is_correct = None
        if forced_answer and task.expected_answer:
            is_correct = forced_answer == task.expected_answer

        # Persona context (base-model equivalent of system prompt)
        persona_ctx = get_persona_context(persona.name)

        return ConfidenceEntropyRecord(
            experiment="confidence_entropy_base",
            model=self.config.model_name,
            persona_name=persona.name,
            question_id=task.task_id,
            domain=task.metadata.get("domain", ""),
            source_dataset=task.metadata.get("source_dataset", ""),
            correct_answer=task.expected_answer,
            open_ended_response="",  # not applicable for base models
            forced_choice_answer=forced_answer,
            forced_answer_validity=forced_validity,
            is_correct=is_correct,
            forced_choice_raw=forced_raw,
            option_probs=option_probs,
            option_logits=option_logits,
            answer_option_entropy=entropy,
            chosen_answer_probability=chosen_prob,
            margin_between_top_two=margin,
            answer_ranking=answer_ranking,
            confidence_open_response="",  # not applicable for base models
            stated_confidence_letter=conf_letter,
            stated_confidence_midpoint=conf_midpoint,
            stated_confidence_raw=conf_raw,
            confidence_answer_validity=conf_validity,
            confidence_option_probs=confidence_option_probs,
            confidence_option_logits=confidence_option_logits,
            system_prompt=persona_ctx.strip(),  # store persona context for reference
            temperature=0.0,
            mc_prompt_text=mc_prompt,
            confidence_prompt_text=conf_prompt,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _log_sample_prompts(
        self,
        persona: PersonaConfig,
        task: TaskItem,
        mc_prompt: str,
        forced_raw: str,
        conf_prompt: str,
        conf_raw: str,
        option_probs: dict[str, float],
        option_logits: dict[str, float],
        confidence_option_probs: dict[str, float],
        confidence_option_logits: dict[str, float],
    ) -> None:
        """Write a sample of exact prompts and outputs for one trial to the manifest."""
        lines = [
            "=" * 70,
            "SAMPLE TRIAL — Exact prompts and model outputs (BASE MODEL)",
            "=" * 70,
            f"Persona: {persona.name}",
            f"Question ID: {task.task_id}",
            f"Domain: {task.metadata.get('domain', '')}",
            f"Correct answer: {task.expected_answer}",
            "",
            "NOTE: Base model uses raw text prompts (no chat template).",
            "Only 2 prompts per trial (no open-ended prompts).",
            "",
            "-" * 70,
            "PROMPT 1: FORCED-CHOICE MC (few-shot pattern completion)",
            "-" * 70,
            "[RAW PROMPT]",
            mc_prompt,
            "",
            "[MODEL OUTPUT]",
            forced_raw,
            "",
            "[OPTION PROBS (softmax)]",
        ]
        for k, v in sorted(option_probs.items()):
            lines.append(f"  {k}: {v:.6f}")
        lines.append("")
        lines.append("[OPTION LOGITS (raw)]")
        for k, v in sorted(option_logits.items()):
            lines.append(f"  {k}: {v:.4f}")
        lines.append("")

        lines.append("-" * 70)
        lines.append("PROMPT 2: STATED CONFIDENCE (few-shot pattern completion)")
        lines.append("-" * 70)
        lines.append("[RAW PROMPT]")
        lines.append(conf_prompt)
        lines.append("")
        lines.append("[MODEL OUTPUT]")
        lines.append(conf_raw)
        lines.append("")
        lines.append("[CONFIDENCE OPTION PROBS (softmax)]")
        for k, v in sorted(confidence_option_probs.items()):
            label = STATED_CONFIDENCE_OPTIONS.get(k, "")
            lines.append(f"  {k} ({label}): {v:.6f}")
        lines.append("")
        lines.append("[CONFIDENCE OPTION LOGITS (raw)]")
        for k, v in sorted(confidence_option_logits.items()):
            label = STATED_CONFIDENCE_OPTIONS.get(k, "")
            lines.append(f"  {k} ({label}): {v:.4f}")
        lines.append("")
        lines.append("=" * 70)

        self._sample_lines = lines

    def _write_manifest(self) -> None:
        """Write the run manifest file."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "=" * 70,
            "CONFIDENCE vs ENTROPY (BASE MODEL) — RUN MANIFEST",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 70,
            "",
            "── RUN CONFIGURATION ──",
            f"Experiment:    {self.config.experiment_name}",
            f"Model:         {self.config.model_name}",
            f"Model type:    BASE (non-instruct)",
            f"Personas:      {', '.join(self.config.personas)}",
            f"Sample size:   {self.config.sample_size or 'all'}",
            f"Seed:          {self.config.seed}",
            f"Max new tokens:{self.config.max_new_tokens}",
            f"Temperature:   {self.config.temperature}",
            f"N tasks:       {len(self.tasks)}",
            f"N personas:    {len(self.config.personas)}",
            f"Total trials:  {len(self.tasks) * len(self.config.personas)}",
            f"Results file:  {self.output_path}",
            "",
            "── PROMPTING STRATEGY ──",
            "Base models use few-shot pattern completion instead of instructions.",
            "Persona induction is done via a text prefix before the few-shot block.",
            "Only 2 prompts per trial: forced-choice MC + stated confidence.",
            "Open-ended prompts are skipped (base models don't follow instructions).",
            f"Few-shot mode: {self.config.few_shot_mode}",
            f"Persona suffixes: {self.config.use_persona_suffixes}",
            "",
            "── PERSONA CONTEXT PREFIXES ──",
        ]

        for name in self.config.personas:
            ctx = get_persona_context(name)
            lines.append(f"\n[{name}]")
            lines.append(ctx.strip() if ctx.strip() else "(no extra context)")

        lines.append("")

        if hasattr(self, "_sample_lines"):
            lines.append("")
            lines.extend(self._sample_lines)

        if self._warnings:
            lines.append("")
            lines.append("── WARNINGS ──")
            for w in self._warnings:
                lines.append(f"  - {w}")

        with open(self.manifest_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Manifest saved to: {self.manifest_path}")

    def evaluate(self) -> dict:
        """Compute aggregate metrics from logged results."""
        from cross_persona_introspection.core.results_logger import load_results

        df = load_results(self.output_path)
        df_valid = df[df["error"].isna()]

        if df_valid.empty:
            return {"n_trials": 0, "n_errors": len(df), "error": "no valid trials"}

        summary = {
            "n_trials": len(df_valid),
            "n_errors": len(df) - len(df_valid),
            "model_type": "base",
            "forced_choice_parse_rate": float(df_valid["forced_choice_answer"].notna().mean()),
            "confidence_parse_rate": float(df_valid["stated_confidence_letter"].notna().mean()),
            "overall_accuracy": float(df_valid["is_correct"].mean())
            if df_valid["is_correct"].notna().any()
            else None,
            "mean_stated_confidence": float(df_valid["stated_confidence_midpoint"].mean())
            if df_valid["stated_confidence_midpoint"].notna().any()
            else None,
            "mean_entropy": float(df_valid["answer_option_entropy"].mean()),
            "mean_margin": float(df_valid["margin_between_top_two"].mean())
            if df_valid["margin_between_top_two"].notna().any()
            else None,
        }

        # Per-persona breakdown
        per_persona = {}
        for persona_name, group in df_valid.groupby("persona_name"):
            entry = {
                "n": len(group),
                "accuracy": float(group["is_correct"].mean())
                if group["is_correct"].notna().any()
                else None,
                "mean_stated_confidence": float(group["stated_confidence_midpoint"].mean())
                if group["stated_confidence_midpoint"].notna().any()
                else None,
                "mean_entropy": float(group["answer_option_entropy"].mean()),
                "mean_margin": float(group["margin_between_top_two"].mean())
                if group["margin_between_top_two"].notna().any()
                else None,
            }
            per_persona[persona_name] = entry
        summary["per_persona"] = per_persona

        # Per-domain breakdown
        per_domain = {}
        for domain, group in df_valid.groupby("domain"):
            entry = {
                "n": len(group),
                "accuracy": float(group["is_correct"].mean())
                if group["is_correct"].notna().any()
                else None,
                "mean_entropy": float(group["answer_option_entropy"].mean()),
            }
            per_domain[domain] = entry
        summary["per_domain"] = per_domain

        return summary

    def save_results(self) -> str:
        """Write manifest and return the results path."""
        self._write_manifest()
        return str(self.output_path)
