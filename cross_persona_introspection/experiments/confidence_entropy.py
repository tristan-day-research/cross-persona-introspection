"""Confidence vs Entropy experiment.

Tests whether LLM personas change actual model uncertainty (entropy over
answer options) or just change self-reported confidence (surface-level style).

Each trial consists of 3 independent prompts (separate contexts):
1. Open-ended: free-form reasoning about the MCQ
2. Forced-choice: model outputs only the answer letter → we extract logprobs here
3. Stated confidence: model rates its confidence on a categorical scale (S-Z)

Scientific notes:
- All personas answer the SAME shared question set.
- Entropy is computed over answer options only (A/B/C/D), not the full vocabulary.
- Reported confidence can be distorted by persona style; that is part of what we test.
- Domain personas (Chemist, Historian, Artist) and style personas (Cautious, Bold)
  should be compared separately in analysis.
"""

import json
import logging
import math
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from cross_persona_introspection.core.config_loader import load_prompts
from cross_persona_introspection.core.persona_inducer import induce_persona
from cross_persona_introspection.core.results_logger import ResultsLogger
from cross_persona_introspection.core.task_loader import sample_tasks
from cross_persona_introspection.experiments.base import BaseExperiment
from cross_persona_introspection.schemas import (
    ConfidenceEntropyRecord,
    PersonaConfig,
    RunConfig,
    TaskItem,
)

logger = logging.getLogger(__name__)

# ── Stated confidence scale ───────────────────────────────────────────
# Letters chosen to avoid overlap with answer options A-E.

STATED_CONFIDENCE_OPTIONS = {
    "S": "<5%", "T": "5-10%", "U": "10-20%", "V": "20-40%",
    "W": "40-60%", "X": "60-80%", "Y": "80-90%", "Z": ">90%",
}

STATED_CONFIDENCE_MIDPOINTS = {
    "S": 0.025, "T": 0.075, "U": 0.15, "V": 0.3,
    "W": 0.5, "X": 0.7, "Y": 0.85, "Z": 0.95,
}


def _format_options_block(choices: dict[str, str] | list[str], task_prompt: str) -> tuple[str, str, dict[str, str]]:
    """Extract question text and format options block from a task.

    Returns (question_text, formatted_options, choices_dict).
    """
    # Parse the task prompt to separate question text from options
    lines = task_prompt.strip().split("\n")
    question_lines = []
    option_lines = {}

    for line in lines:
        stripped = line.strip()
        match = re.match(r"^([A-E])\)\s*(.+)$", stripped)
        if match:
            option_lines[match.group(1)] = match.group(2)
        elif stripped:
            question_lines.append(stripped)

    question_text = "\n".join(question_lines)
    if not option_lines and isinstance(choices, dict):
        option_lines = choices

    formatted = "\n".join(f"  {k}: {v}" for k, v in sorted(option_lines.items()))
    return question_text, formatted, option_lines


class ConfidenceEntropy(BaseExperiment):
    """Test whether personas change actual uncertainty or just reported confidence."""

    def __init__(self, config: RunConfig, personas: dict[str, PersonaConfig]):
        super().__init__(config)
        self.personas = personas
        self.backend = None
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = Path(config.output_dir) / (
            f"confidence_entropy_{self._run_timestamp}.jsonl"
        )
        self.manifest_path = Path(config.output_dir) / (
            f"confidence_entropy_{self._run_timestamp}_manifest.txt"
        )
        self.tasks: list[TaskItem] = []
        self.results_logger: ResultsLogger | None = None
        self._sample_logged = False
        self._warnings: list[str] = []

    def setup(self) -> None:
        """Load model and dataset."""
        from cross_persona_introspection.backends.hf_backend import HFBackend

        logger.info(f"Loading model: {self.config.model_name}")
        self.backend = HFBackend(self.config.model_name)

        # Build or load cached dataset
        from cross_persona_introspection.core.dataset_builder import build_confidence_entropy_dataset

        cache_path = Path("tasks/confidence_entropy_tasks.json")
        art_path = Path("tasks/art_tasks_template.json")

        self.tasks = build_confidence_entropy_dataset(
            art_path=art_path if art_path.exists() else None,
            cache_path=cache_path,
        )

        # Apply sampling if configured
        if self.config.sample_size is not None:
            self.tasks = sample_tasks(self.tasks, self.config.sample_size, self.config.seed)

        logger.info(f"Loaded {len(self.tasks)} tasks, {len(self.config.personas)} personas")
        self.results_logger = ResultsLogger(self.output_path)

    def run(self) -> None:
        """Execute all trials: each persona answers each question (3 prompts each)."""
        assert self.backend is not None and self.results_logger is not None

        total = len(self.config.personas) * len(self.tasks)
        pbar = tqdm(total=total, desc="Confidence vs Entropy")

        for persona_name in self.config.personas:
            persona = self.personas[persona_name]
            for task in self.tasks:
                try:
                    record = self._run_one_trial(persona, task)
                except Exception as exc:
                    logger.error(f"Trial failed: {persona_name}/{task.task_id}: {exc}")
                    record = ConfidenceEntropyRecord(
                        experiment="confidence_entropy",
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
        """Run one (persona, question) trial with 3 independent prompts."""
        assert self.backend is not None

        prompts = load_prompts()
        question_text, formatted_options, choices_dict = _format_options_block(
            task.choices, task.prompt
        )
        option_keys = sorted(choices_dict.keys()) if choices_dict else task.choices
        options_str = ", ".join(option_keys[:-1]) + f", or {option_keys[-1]}" if len(option_keys) > 2 else " or ".join(option_keys)

        # ── Prompt 1: Open-ended reasoning ────────────────────────────
        open_ended_prompt = prompts["ce_open_ended"].format(
            question_prompt=task.prompt,
        )
        open_ended_messages = induce_persona(
            persona, [{"role": "user", "content": open_ended_prompt}]
        )
        open_ended_response = self.backend.generate(
            open_ended_messages,
            max_new_tokens=self.config.max_new_tokens,
            temperature=0.0,
        )

        # ── Prompt 2: Forced-choice (letter only) ─────────────────────
        forced_setup = prompts["ce_forced_choice_setup"]
        forced_question = prompts["ce_forced_choice"].format(
            question_text=question_text,
            formatted_options=formatted_options,
            options_str=options_str,
        )
        forced_user_content = forced_setup + "\n\n" + forced_question
        forced_messages = induce_persona(
            persona, [{"role": "user", "content": forced_user_content}]
        )

        # Get logprob-based option probabilities BEFORE generating
        option_probs = self.backend.get_choice_probs(forced_messages, option_keys)

        # Generate the forced-choice response
        forced_raw = self.backend.generate(
            forced_messages,
            max_new_tokens=8,  # only need a single letter
            temperature=0.0,
        )

        # Parse forced-choice answer
        forced_answer = None
        ans_match = re.search(r"[A-Ea-e]", forced_raw.strip())
        if ans_match:
            forced_answer = ans_match.group(0).upper()

        # Compute entropy metrics over answer options
        prob_values = list(option_probs.values())
        entropy = -sum(p * math.log(p + 1e-12) for p in prob_values)
        sorted_probs = sorted(prob_values, reverse=True)
        chosen_prob = option_probs.get(forced_answer) if forced_answer else None
        margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else None

        # ── Prompt 3: Stated confidence ───────────────────────────────
        conf_setup = prompts["ce_stated_confidence_setup"]
        conf_formatted = "\n".join(
            f"  {k}: {v}" for k, v in STATED_CONFIDENCE_OPTIONS.items()
        )
        conf_keys = list(STATED_CONFIDENCE_OPTIONS.keys())
        conf_options_str = ", ".join(conf_keys[:-1]) + f", or {conf_keys[-1]}"

        conf_question = prompts["ce_stated_confidence"].format(
            question_text=question_text,
            formatted_options=formatted_options,
            confidence_formatted_options=conf_formatted,
            confidence_options_str=conf_options_str,
        )
        conf_user_content = conf_setup + "\n\n" + conf_question
        conf_messages = induce_persona(
            persona, [{"role": "user", "content": conf_user_content}]
        )

        # Get logprob-based confidence option probabilities
        confidence_option_probs = self.backend.get_choice_probs(conf_messages, conf_keys)

        # Generate the confidence response
        conf_raw = self.backend.generate(
            conf_messages,
            max_new_tokens=8,
            temperature=0.0,
        )

        # Parse confidence letter
        conf_letter = None
        conf_midpoint = None
        conf_match = re.search(r"[S-Zs-z]", conf_raw.strip())
        if conf_match:
            letter = conf_match.group(0).upper()
            if letter in STATED_CONFIDENCE_OPTIONS:
                conf_letter = letter
                conf_midpoint = STATED_CONFIDENCE_MIDPOINTS[letter]

        # ── Log sample prompt for the first trial ─────────────────────
        if not self._sample_logged:
            self._log_sample_prompts(
                persona, task,
                open_ended_messages, open_ended_response,
                forced_messages, forced_raw,
                conf_messages, conf_raw,
                option_probs, confidence_option_probs,
            )
            self._sample_logged = True

        # ── Build record ──────────────────────────────────────────────
        is_correct = None
        if forced_answer and task.expected_answer:
            is_correct = forced_answer == task.expected_answer

        return ConfidenceEntropyRecord(
            experiment="confidence_entropy",
            model=self.config.model_name,
            persona_name=persona.name,
            question_id=task.task_id,
            domain=task.metadata.get("domain", ""),
            source_dataset=task.metadata.get("source_dataset", ""),
            correct_answer=task.expected_answer,
            open_ended_response=open_ended_response,
            forced_choice_answer=forced_answer,
            is_correct=is_correct,
            forced_choice_raw=forced_raw,
            option_probs=option_probs,
            answer_option_entropy=entropy,
            chosen_answer_probability=chosen_prob,
            margin_between_top_two=margin,
            stated_confidence_letter=conf_letter,
            stated_confidence_midpoint=conf_midpoint,
            stated_confidence_raw=conf_raw,
            confidence_option_probs=confidence_option_probs,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _log_sample_prompts(
        self,
        persona: PersonaConfig,
        task: TaskItem,
        open_ended_messages: list[dict],
        open_ended_response: str,
        forced_messages: list[dict],
        forced_raw: str,
        conf_messages: list[dict],
        conf_raw: str,
        option_probs: dict[str, float],
        confidence_option_probs: dict[str, float],
    ) -> None:
        """Write a sample of exact prompts and outputs for one trial to the manifest."""
        lines = [
            "=" * 70,
            "SAMPLE TRIAL — Exact prompts and model outputs",
            "=" * 70,
            f"Persona: {persona.name}",
            f"Question ID: {task.task_id}",
            f"Domain: {task.metadata.get('domain', '')}",
            f"Correct answer: {task.expected_answer}",
            "",
            "-" * 70,
            "PROMPT 1: OPEN-ENDED REASONING",
            "-" * 70,
        ]
        for msg in open_ended_messages:
            lines.append(f"[{msg['role']}]")
            lines.append(msg["content"])
            lines.append("")
        lines.append("[MODEL OUTPUT]")
        lines.append(open_ended_response)
        lines.append("")

        lines.append("-" * 70)
        lines.append("PROMPT 2: FORCED-CHOICE (letter only)")
        lines.append("-" * 70)
        for msg in forced_messages:
            lines.append(f"[{msg['role']}]")
            lines.append(msg["content"])
            lines.append("")
        lines.append("[MODEL OUTPUT]")
        lines.append(forced_raw)
        lines.append("")
        lines.append("[OPTION PROBS (from logits)]")
        for k, v in sorted(option_probs.items()):
            lines.append(f"  {k}: {v:.6f}")
        lines.append("")

        lines.append("-" * 70)
        lines.append("PROMPT 3: STATED CONFIDENCE")
        lines.append("-" * 70)
        for msg in conf_messages:
            lines.append(f"[{msg['role']}]")
            lines.append(msg["content"])
            lines.append("")
        lines.append("[MODEL OUTPUT]")
        lines.append(conf_raw)
        lines.append("")
        lines.append("[CONFIDENCE OPTION PROBS (from logits)]")
        for k, v in sorted(confidence_option_probs.items()):
            label = STATED_CONFIDENCE_OPTIONS.get(k, "")
            lines.append(f"  {k} ({label}): {v:.6f}")
        lines.append("")
        lines.append("=" * 70)

        self._sample_lines = lines

    def _write_manifest(self) -> None:
        """Write the run manifest file with config, sample prompts, and warnings."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "=" * 70,
            "CONFIDENCE vs ENTROPY — RUN MANIFEST",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "=" * 70,
            "",
            "── RUN CONFIGURATION ──",
            f"Experiment:    {self.config.experiment_name}",
            f"Model:         {self.config.model_name}",
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
            "── PERSONA SYSTEM PROMPTS ──",
        ]

        for name in self.config.personas:
            persona = self.personas.get(name)
            if persona:
                lines.append(f"\n[{name}]")
                lines.append(persona.system_prompt if persona.system_prompt else "(empty)")

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
