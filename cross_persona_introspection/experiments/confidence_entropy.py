"""Confidence vs Entropy experiment.

Tests whether LLM personas change actual model uncertainty (entropy over
answer options) or just change self-reported confidence (surface-level style).

For each persona × question:
1. Send the question with persona system prompt, request JSON answer + confidence_prob
2. Compute logprob-based option_probs over answer choices using get_choice_probs()
3. Compute entropy, chosen-answer probability, margin between top two options
4. Log a ConfidenceEntropyRecord

Scientific notes:
- All personas answer the SAME shared question set.
- Entropy is computed over answer options only (A/B/C/D), not the full vocabulary.
- Reported confidence can be distorted by persona style; that is part of what we test.
- Domain personas (Chemist, Historian, Artist) and style personas (Cautious, Bold)
  should be compared separately in analysis.
"""

import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from cross_persona_introspection.core.config_loader import load_prompts
from cross_persona_introspection.core.persona_inducer import induce_persona
from cross_persona_introspection.core.response_parser import parse_json_response
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


class ConfidenceEntropy(BaseExperiment):
    """Test whether personas change actual uncertainty or just reported confidence."""

    def __init__(self, config: RunConfig, personas: dict[str, PersonaConfig]):
        super().__init__(config)
        self.personas = personas
        self.backend = None
        self.tasks: list[TaskItem] = []
        self.output_path = Path(config.output_dir) / (
            f"confidence_entropy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        self.results_logger: ResultsLogger | None = None

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
        """Execute all trials: each persona answers each question."""
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
        """Run one (persona, question) trial."""
        assert self.backend is not None

        prompts = load_prompts()
        prompt_text = prompts["confidence_entropy_direct"].format(
            question_prompt=task.prompt,
        )
        messages = induce_persona(persona, [{"role": "user", "content": prompt_text}])

        # Step 1: Generate text response (should be JSON)
        raw_text = self.backend.generate(
            messages,
            max_new_tokens=self.config.max_new_tokens,
            temperature=0.0,
        )

        # Step 2: Parse JSON response
        answer, confidence_prob, parsed_ok = parse_json_response(raw_text)

        # Step 3: Get logprob-based option probabilities
        option_probs = self.backend.get_choice_probs(messages, task.choices)

        # Step 4: Compute entropy metrics over answer options
        prob_values = list(option_probs.values())
        entropy = -sum(p * math.log(p + 1e-12) for p in prob_values)

        sorted_probs = sorted(prob_values, reverse=True)
        chosen_prob = option_probs.get(answer) if answer else None
        margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else None

        # Step 5: Build record
        is_correct = None
        if answer and task.expected_answer:
            is_correct = answer == task.expected_answer

        return ConfidenceEntropyRecord(
            experiment="confidence_entropy",
            model=self.config.model_name,
            persona_name=persona.name,
            question_id=task.task_id,
            domain=task.metadata.get("domain", ""),
            source_dataset=task.metadata.get("source_dataset", ""),
            predicted_answer=answer,
            correct_answer=task.expected_answer,
            is_correct=is_correct,
            reported_confidence_prob=confidence_prob,
            parsed_success=parsed_ok,
            option_probs=option_probs,
            answer_option_entropy=entropy,
            chosen_answer_probability=chosen_prob,
            margin_between_top_two=margin,
            raw_text_output=raw_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

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
            "parse_success_rate": float(df_valid["parsed_success"].mean()),
            "overall_accuracy": float(df_valid["is_correct"].mean())
            if "is_correct" in df_valid and df_valid["is_correct"].notna().any()
            else None,
            "mean_reported_confidence": float(df_valid["reported_confidence_prob"].mean())
            if df_valid["reported_confidence_prob"].notna().any()
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
                "mean_reported_confidence": float(group["reported_confidence_prob"].mean())
                if group["reported_confidence_prob"].notna().any()
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
        """Return the path where results were logged."""
        return str(self.output_path)
