"""Cross-persona prediction experiment.

Put the model in reporter persona A, ask it to predict what target persona B
would do on a task, then run target persona B directly to get ground truth.
Compare predictions to actuals.
"""

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from tqdm import tqdm

from cross_persona_introspection.backends.hf_backend import HFBackend
from cross_persona_introspection.core.persona_inducer import induce_persona
from cross_persona_introspection.core.response_parser import parse_constrained_response
from cross_persona_introspection.core.results_logger import ResultsLogger, load_results
from cross_persona_introspection.core.task_loader import load_tasks, sample_tasks
from cross_persona_introspection.evaluation.choice_matching import score_trial
from cross_persona_introspection.experiments.base import BaseExperiment
from cross_persona_introspection.schemas import PersonaConfig, RunConfig, TaskItem, TrialRecord

logger = logging.getLogger(__name__)

from cross_persona_introspection.core.config_loader import load_prompts, get_response_format


class CrossPersonaPrediction(BaseExperiment):
    """Reporter persona A predicts target persona B's behavior."""

    def __init__(self, config: RunConfig, personas: dict[str, PersonaConfig]):
        super().__init__(config)
        self.personas = personas
        self.backend: HFBackend | None = None
        self.tasks: list[TaskItem] = []
        self.output_path = Path(config.output_dir) / f"cross_persona_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.results_logger: ResultsLogger | None = None

    def setup(self) -> None:
        """Load model and tasks."""
        logger.info(f"Loading model: {self.config.model_name}")
        self.backend = HFBackend(self.config.model_name)

        all_tasks = load_tasks(self.config.task_sets)
        self.tasks = sample_tasks(all_tasks, self.config.sample_size, self.config.seed)
        logger.info(f"Loaded {len(self.tasks)} tasks")

        self.results_logger = ResultsLogger(self.output_path)

    def run(self) -> None:
        """Run all reporter × target × task trials."""
        assert self.backend is not None and self.results_logger is not None

        persona_names = self.config.personas
        pairs = [(r, t) for r, t in product(persona_names, persona_names) if r != t]

        total = len(pairs) * len(self.tasks)
        pbar = tqdm(total=total, desc="Cross-persona prediction")

        for reporter_name, target_name in pairs:
            reporter = self.personas[reporter_name]
            target = self.personas[target_name]

            for task in self.tasks:
                try:
                    record = self._run_one_trial(reporter, target, task)
                    self.results_logger.log_trial(record)
                except Exception as e:
                    logger.error(f"Trial failed: {reporter_name}->{target_name} on {task.task_id}: {e}")
                    self.results_logger.log_error(
                        experiment="cross_persona_prediction",
                        task_id=task.task_id,
                        error=str(e),
                        source_persona=target_name,
                        reporter_persona=reporter_name,
                    )
                pbar.update(1)

        pbar.close()

    def _run_one_trial(
        self,
        reporter: PersonaConfig,
        target: PersonaConfig,
        task: TaskItem,
    ) -> TrialRecord:
        """Run one prediction trial and one ground-truth trial."""
        assert self.backend is not None

        prompts = load_prompts()
        response_format = get_response_format(task.choices)

        # Step 1: Reporter predicts target's behavior
        prediction_prompt = prompts["cross_persona_prediction"].format(
            target_persona_name=target.name,
            target_persona_description=f"Description: {target.description}" if target.description else "",
            task_prompt=task.prompt,
            response_format=response_format,
        )
        prediction_messages = induce_persona(
            reporter, [{"role": "user", "content": prediction_prompt}]
        )
        prediction_raw = self.backend.generate(
            prediction_messages,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        prediction = parse_constrained_response(prediction_raw)

        # Step 2: Target answers directly
        direct_prompt = f"{task.prompt}\n\nRespond in exactly this format:\n{response_format}"
        direct_messages = induce_persona(
            target, [{"role": "user", "content": direct_prompt}]
        )
        actual_raw = self.backend.generate(
            direct_messages,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        actual = parse_constrained_response(actual_raw)

        # Step 3: Score
        record = TrialRecord(
            experiment="cross_persona_prediction",
            model=self.config.model_name,
            task_id=task.task_id,
            task_set=task.task_set,
            reporter_persona=reporter.name,
            source_persona=target.name,
            predicted_answer=prediction.answer,
            predicted_confidence=prediction.confidence,
            predicted_refuse=prediction.refuse,
            actual_answer=actual.answer,
            actual_confidence=actual.confidence,
            actual_refuse=actual.refuse,
            raw_response=f"PREDICTION: {prediction_raw}\n---\nACTUAL: {actual_raw}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        record = score_trial(record)
        return record

    def evaluate(self) -> dict:
        """Compute aggregate metrics from saved results."""
        df = load_results(self.output_path)
        df_valid = df[df["error"].isna()]

        if df_valid.empty:
            return {"n_trials": 0, "error": "no valid trials"}

        return {
            "n_trials": len(df_valid),
            "choice_match_rate": df_valid["choice_match"].mean(),
            "mean_confidence_error": df_valid["confidence_error"].abs().mean() if "confidence_error" in df_valid else None,
            "refuse_match_rate": df_valid["refuse_match"].mean() if "refuse_match" in df_valid else None,
            "n_errors": len(df) - len(df_valid),
        }

    def save_results(self) -> str:
        """Results are already saved incrementally. Return path."""
        return str(self.output_path)
