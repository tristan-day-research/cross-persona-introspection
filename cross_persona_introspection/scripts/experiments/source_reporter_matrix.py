"""Source × Reporter matrix experiment using shared-prefix KV cache reuse.

For each task and source persona S:
1. Run the source prefix (system prompt + task + pause cue) and save past_key_values
2. Measure source-state metrics at the pause point
3. Source completion: let the source continue and produce its answer
4. Reporter continuation: for each reporter R, append a reporter suffix
   that reuses the same KV cache and asks R to report the model's inclination
5. No-cache baseline: ask each reporter to predict the source from scratch

The matrix is Source × Reporter. Each cell measures how accurately reporter R
can read the internal state produced by source S.
"""

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from tqdm import tqdm

from cross_persona_introspection.backends.hf_backend import HFBackend
from cross_persona_introspection.core.config_loader import load_prompts, get_response_format
from cross_persona_introspection.core.kv_cache import run_source_prefix
from cross_persona_introspection.core.persona_inducer import induce_persona, build_reporter_suffix
from cross_persona_introspection.core.response_parser import parse_constrained_response
from cross_persona_introspection.core.results_logger import ResultsLogger, load_results
from cross_persona_introspection.core.task_loader import load_tasks, sample_tasks
from cross_persona_introspection.evaluation.choice_matching import score_reporter_trial
from cross_persona_introspection.experiments.base import BaseExperiment
from cross_persona_introspection.schemas import PersonaConfig, RunConfig, TaskItem, TrialRecord, SourceStateMetrics

logger = logging.getLogger(__name__)


class SourceReporterMatrix(BaseExperiment):
    """Source × Reporter matrix with shared-prefix KV cache reuse."""

    def __init__(self, config: RunConfig, personas: dict[str, PersonaConfig]):
        super().__init__(config)
        self.personas = personas
        self.backend: HFBackend | None = None
        self.tasks: list[TaskItem] = []
        self.output_path = Path(config.output_dir) / f"source_reporter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.results_logger: ResultsLogger | None = None

    def setup(self) -> None:
        logger.info(f"Loading model: {self.config.model_name}")
        self.backend = HFBackend(self.config.model_name)

        all_tasks = load_tasks(self.config.task_sets)
        self.tasks = sample_tasks(all_tasks, self.config.sample_size, self.config.seed)
        logger.info(f"Loaded {len(self.tasks)} tasks")

        self.results_logger = ResultsLogger(self.output_path)

    def run(self) -> None:
        assert self.backend is not None and self.results_logger is not None

        persona_names = self.config.personas
        n_sources = len(persona_names)
        n_reporters = len(persona_names)
        total = len(self.tasks) * n_sources * (1 + n_reporters + n_reporters)
        # Per task per source: 1 source completion + N reporter continuations + N no-cache baselines

        pbar = tqdm(total=total, desc="Source × Reporter matrix")

        for task in self.tasks:
            for source_name in persona_names:
                source_persona = self.personas[source_name]

                try:
                    # Step 1: Run source prefix and get cached state
                    prefix_ids, past_key_values, source_metrics = run_source_prefix(
                        self.backend,
                        source_persona,
                        task,
                        self.config.pause_cue,
                    )
                    metrics_dict = asdict(source_metrics)

                    # Step 2: Source completion — let source continue
                    source_record = self._run_source_completion(
                        source_persona, task, prefix_ids, past_key_values, metrics_dict
                    )
                    self.results_logger.log_trial(source_record)
                    pbar.update(1)

                    # Step 3: Reporter continuations (with KV cache reuse)
                    for reporter_name in persona_names:
                        reporter_persona = self.personas[reporter_name]
                        try:
                            record = self._run_reporter_continuation(
                                source_persona, reporter_persona, task,
                                prefix_ids, past_key_values, metrics_dict,
                            )
                            self.results_logger.log_trial(record)
                        except Exception as e:
                            logger.error(f"Reporter continuation failed: {source_name}->{reporter_name} on {task.task_id}: {e}")
                            self.results_logger.log_error(
                                experiment="source_reporter_matrix",
                                task_id=task.task_id,
                                error=str(e),
                                source_persona=source_name,
                                reporter_persona=reporter_name,
                            )
                        pbar.update(1)

                    # Step 4: No-cache baselines
                    for reporter_name in persona_names:
                        reporter_persona = self.personas[reporter_name]
                        try:
                            record = self._run_no_cache_baseline(
                                source_persona, reporter_persona, task, metrics_dict
                            )
                            self.results_logger.log_trial(record)
                        except Exception as e:
                            logger.error(f"No-cache baseline failed: {source_name}->{reporter_name} on {task.task_id}: {e}")
                            self.results_logger.log_error(
                                experiment="source_reporter_matrix_baseline",
                                task_id=task.task_id,
                                error=str(e),
                                source_persona=source_name,
                                reporter_persona=reporter_name,
                            )
                        pbar.update(1)

                except Exception as e:
                    logger.error(f"Source prefix failed: {source_name} on {task.task_id}: {e}")
                    self.results_logger.log_error(
                        experiment="source_reporter_matrix",
                        task_id=task.task_id,
                        error=f"Source prefix failed: {e}",
                        source_persona=source_name,
                    )
                    pbar.update(1 + 2 * n_reporters)

        pbar.close()

    def _run_source_completion(
        self,
        source: PersonaConfig,
        task: TaskItem,
        prefix_ids,
        past_key_values,
        metrics_dict: dict,
    ) -> TrialRecord:
        """Let the source persona continue from the pause point and produce its answer."""
        assert self.backend is not None

        prompts = load_prompts()
        response_format = get_response_format(task.choices)
        continuation_suffix = prompts["source_continuation"].format(
            response_format=response_format,
        )

        raw = self.backend.continue_from_cache(
            prefix_ids, past_key_values, continuation_suffix,
            max_new_tokens=self.config.max_new_tokens,
        )
        parsed = parse_constrained_response(raw)

        return TrialRecord(
            experiment="source_reporter_matrix_source",
            model=self.config.model_name,
            task_id=task.task_id,
            task_set=task.task_set,
            source_persona=source.name,
            reporter_persona=source.name,  # source reports on itself
            source_metrics=metrics_dict,
            reporter_answer=parsed.answer,
            reporter_confidence=parsed.confidence,
            reporter_refuse=parsed.refuse,
            used_kv_cache=True,
            raw_response=raw,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _run_reporter_continuation(
        self,
        source: PersonaConfig,
        reporter: PersonaConfig,
        task: TaskItem,
        prefix_ids,
        past_key_values,
        metrics_dict: dict,
    ) -> TrialRecord:
        """Reporter continues from the source's KV cache to report the model's state."""
        assert self.backend is not None

        suffix = build_reporter_suffix(reporter, task.choices)
        raw = self.backend.continue_from_cache(
            prefix_ids, past_key_values, suffix,
            max_new_tokens=self.config.max_new_tokens,
        )
        parsed = parse_constrained_response(raw)

        record = TrialRecord(
            experiment="source_reporter_matrix",
            model=self.config.model_name,
            task_id=task.task_id,
            task_set=task.task_set,
            source_persona=source.name,
            reporter_persona=reporter.name,
            source_metrics=metrics_dict,
            reporter_answer=parsed.answer,
            reporter_confidence=parsed.confidence,
            reporter_refuse=parsed.refuse,
            used_kv_cache=True,
            raw_response=raw,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return score_reporter_trial(record)

    def _run_no_cache_baseline(
        self,
        source: PersonaConfig,
        reporter: PersonaConfig,
        task: TaskItem,
        metrics_dict: dict,
    ) -> TrialRecord:
        """Baseline: reporter predicts source's answer from scratch (no KV cache reuse).

        This distinguishes privileged access from ordinary guessing.
        """
        assert self.backend is not None

        prompts = load_prompts()
        response_format = get_response_format(task.choices)
        prompt = prompts["no_cache_baseline"].format(
            source_name=source.name,
            task_prompt=task.prompt,
            response_format=response_format,
        )
        messages = induce_persona(reporter, [{"role": "user", "content": prompt}])
        raw = self.backend.generate(
            messages,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        parsed = parse_constrained_response(raw)

        record = TrialRecord(
            experiment="source_reporter_matrix_baseline",
            model=self.config.model_name,
            task_id=task.task_id,
            task_set=task.task_set,
            source_persona=source.name,
            reporter_persona=reporter.name,
            source_metrics=metrics_dict,
            reporter_answer=parsed.answer,
            reporter_confidence=parsed.confidence,
            reporter_refuse=parsed.refuse,
            used_kv_cache=False,
            raw_response=raw,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return score_reporter_trial(record)

    def evaluate(self) -> dict:
        """Compute aggregate metrics: per-cell accuracy in the S×R matrix."""
        df = load_results(self.output_path)

        results = {}
        for exp_type in ["source_reporter_matrix", "source_reporter_matrix_baseline"]:
            sub = df[df["experiment"] == exp_type]
            sub_valid = sub[sub["error"].isna()]
            if sub_valid.empty:
                results[exp_type] = {"n_trials": 0}
                continue

            results[exp_type] = {
                "n_trials": len(sub_valid),
                "choice_match_rate": sub_valid["choice_match"].mean(),
                "mean_confidence_error": sub_valid["confidence_error"].abs().mean() if "confidence_error" in sub_valid else None,
            }

            # Per source-reporter cell
            cells = {}
            for (s, r), group in sub_valid.groupby(["source_persona", "reporter_persona"]):
                cells[f"{s}->{r}"] = {
                    "n": len(group),
                    "choice_match_rate": group["choice_match"].mean(),
                }
            results[exp_type]["cells"] = cells

        return results

    def save_results(self) -> str:
        return str(self.output_path)
