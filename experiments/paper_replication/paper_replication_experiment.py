"""Faithful replication of Panickssery, Bowman & Feng (2024),
"LLM Evaluators Recognize and Favor Their Own Generations" (arXiv:2404.13076).
Reference code: https://github.com/ArjunPanickssery/self_recognition

Out-of-the-box experiment (paper §2): a single local evaluator model judges
candidate summaries drawn from several sources. The "self" source is the paper's
own released `llama` summaries; the alternatives are human / gpt4 / gpt35 / claude.
All candidate texts are the authors' released outputs (vendored under data/), so
no API keys are needed — only the local evaluator runs, reading constrained
first-token probabilities (the paper's confidence measure).

Four measurements, all using HFBackend.get_choice_probs:
  detection   — pairwise self-recognition  ("which did you write?", options 1/2)
  comparison  — pairwise self-preference   ("which do you prefer?",  options 1/2)
  recognition — individual self-recognition ("did you write this?",  Yes/No)
  scoring     — individual self-preference  (Likert 1-5, prob-weighted mean)

Pairwise scores are corrected for ordering bias exactly as the reference repo
(experiments.py): each pair is run in both orders and the evaluator's confidence
in its OWN summary is averaged — 0.5·(P_fwd(self) + P_bwd(self)). This is
equivalent to the reference's `match (forward, backward)` block, since across all
four (top-token) cases that block reduces to the probability mass the model put
on the self-summary's position, averaged over the two orders.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

from core.backends.hf_backend import HFBackend
from core.results_logger import ResultsLogger
from core.schemas import PaperReplicationRecord, PersonaConfig, RunConfig
from core.self_recognition import (
    ANSWER_PRIMER,
    averaged_pairwise_score,
    choice_probs_batch,
    render_with_primer,
)
from experiments.base import BaseExperiment

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "paper_replication_config.yaml"
DATA_DIR = Path(__file__).parent / "data"

ALL_PHASES = ("detection", "comparison", "recognition", "scoring")

# The assistant-turn primer, constrained-choice probability reader, and the
# pairwise ordering-bias correction now live in core.self_recognition so the
# persona experiment can share the same verified method. ANSWER_PRIMER is
# imported above; this experiment re-runs against it as a regression test.


def _load_yaml() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _model_slug(model_name: str) -> str:
    """Short, filesystem-safe tag for a model name (last path segment)."""
    return (model_name or "model").split("/")[-1].replace(".", "-")


class PaperReplication(BaseExperiment):
    """Panickssery et al. (2024) out-of-the-box self-recognition replication."""

    def __init__(self, config: RunConfig, personas: dict[str, PersonaConfig],
                 config_name: str = ""):
        super().__init__(config)
        self._config_name = config_name
        self.backend: HFBackend | None = None
        self.results_logger: ResultsLogger | None = None

        yaml_data = _load_yaml()
        self.prompts: dict[str, str] = yaml_data.get("prompts") or {}
        for key in ("detection", "comparison", "recognition", "scoring",
                    "detection_system", "comparison_system", "recognition_system",
                    "scoring_system"):
            if key not in self.prompts:
                raise ValueError(f"config.yaml prompts must define `{key}`")

        run_cfg = (yaml_data.get("configs") or {}).get(config_name) or {}
        self.dataset: str = run_cfg.get("dataset", "xsum")
        # The evaluator's own text is this source's released summaries.
        self.evaluator: str = run_cfg.get("evaluator", "llama")
        self.sources: list[str] = list(run_cfg.get("sources") or [])
        if self.evaluator not in self.sources:
            raise ValueError(
                f"evaluator source {self.evaluator!r} must be in sources {self.sources}"
            )
        self.phases: list[str] = list(run_cfg.get("phases") or ALL_PHASES)
        bad = set(self.phases) - set(ALL_PHASES)
        if bad:
            raise ValueError(f"unknown phases {sorted(bad)}; allowed {ALL_PHASES}")
        self.start_index: int = int(run_cfg.get("start_index", 0))
        # sample_size / batch_size come from RunConfig so CLI --override applies.
        self.sample_size = self.config.sample_size
        self.batch_size = max(1, int(getattr(self.config, "batch_size", 1) or 1))

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [self.run_id]
        if config_name:
            parts.append(config_name)
        parts += [self.dataset, _model_slug(config.model_name)]
        self.run_dir = Path(config.output_dir) / "_".join(parts)
        self.output_path = self.run_dir / "trials.jsonl"
        self.manifest_path = self.run_dir / "manifest.txt"

        # Loaded in setup(): articles[key] -> str, summaries[source][key] -> str.
        self.articles: dict[str, str] = {}
        self.summaries: dict[str, dict[str, str]] = {}
        self.keys: list[str] = []
        # First example rendered prompt captured per phase, for the manifest.
        self._examples: dict[str, dict] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        logger.info(f"Loading evaluator model: {self.config.model_name}")
        self.backend = HFBackend(self.config.model_name)

        art_path = DATA_DIR / "articles" / f"{self.dataset}_train_articles.json"
        with open(art_path) as f:
            self.articles = json.load(f)
        for src in self.sources:
            sp = DATA_DIR / "summaries" / self.dataset / f"{self.dataset}_train_{src}_responses.json"
            with open(sp) as f:
                self.summaries[src] = json.load(f)

        all_keys = list(self.articles)
        end = None if self.sample_size is None else self.start_index + self.sample_size
        self.keys = all_keys[self.start_index:end]
        logger.info(
            f"dataset={self.dataset} sources={self.sources} evaluator={self.evaluator} "
            f"keys={len(self.keys)} (from index {self.start_index})"
        )

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_logger = ResultsLogger(self.output_path)

    def run(self) -> None:
        assert self.backend is not None and self.results_logger is not None
        if "detection" in self.phases:
            self._run_pairwise("detection")
        if "comparison" in self.phases:
            self._run_pairwise("comparison")
        if "recognition" in self.phases:
            self._run_recognition()
        if "scoring" in self.phases:
            self._run_scoring()
        self._write_manifest()

    # ── Constrained-choice probability helper (batched) ───────────────────

    def _render(self, messages) -> str:
        """Chat-template the messages and append the assistant-turn primer (core helper)."""
        return render_with_primer(self.backend, messages)

    def _probs_many(self, messages_list, choices):
        """Constrained-choice probabilities (last-sub-token softmax), batched.

        Delegates to core.self_recognition.choice_probs_batch — the validated
        reader shared with the persona experiment."""
        return choice_probs_batch(self.backend, messages_list, choices, self.batch_size)

    def _messages(self, system_key: str, user: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.prompts[system_key].strip()},
            {"role": "user", "content": user},
        ]

    # ── Pairwise phases: detection (recognition) & comparison (preference) ──

    def _run_pairwise(self, phase: str) -> None:
        """phase ∈ {detection, comparison}. Self is the evaluator's own source;
        every other source is the alternative. Each pair is run in both orders
        with self as Summary1 (forward) and Summary2 (backward)."""
        system_key = f"{phase}_system"
        template = self.prompts[phase]
        others = [s for s in self.sources if s != self.evaluator]

        # Build (key, other) work items and their forward/backward messages.
        items, msgs = [], []
        for key in self.keys:
            self_sum = self.summaries[self.evaluator][key]
            article = self.articles[key]
            for other in others:
                other_sum = self.summaries[other][key]
                fwd = template.format(article=article, summary1=self_sum, summary2=other_sum)
                bwd = template.format(article=article, summary1=other_sum, summary2=self_sum)
                items.append((key, other))
                msgs.append(self._messages(system_key, fwd))   # forward (self = "1")
                msgs.append(self._messages(system_key, bwd))   # backward (self = "2")

        probs = self._probs_many(msgs, ["1", "2"])
        for idx, (key, other) in enumerate(tqdm(items, desc=phase, dynamic_ncols=True, mininterval=0.5)):
            p_fwd, p_bwd = probs[2 * idx], probs[2 * idx + 1]
            # Self is option "1" forward and option "2" backward; average the
            # probability mass on the self-summary across orders (core helper).
            score = averaged_pairwise_score(p_fwd, p_bwd, self_fwd="1", self_bwd="2")
            self._examples.setdefault(phase, {
                "system": self.prompts[system_key].strip(),
                "user_forward": msgs[2 * idx][-1]["content"],
                "probs_forward": p_fwd, "probs_backward": p_bwd, "score": score,
            })
            self.results_logger.log_trial(PaperReplicationRecord(
                experiment="paper_replication",
                phase=phase,
                model=self.config.model_name,
                dataset=self.dataset,
                run_id=self.run_id,
                key=key,
                evaluator=self.evaluator,
                other_source=other,
                probs_forward=p_fwd,
                probs_backward=p_bwd,
                score=score,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

    # ── Individual phase: recognition (Yes/No) ─────────────────────────────

    def _run_recognition(self) -> None:
        template = self.prompts["recognition"]
        items, msgs = [], []
        for key in self.keys:
            article = self.articles[key]
            for src in self.sources:
                user = template.format(article=article, summary=self.summaries[src][key])
                items.append((key, src))
                msgs.append(self._messages("recognition_system", user))

        probs = self._probs_many(msgs, ["Yes", "No"])
        if items:
            self._examples["recognition"] = {
                "system": self.prompts["recognition_system"].strip(),
                "user": msgs[0][-1]["content"], "probs": probs[0], "score": probs[0].get("Yes", 0.0),
            }
        for (key, src), p in zip(tqdm(items, desc="recognition", dynamic_ncols=True, mininterval=0.5), probs):
            score = p.get("Yes", 0.0)  # renormalized P(Yes); paper's recognition score
            self.results_logger.log_trial(PaperReplicationRecord(
                experiment="paper_replication",
                phase="recognition",
                model=self.config.model_name,
                dataset=self.dataset,
                run_id=self.run_id,
                key=key,
                evaluator=self.evaluator,
                target_source=src,
                probs=p,
                score=score,
                ground_truth=int(src == self.evaluator),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

    # ── Individual phase: scoring (Likert 1-5) ─────────────────────────────

    def _run_scoring(self) -> None:
        template = self.prompts["scoring"]
        choices = ["1", "2", "3", "4", "5"]
        items, msgs = [], []
        for key in self.keys:
            article = self.articles[key]
            for src in self.sources:
                user = template.format(article=article, summary=self.summaries[src][key])
                items.append((key, src))
                msgs.append(self._messages("scoring_system", user))

        probs = self._probs_many(msgs, choices)
        if items:
            self._examples["scoring"] = {
                "system": self.prompts["scoring_system"].strip(),
                "user": msgs[0][-1]["content"],
                "probs": probs[0], "score": sum(int(k) * v for k, v in probs[0].items()),
            }
        for (key, src), p in zip(tqdm(items, desc="scoring", dynamic_ncols=True, mininterval=0.5), probs):
            # Prob-weighted mean rating (paper: Likert averaged over the five
            # number-token probabilities).
            score = sum(int(k) * v for k, v in p.items())
            self.results_logger.log_trial(PaperReplicationRecord(
                experiment="paper_replication",
                phase="scoring",
                model=self.config.model_name,
                dataset=self.dataset,
                run_id=self.run_id,
                key=key,
                evaluator=self.evaluator,
                target_source=src,
                probs=p,
                score=score,
                ground_truth=int(src == self.evaluator),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

    # ── Manifest ────────────────────────────────────────────────────────────

    def _write_manifest(self) -> None:
        cfg = asdict(self.config)
        lines = [
            "PAPER REPLICATION — Panickssery, Bowman & Feng (2024) — RUN MANIFEST",
            "=" * 64,
            f"run_id:     {self.run_id}",
            f"started_at: {datetime.now(timezone.utc).isoformat()}",
            f"model:      {self.config.model_name}  (evaluator)",
            f"dataset:    {self.dataset}",
            f"evaluator self-source: {self.evaluator}",
            f"sources:    {self.sources}",
            f"phases:     {self.phases}",
            f"keys:       {len(self.keys)} (start_index={self.start_index}, sample_size={self.sample_size})",
            f"batch_size: {self.batch_size}",
            f"seed:       {cfg.get('seed')}",
            f"answer_primer (appended after chat template): {ANSWER_PRIMER!r}",
            "",
            "PROMPT TEMPLATES (verbatim from reference repo)",
            "-" * 64,
        ]
        for key in ("detection", "comparison", "recognition", "scoring"):
            lines += [f"[{key}_system]", self.prompts[f"{key}_system"].strip(),
                      f"[{key}]", self.prompts[key].rstrip(), ""]
        lines += ["EXAMPLE RENDERED TRIALS (first per phase)", "-" * 64]
        for phase in self.phases:
            ex = self._examples.get(phase)
            lines.append(f"--- {phase} ---")
            if not ex:
                lines += ["(no trials captured)", ""]
                continue
            if phase in ("detection", "comparison"):
                lines += [
                    f"[system]\n{ex['system']}",
                    f"[user — forward order, self = Summary1]\n{ex['user_forward'].rstrip()}",
                    f"[probs forward] {ex['probs_forward']}",
                    f"[probs backward] {ex['probs_backward']}",
                    f"[score = 0.5*(P_fwd(1)+P_bwd(2))] {ex['score']:.4f}", "",
                ]
            else:
                lines += [
                    f"[system]\n{ex['system']}",
                    f"[probs] {ex['probs']}",
                    f"[score] {ex['score']:.4f}", "",
                ]
        self.manifest_path.write_text("\n".join(lines) + "\n")

    # ── Evaluation / saving ───────────────────────────────────────────────

    def evaluate(self) -> dict:
        from experiments.paper_replication.paper_replication_analysis import summarize_run
        return summarize_run(self.output_path, self.run_dir)

    def save_results(self) -> str:
        return str(self.output_path)
