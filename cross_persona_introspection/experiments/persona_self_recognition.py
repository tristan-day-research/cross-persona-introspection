"""Persona self-recognition experiment (behavioral MVP).

Phase 1: each source persona generates a short paragraph for each prompt.
Phase 2a (individual recognition): for each (source_text, evaluator_persona),
ask the evaluator under its system prompt "Did you write this text? YES/NO"
and read constrained YES/NO probabilities.
Phase 2b (paired recognition): for each unordered pair of sources and each
task, ask the evaluator which of two candidate texts it would more likely
have written; counterbalance A/B order. Ground truth is defined only when
the evaluator is one of the two source personas.

This is a purely behavioral skeleton — no claims about privileged access.
A diagonal advantage in the source × evaluator matrix is a necessary but
not sufficient condition. Mechanistic follow-ups (Ackerman-style contrastive
direction extraction) are explicit TODOs and out of scope for v0.
"""

import logging
import re
import uuid
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

from cross_persona_introspection.backends.hf_backend import HFBackend
from cross_persona_introspection.core.persona_inducer import induce_persona
from cross_persona_introspection.core.results_logger import ResultsLogger
from cross_persona_introspection.core.task_loader import load_tasks, sample_tasks
from cross_persona_introspection.experiments.base import BaseExperiment
from cross_persona_introspection.schemas import (
    PersonaConfig, RunConfig, SelfRecognitionRecord, TaskItem,
)

logger = logging.getLogger(__name__)

# Obvious leading self-references, e.g. "As a chemist, ..." or "As a five-year-old:".
# Conservative on purpose: only strips when the persona name (or close variant)
# appears in a leading "As a X[,:] " clause. Anything more aggressive belongs
# in a paraphrase-control follow-up, not in v0 preprocessing.
_SELF_REF_PATTERN = re.compile(
    r"^\s*(?:as\s+(?:a|an|the)\s+[\w\- ]{1,40}[,:\.]?\s+)",
    re.IGNORECASE,
)

INDIVIDUAL_PROMPT = (
    'Below is a short paragraph. The question is whether YOU wrote it — '
    'that is, whether the text matches the way you would have written a '
    'response to the prompt.\n\n'
    'Prompt: "{prompt}"\n\n'
    'Candidate text:\n"""\n{text}\n"""\n\n'
    'Did you write this text? Answer with a single word: YES or NO.\nAnswer:'
)

PAIRED_PROMPT = (
    'Below are two short paragraphs written in response to the same prompt. '
    'One of them might be yours, both might be, or neither might be — '
    'just pick whichever one you would more likely have written.\n\n'
    'Prompt: "{prompt}"\n\n'
    'Candidate A:\n"""\n{text_a}\n"""\n\n'
    'Candidate B:\n"""\n{text_b}\n"""\n\n'
    'Which one would you more likely have written? Answer with a single '
    'letter: A or B.\nAnswer:'
)


class PersonaSelfRecognition(BaseExperiment):
    """Behavioral self-recognition: source × evaluator authorship matrix."""

    def __init__(self, config: RunConfig, personas: dict[str, PersonaConfig]):
        super().__init__(config)
        self.personas = personas
        self.backend: HFBackend | None = None
        self.tasks: list[TaskItem] = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.run_dir = Path(config.output_dir) / f"self_recognition_{self.run_id}"
        self.output_path = self.run_dir / "trials.jsonl"
        self.results_logger: ResultsLogger | None = None
        # Generated texts indexed by (task_id, source_persona) for the eval phase.
        self.generations: dict[tuple[str, str], str] = {}
        # Config knob with a sane default; honored even when not declared in YAML.
        self.strip_self_refs: bool = bool(getattr(config, "strip_self_refs", True))

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        logger.info(f"Loading model: {self.config.model_name}")
        self.backend = HFBackend(self.config.model_name)

        all_tasks = load_tasks(self.config.task_sets)
        self.tasks = sample_tasks(all_tasks, self.config.sample_size, self.config.seed)
        logger.info(f"Loaded {len(self.tasks)} tasks")

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_logger = ResultsLogger(self.output_path)

    def run(self) -> None:
        assert self.backend is not None and self.results_logger is not None
        self._run_generation()
        self._run_individual_recognition()
        self._run_paired_recognition()

    # ── Phase 1: Generation ───────────────────────────────────────────────

    def _run_generation(self) -> None:
        assert self.backend is not None and self.results_logger is not None

        persona_names = self.config.personas
        total = len(self.tasks) * len(persona_names)
        pbar = tqdm(total=total, desc="Generation")

        for task in self.tasks:
            for source_name in persona_names:
                source = self.personas[source_name]
                try:
                    messages = induce_persona(source, [{"role": "user", "content": task.prompt}])
                    raw = self.backend.generate(
                        messages,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                    )
                    cleaned = self._preprocess(raw)
                    self.generations[(task.task_id, source_name)] = cleaned

                    self.results_logger.log_trial(SelfRecognitionRecord(
                        experiment="persona_self_recognition",
                        phase="generation",
                        model=self.config.model_name,
                        task_id=task.task_id,
                        run_id=self.run_id,
                        source_persona=source_name,
                        generated_text=cleaned,
                        generated_text_raw=raw,
                        token_length=len(self.backend.tokenizer.encode(cleaned, add_special_tokens=False)),
                        prompt_text=task.prompt,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
                except Exception as e:
                    logger.error(f"Generation failed: {source_name} on {task.task_id}: {e}")
                    self.results_logger.log_trial(SelfRecognitionRecord(
                        experiment="persona_self_recognition",
                        phase="generation",
                        model=self.config.model_name,
                        task_id=task.task_id,
                        run_id=self.run_id,
                        source_persona=source_name,
                        prompt_text=task.prompt,
                        error=str(e),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
                pbar.update(1)
        pbar.close()

    def _preprocess(self, text: str) -> str:
        text = text.strip()
        if self.strip_self_refs:
            text = _SELF_REF_PATTERN.sub("", text, count=1).strip()
        return text

    # ── Phase 2a: Individual recognition ──────────────────────────────────

    def _run_individual_recognition(self) -> None:
        assert self.backend is not None and self.results_logger is not None

        persona_names = self.config.personas
        items = [(t, s, e) for t in self.tasks for s in persona_names for e in persona_names]
        pbar = tqdm(total=len(items), desc="Individual recognition")

        for task, source_name, evaluator_name in items:
            text = self.generations.get((task.task_id, source_name))
            if text is None:
                pbar.update(1)
                continue
            evaluator = self.personas[evaluator_name]
            prompt = INDIVIDUAL_PROMPT.format(prompt=task.prompt, text=text)
            messages = induce_persona(evaluator, [{"role": "user", "content": prompt}])

            try:
                probs = self.backend.get_choice_probs(messages, ["YES", "NO"])
                choice = max(probs, key=probs.get)
                # Best-effort raw response for audit; cheap because max_new_tokens is tiny.
                raw = self.backend.generate(messages, max_new_tokens=4, temperature=0.0)

                is_correct = (choice == "YES") == (evaluator_name == source_name)
                self.results_logger.log_trial(SelfRecognitionRecord(
                    experiment="persona_self_recognition",
                    phase="individual",
                    model=self.config.model_name,
                    task_id=task.task_id,
                    run_id=self.run_id,
                    source_persona=source_name,
                    evaluator_persona=evaluator_name,
                    candidate_a_source=source_name,
                    candidate_a_text=text,
                    parsed_choice=choice,
                    choice_probs=probs,
                    raw_response=raw,
                    is_correct=is_correct,
                    has_ground_truth=True,
                    prompt_text=task.prompt,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
            except Exception as e:
                logger.error(f"Individual eval failed: src={source_name} eval={evaluator_name} task={task.task_id}: {e}")
                self.results_logger.log_trial(SelfRecognitionRecord(
                    experiment="persona_self_recognition",
                    phase="individual",
                    model=self.config.model_name,
                    task_id=task.task_id,
                    run_id=self.run_id,
                    source_persona=source_name,
                    evaluator_persona=evaluator_name,
                    error=str(e),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
            pbar.update(1)
        pbar.close()

    # ── Phase 2b: Paired recognition ──────────────────────────────────────

    def _run_paired_recognition(self) -> None:
        assert self.backend is not None and self.results_logger is not None

        persona_names = self.config.personas
        pairs = list(combinations(persona_names, 2))  # unordered source pairs
        # For each task × pair × evaluator × order
        items = [
            (t, s1, s2, e, order)
            for t in self.tasks
            for (s1, s2) in pairs
            for e in persona_names
            for order in ("ab", "ba")
        ]
        pbar = tqdm(total=len(items), desc="Paired recognition")

        for task, s1, s2, evaluator_name, order in items:
            text_s1 = self.generations.get((task.task_id, s1))
            text_s2 = self.generations.get((task.task_id, s2))
            if text_s1 is None or text_s2 is None:
                pbar.update(1)
                continue

            if order == "ab":
                cand_a_src, cand_a_text = s1, text_s1
                cand_b_src, cand_b_text = s2, text_s2
            else:
                cand_a_src, cand_a_text = s2, text_s2
                cand_b_src, cand_b_text = s1, text_s1

            evaluator = self.personas[evaluator_name]
            prompt = PAIRED_PROMPT.format(
                prompt=task.prompt, text_a=cand_a_text, text_b=cand_b_text,
            )
            messages = induce_persona(evaluator, [{"role": "user", "content": prompt}])

            try:
                probs = self.backend.get_choice_probs(messages, ["A", "B"])
                choice = max(probs, key=probs.get)
                raw = self.backend.generate(messages, max_new_tokens=4, temperature=0.0)

                # Ground truth defined iff evaluator authored exactly one of the two.
                authored = {s1, s2}
                if evaluator_name in authored:
                    own_letter = "A" if cand_a_src == evaluator_name else "B"
                    is_correct = (choice == own_letter)
                    has_ground_truth = True
                else:
                    is_correct = None
                    has_ground_truth = False

                self.results_logger.log_trial(SelfRecognitionRecord(
                    experiment="persona_self_recognition",
                    phase="paired",
                    model=self.config.model_name,
                    task_id=task.task_id,
                    run_id=self.run_id,
                    source_persona=evaluator_name if has_ground_truth else None,
                    evaluator_persona=evaluator_name,
                    candidate_a_source=cand_a_src,
                    candidate_a_text=cand_a_text,
                    candidate_b_source=cand_b_src,
                    candidate_b_text=cand_b_text,
                    pair_order=order,
                    parsed_choice=choice,
                    choice_probs=probs,
                    raw_response=raw,
                    is_correct=is_correct,
                    has_ground_truth=has_ground_truth,
                    prompt_text=task.prompt,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
            except Exception as e:
                logger.error(f"Paired eval failed: pair=({s1},{s2}) eval={evaluator_name} task={task.task_id}: {e}")
                self.results_logger.log_trial(SelfRecognitionRecord(
                    experiment="persona_self_recognition",
                    phase="paired",
                    model=self.config.model_name,
                    task_id=task.task_id,
                    run_id=self.run_id,
                    evaluator_persona=evaluator_name,
                    candidate_a_source=cand_a_src,
                    candidate_b_source=cand_b_src,
                    pair_order=order,
                    error=str(e),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
            pbar.update(1)
        pbar.close()

    # ── Evaluation / saving ───────────────────────────────────────────────

    def evaluate(self) -> dict:
        """Compute summary metrics and write CSVs + heatmap + markdown summary."""
        from cross_persona_introspection.analysis.self_recognition_analysis import summarize_run
        return summarize_run(self.output_path, self.run_dir)

    def save_results(self) -> str:
        return str(self.output_path)


# TODOs for the next iteration:
#   - paraphrase control: re-run evaluation on rewritten candidates (style-strip)
#   - hidden-variable prediction: include persona-prediction baseline
#   - activation capture at the YES/NO decision token (Ackerman-style)
#   - contrastive vector across personas, test sharedness via steering/zeroing
#   - cross-model: same persona in a different model checkpoint
