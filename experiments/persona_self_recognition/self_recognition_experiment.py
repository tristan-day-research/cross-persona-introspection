"""Persona self-recognition experiment (behavioral MVP).

Phase 1: each source persona generates a short paragraph for each prompt.
Phase 2a (individual recognition): for each (source_text, evaluator_persona),
ask the evaluator under its system prompt "Did you write this text? YES/NO"
and read constrained YES/NO probabilities.
Phase 2b (paired recognition): for each unordered pair of sources and each
task, ask the evaluator which of two candidate texts it would more likely
have written; counterbalance A/B order. Ground truth is defined only when
the evaluator is one of the two source personas.

All knobs (prompts, strip_self_refs, manifest examples count) live in
experiments/persona_self_recognition/self_recognition_config.yaml.
"""


import logging
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import yaml
from tqdm import tqdm

# If this many consecutive trials raise inside a phase, abort the whole run.
# This catches systemic infra failures (cuDNN init, OOM, driver) where every
# call fails identically — without it, the run silently produces a results
# file full of error rows. Set to 0 to disable.
CONSECUTIVE_FAILURE_LIMIT = 3

from core.backends.hf_backend import HFBackend
from core.persona_inducer import induce_persona
from core.results_logger import ResultsLogger
from core.task_loader import load_tasks, sample_tasks
from experiments.base import BaseExperiment
from core.schemas import (
    PersonaConfig, RunConfig, SelfRecognitionRecord, TaskItem,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "self_recognition_config.yaml"

# Obvious leading self-references, e.g. "As a chemist, ..." or "As a five-year-old:".
# Conservative on purpose: only strips a leading "As a X[,:] " clause. Anything
# more aggressive belongs in a paraphrase-control follow-up, not in v0 preprocessing.
_SELF_REF_PATTERN = re.compile(
    r"^\s*(?:as\s+(?:a|an|the)\s+[\w\- ]{1,40}[,:\.]?\s+)",
    re.IGNORECASE,
)


def _load_yaml() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


class PersonaSelfRecognition(BaseExperiment):
    """Behavioral self-recognition: source × evaluator authorship matrix."""

    def __init__(self, config: RunConfig, personas: dict[str, PersonaConfig]):
        super().__init__(config)
        self.personas = personas
        self.backend: HFBackend | None = None
        self.tasks: list[TaskItem] = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

        # ── Knobs read from config.yaml ───────────────────────────────────
        yaml_data = _load_yaml()
        self.prompts: dict[str, str] = yaml_data.get("prompts") or {}
        if "individual" not in self.prompts or "paired" not in self.prompts:
            raise ValueError("config.yaml must define `prompts.individual` and `prompts.paired`")
        # Per-target 3rd-person source prompts (used only when source_pov is "3rd_person")
        self.third_person_source_prompts: dict[str, dict] = (
            yaml_data.get("third_person_source_prompts") or {}
        )
        # Per-config overrides
        named_cfgs = yaml_data.get("configs") or {}
        # Fall back to first config in file if we can't tell which named one — usually
        # run.py has injected enough context. Strip-self-refs and example count are
        # cheap to read from any matching config: we look up by model_name as a heuristic.
        run_cfg = next(
            (c for c in named_cfgs.values() if c.get("model_name") == config.model_name),
            {},
        )
        self.strip_self_refs: bool = bool(run_cfg.get("strip_self_refs", True))
        self.n_manifest_examples: int = int(run_cfg.get("n_manifest_examples", 3))
        pov = str(run_cfg.get("source_pov", "1st_person")).lower()
        if pov not in ("1st_person", "3rd_person"):
            raise ValueError(
                f"source_pov must be '1st_person' or '3rd_person', got: {pov!r}"
            )
        self.source_pov: str = pov
        if self.source_pov == "3rd_person":
            missing = [
                p for p in config.personas
                if not (self.third_person_source_prompts.get(p) or {}).get("template")
            ]
            if missing:
                raise ValueError(
                    "source_pov is '3rd_person' but `third_person_source_prompts."
                    f"<persona>.template` is missing for: {missing}"
                )
        # Snapshot for the manifest header
        self._run_cfg_snapshot: dict = run_cfg

        pov_tag = "SOURCE_POV_3" if self.source_pov == "3rd_person" else "SOURCE_POV_1"
        self.run_dir = Path(config.output_dir) / f"self_recognition_{pov_tag}_{self.run_id}"
        self.output_path = self.run_dir / "trials.jsonl"
        self.manifest_path = self.run_dir / "manifest.txt"
        self.results_logger: ResultsLogger | None = None
        # Generated texts indexed by (task_id, induced_persona, target_persona).
        # In 1st_person mode, only diagonal entries (induced == target) are produced.
        # In 3rd_person mode, the full induced × target grid is produced — the
        # diagonal is the 1st-person comparison.
        self.generations: dict[tuple[str, str, str], str] = {}
        # Phases for which we have already appended examples to manifest.txt.
        self._manifest_phases_written: set[str] = set()
        # Eagerly captured example rendered-prompts per phase (filled during run).
        self._examples: dict[str, list[dict]] = {"generation": [], "individual": [], "paired": []}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        logger.info(f"Loading model: {self.config.model_name}")
        self.backend = HFBackend(self.config.model_name)

        all_tasks = load_tasks(self.config.task_sets)
        self.tasks = sample_tasks(all_tasks, self.config.sample_size, self.config.seed)
        logger.info(f"Loaded {len(self.tasks)} tasks")

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_logger = ResultsLogger(self.output_path)
        self._write_manifest_header()

    def run(self) -> None:
        assert self.backend is not None and self.results_logger is not None
        self._run_generation()
        self._flush_manifest_phase("generation")
        self._run_individual_recognition()
        self._flush_manifest_phase("individual")
        self._run_paired_recognition()
        self._flush_manifest_phase("paired")

    # ── Failure-tracking helper ───────────────────────────────────────────

    @staticmethod
    def _check_streak(streak: int, last_error: BaseException | None, phase: str) -> None:
        """Abort the run if we've hit the consecutive-failure limit.

        Called after each trial. `streak` is the count of consecutive failures
        in the current phase. `last_error` is the most recent exception (used
        to surface the underlying cause in the chained traceback).
        """
        if CONSECUTIVE_FAILURE_LIMIT and streak >= CONSECUTIVE_FAILURE_LIMIT:
            raise RuntimeError(
                f"{phase} aborted: {streak} consecutive trial failures. "
                f"This usually means a systemic GPU/driver/library issue, not a "
                f"per-trial bug — fix the root cause rather than retrying."
            ) from last_error

    # ── Phase 1: Generation ───────────────────────────────────────────────

    def _writer_identities(self) -> list[tuple[str, str]]:
        """All (induced, target) pairs we generate text for in this run.

        1st_person: only diagonal — each persona writes as itself.
        3rd_person: full grid — every persona writes as every persona, including
        the diagonal (which is the 1st-person comparison).
        """
        persona_names = self.config.personas
        if self.source_pov == "1st_person":
            return [(p, p) for p in persona_names]
        return [(induced, target) for induced in persona_names for target in persona_names]

    def _build_generation_messages(
        self, induced: PersonaConfig, target: PersonaConfig, prompt: str,
    ) -> list[dict[str, str]]:
        """Construct chat messages for one generation trial.

        Diagonal (induced == target) is always 1st-person: just the bare prompt.
        Off-diagonal uses the target persona's tailored entry under
        `third_person_source_prompts` to ask the induced persona to write *as
        if* it were the target.
        """
        if induced.name == target.name:
            user_content = prompt
        else:
            entry = self.third_person_source_prompts[target.name]
            template = entry["template"]
            user_content = template.format(prompt=prompt)
        return induce_persona(induced, [{"role": "user", "content": user_content}])

    def _run_generation(self) -> None:
        assert self.backend is not None and self.results_logger is not None

        identities = self._writer_identities()
        total = len(self.tasks) * len(identities)
        pbar = tqdm(total=total, desc="Generation", dynamic_ncols=True, mininterval=0.5)
        fail_streak = 0
        last_error: BaseException | None = None

        for task in self.tasks:
            for induced_name, target_name in identities:
                induced = self.personas[induced_name]
                target = self.personas[target_name]
                row_pov = "1st_person" if induced_name == target_name else "3rd_person"
                try:
                    messages = self._build_generation_messages(induced, target, task.prompt)
                    raw = self.backend.generate(
                        messages,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                    )
                    cleaned = self._preprocess(raw)
                    self.generations[(task.task_id, induced_name, target_name)] = cleaned

                    self.results_logger.log_trial(SelfRecognitionRecord(
                        experiment="persona_self_recognition",
                        phase="generation",
                        model=self.config.model_name,
                        task_id=task.task_id,
                        run_id=self.run_id,
                        source_persona=induced_name,
                        target_persona=target_name,
                        source_pov=row_pov,
                        generated_text=cleaned,
                        generated_text_raw=raw,
                        token_length=len(self.backend.tokenizer.encode(cleaned, add_special_tokens=False)),
                        prompt_text=task.prompt,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
                    if len(self._examples["generation"]) < self.n_manifest_examples:
                        # Recover the user-message we actually sent so it appears verbatim in the manifest.
                        rendered_user = messages[-1]["content"]
                        self._examples["generation"].append({
                            "task_id": task.task_id,
                            "source": induced_name,
                            "target": target_name,
                            "row_pov": row_pov,
                            "system": induced.system_prompt,
                            "user": rendered_user,
                            "model_output_clean": cleaned,
                            "model_output_raw": raw,
                        })
                    fail_streak = 0
                except Exception as e:
                    logger.error(
                        f"Generation failed: induced={induced_name} target={target_name} "
                        f"on {task.task_id}: {e}"
                    )
                    self.results_logger.log_trial(SelfRecognitionRecord(
                        experiment="persona_self_recognition",
                        phase="generation",
                        model=self.config.model_name,
                        task_id=task.task_id,
                        run_id=self.run_id,
                        source_persona=induced_name,
                        target_persona=target_name,
                        source_pov=row_pov,
                        prompt_text=task.prompt,
                        error=str(e),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
                    fail_streak += 1
                    last_error = e
                pbar.update(1)
                self._check_streak(fail_streak, last_error, "generation")
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
        identities = self._writer_identities()
        items = [(t, ind, tgt, e) for t in self.tasks for (ind, tgt) in identities for e in persona_names]
        pbar = tqdm(total=len(items), desc="Individual recognition", dynamic_ncols=True, mininterval=0.5)
        fail_streak = 0
        last_error: BaseException | None = None

        for task, induced_name, target_name, evaluator_name in items:
            text = self.generations.get((task.task_id, induced_name, target_name))
            if text is None:
                pbar.update(1)
                continue
            evaluator = self.personas[evaluator_name]
            user_msg = self.prompts["individual"].format(prompt=task.prompt, text=text)
            messages = induce_persona(evaluator, [{"role": "user", "content": user_msg}])
            row_pov = "1st_person" if induced_name == target_name else "3rd_person"

            try:
                probs = self.backend.get_choice_probs(messages, ["YES", "NO"])
                choice = max(probs, key=probs.get)
                raw = self.backend.generate(messages, max_new_tokens=4, temperature=0.0)

                # Ground truth: YES is correct iff the evaluator is the inducing
                # persona (the one whose system prompt was active when the text
                # was generated). target_persona is recorded so downstream
                # analysis can compute alternative notions of correctness
                # (e.g. style-recognition: evaluator == target).
                is_correct = (choice == "YES") == (evaluator_name == induced_name)
                if len(self._examples["individual"]) < self.n_manifest_examples:
                    self._examples["individual"].append({
                        "task_id": task.task_id,
                        "source": induced_name,
                        "target": target_name,
                        "row_pov": row_pov,
                        "evaluator": evaluator_name,
                        "system": evaluator.system_prompt,
                        "user": user_msg,
                        "probs": probs,
                        "choice": choice,
                        "is_correct": is_correct,
                        "raw": raw,
                    })
                self.results_logger.log_trial(SelfRecognitionRecord(
                    experiment="persona_self_recognition",
                    phase="individual",
                    model=self.config.model_name,
                    task_id=task.task_id,
                    run_id=self.run_id,
                    source_persona=induced_name,
                    target_persona=target_name,
                    source_pov=row_pov,
                    evaluator_persona=evaluator_name,
                    candidate_a_source=induced_name,
                    candidate_a_target=target_name,
                    candidate_a_text=text,
                    parsed_choice=choice,
                    choice_probs=probs,
                    raw_response=raw,
                    is_correct=is_correct,
                    has_ground_truth=True,
                    prompt_text=task.prompt,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
                fail_streak = 0
            except Exception as e:
                logger.error(
                    f"Individual eval failed: induced={induced_name} target={target_name} "
                    f"eval={evaluator_name} task={task.task_id}: {e}"
                )
                self.results_logger.log_trial(SelfRecognitionRecord(
                    experiment="persona_self_recognition",
                    phase="individual",
                    model=self.config.model_name,
                    task_id=task.task_id,
                    run_id=self.run_id,
                    source_persona=induced_name,
                    target_persona=target_name,
                    source_pov=row_pov,
                    evaluator_persona=evaluator_name,
                    error=str(e),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
                fail_streak += 1
                last_error = e
            pbar.update(1)
            self._check_streak(fail_streak, last_error, "individual")
        pbar.close()

    # ── Phase 2b: Paired recognition ──────────────────────────────────────

    def _run_paired_recognition(self) -> None:
        assert self.backend is not None and self.results_logger is not None

        persona_names = self.config.personas
        identities = self._writer_identities()
        pairs = list(combinations(identities, 2))
        items = [
            (t, id1, id2, e, order)
            for t in self.tasks
            for (id1, id2) in pairs
            for e in persona_names
            for order in ("ab", "ba")
        ]
        pbar = tqdm(total=len(items), desc="Paired recognition", dynamic_ncols=True, mininterval=0.5)
        fail_streak = 0
        last_error: BaseException | None = None

        for task, id1, id2, evaluator_name, order in items:
            text_s1 = self.generations.get((task.task_id, id1[0], id1[1]))
            text_s2 = self.generations.get((task.task_id, id2[0], id2[1]))
            if text_s1 is None or text_s2 is None:
                pbar.update(1)
                continue

            if order == "ab":
                cand_a_id, cand_a_text = id1, text_s1
                cand_b_id, cand_b_text = id2, text_s2
            else:
                cand_a_id, cand_a_text = id2, text_s2
                cand_b_id, cand_b_text = id1, text_s1
            cand_a_src, cand_a_tgt = cand_a_id
            cand_b_src, cand_b_tgt = cand_b_id

            evaluator = self.personas[evaluator_name]
            user_msg = self.prompts["paired"].format(
                prompt=task.prompt, text_a=cand_a_text, text_b=cand_b_text,
            )
            messages = induce_persona(evaluator, [{"role": "user", "content": user_msg}])

            try:
                probs = self.backend.get_choice_probs(messages, ["A", "B"])
                choice = max(probs, key=probs.get)
                raw = self.backend.generate(messages, max_new_tokens=4, temperature=0.0)

                # Ground truth: evaluator is one of the two authors (by induced
                # persona). When the same evaluator authored both candidates
                # (different targets in 3rd_person mode), authorship is
                # ambiguous and we skip ground-truth scoring for this row.
                authored = {cand_a_src, cand_b_src}
                if evaluator_name in authored and cand_a_src != cand_b_src:
                    own_letter = "A" if cand_a_src == evaluator_name else "B"
                    is_correct = (choice == own_letter)
                    has_ground_truth = True
                else:
                    is_correct = None
                    has_ground_truth = False

                if len(self._examples["paired"]) < self.n_manifest_examples:
                    self._examples["paired"].append({
                        "task_id": task.task_id,
                        "a_source": cand_a_src,
                        "a_target": cand_a_tgt,
                        "b_source": cand_b_src,
                        "b_target": cand_b_tgt,
                        "evaluator": evaluator_name,
                        "order": order,
                        "system": evaluator.system_prompt,
                        "user": user_msg,
                        "probs": probs,
                        "choice": choice,
                        "is_correct": is_correct,
                        "has_ground_truth": has_ground_truth,
                        "raw": raw,
                    })

                self.results_logger.log_trial(SelfRecognitionRecord(
                    experiment="persona_self_recognition",
                    phase="paired",
                    model=self.config.model_name,
                    task_id=task.task_id,
                    run_id=self.run_id,
                    source_persona=evaluator_name if has_ground_truth else None,
                    source_pov=self.source_pov,
                    evaluator_persona=evaluator_name,
                    candidate_a_source=cand_a_src,
                    candidate_a_target=cand_a_tgt,
                    candidate_a_text=cand_a_text,
                    candidate_b_source=cand_b_src,
                    candidate_b_target=cand_b_tgt,
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
                fail_streak = 0
            except Exception as e:
                logger.error(
                    f"Paired eval failed: a=({cand_a_src},{cand_a_tgt}) "
                    f"b=({cand_b_src},{cand_b_tgt}) eval={evaluator_name} "
                    f"task={task.task_id}: {e}"
                )
                self.results_logger.log_trial(SelfRecognitionRecord(
                    experiment="persona_self_recognition",
                    phase="paired",
                    model=self.config.model_name,
                    task_id=task.task_id,
                    run_id=self.run_id,
                    source_pov=self.source_pov,
                    evaluator_persona=evaluator_name,
                    candidate_a_source=cand_a_src,
                    candidate_a_target=cand_a_tgt,
                    candidate_b_source=cand_b_src,
                    candidate_b_target=cand_b_tgt,
                    pair_order=order,
                    error=str(e),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
                fail_streak += 1
                last_error = e
            pbar.update(1)
            self._check_streak(fail_streak, last_error, "paired")
        pbar.close()

    # ── Manifest writing ──────────────────────────────────────────────────

    def _write_manifest_header(self) -> None:
        """Top of manifest.txt: run metadata + config snapshot + verbatim prompt templates."""
        cfg = asdict(self.config)
        lines = [
            "PERSONA SELF-RECOGNITION — RUN MANIFEST",
            "=" * 60,
            f"run_id:     {self.run_id}",
            f"started_at: {datetime.now(timezone.utc).isoformat()}",
            f"model:      {self.config.model_name}",
            "",
            "CONFIG (from config.yaml + RunConfig)",
            "-" * 60,
        ]
        for k in (
            "personas", "task_sets", "sample_size", "seed",
            "max_new_tokens", "temperature", "output_dir",
        ):
            lines.append(f"  {k}: {cfg.get(k)}")
        lines.append(f"  strip_self_refs: {self.strip_self_refs}")
        lines.append(f"  n_manifest_examples: {self.n_manifest_examples}")
        lines.append(f"  source_pov: {self.source_pov}")
        lines += [
            "",
            "PERSONA SYSTEM PROMPTS",
            "-" * 60,
        ]
        for name in self.config.personas:
            sp = self.personas[name].system_prompt or "(empty)"
            lines.append(f"[{name}]")
            lines.append(sp.rstrip())
            lines.append("")
        lines += [
            "PROMPT TEMPLATES (verbatim from config.yaml)",
            "-" * 60,
            "[individual]",
            self.prompts["individual"].rstrip(),
            "",
            "[paired]",
            self.prompts["paired"].rstrip(),
            "",
        ]
        if self.source_pov == "3rd_person":
            lines += [
                "3RD-PERSON SOURCE PROMPTS (per target persona, verbatim)",
                "-" * 60,
            ]
            for name in self.config.personas:
                tmpl = (self.third_person_source_prompts.get(name) or {}).get("template", "")
                lines.append(f"[{name}]")
                lines.append(tmpl.rstrip() or "(missing)")
                lines.append("")
        self.manifest_path.write_text("\n".join(lines) + "\n")

    def _flush_manifest_phase(self, phase: str) -> None:
        """Append captured examples for one phase to manifest.txt. Idempotent."""
        if phase in self._manifest_phases_written:
            return
        examples = self._examples.get(phase) or []
        out: list[str] = ["", f"EXAMPLE TRIALS — {phase.upper()}", "=" * 60, ""]
        if not examples:
            out.append("(no successful trials captured)")
            out.append("")
        elif phase == "generation":
            for i, ex in enumerate(examples, 1):
                out += [
                    f"--- generation #{i} | task={ex['task_id']} | "
                    f"source={ex['source']} | target={ex['target']} | pov={ex['row_pov']} ---",
                    f"[system]\n{(ex['system'] or '').rstrip()}",
                    f"[user]\n{ex['user'].rstrip()}",
                    f"[model output (post-cleanup)]\n{ex['model_output_clean'].rstrip()}",
                    f"[model output (raw, pre-cleanup)]\n{ex['model_output_raw'].rstrip()}",
                    "",
                ]
        elif phase == "individual":
            for i, ex in enumerate(examples, 1):
                out += [
                    f"--- individual #{i} | task={ex['task_id']} | "
                    f"source={ex['source']} | target={ex['target']} | pov={ex['row_pov']} | "
                    f"evaluator={ex['evaluator']} ---",
                    f"[system]\n{(ex['system'] or '').rstrip()}",
                    f"[user]\n{ex['user'].rstrip()}",
                    f"[constrained probs] {ex['probs']}",
                    f"[parsed_choice] {ex['choice']}    [is_correct] {ex['is_correct']}",
                    f"[raw response] {ex['raw'].rstrip()}",
                    "",
                ]
        elif phase == "paired":
            for i, ex in enumerate(examples, 1):
                out += [
                    f"--- paired #{i} | task={ex['task_id']} | "
                    f"a=(induced={ex['a_source']}, target={ex['a_target']}) | "
                    f"b=(induced={ex['b_source']}, target={ex['b_target']}) | "
                    f"evaluator={ex['evaluator']} | order={ex['order']} ---",
                    f"[system]\n{(ex['system'] or '').rstrip()}",
                    f"[user]\n{ex['user'].rstrip()}",
                    f"[constrained probs] {ex['probs']}",
                    f"[parsed_choice] {ex['choice']}    [is_correct] {ex['is_correct']}    [has_ground_truth] {ex['has_ground_truth']}",
                    f"[raw response] {ex['raw'].rstrip()}",
                    "",
                ]

        with open(self.manifest_path, "a") as f:
            f.write("\n".join(out) + "\n")
        self._manifest_phases_written.add(phase)

    # ── Evaluation / saving ───────────────────────────────────────────────

    def evaluate(self) -> dict:
        """Compute summary metrics and write CSVs + heatmap + markdown summary."""
        from experiments.persona_self_recognition.self_recognition_analysis_helpers import summarize_run
        return summarize_run(self.output_path, self.run_dir)

    def save_results(self) -> str:
        return str(self.output_path)


# TODOs for the next iteration:
#   - paraphrase control: re-run evaluation on rewritten candidates (style-strip)
#   - hidden-variable prediction: include persona-prediction baseline
#   - activation capture at the YES/NO decision token (Ackerman-style)
#   - contrastive vector across personas, test sharedness via steering/zeroing
#   - cross-model: same persona in a different model checkpoint
