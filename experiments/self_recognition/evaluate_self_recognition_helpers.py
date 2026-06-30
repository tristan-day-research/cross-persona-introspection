"""Plumbing for the self-recognition evaluation, factored out of
evaluate_self_recognition.py so that file holds only the measurement logic and
the multi-GPU orchestration.

Nothing here is a measurement; it is the supporting machinery:

  * config parsing      — EvalOptions + which-measurements / which-groups resolution
  * generation IO       — locate & load the on-disk persona summaries (sources)
  * candidate framing    — per-trial shuffle + rendering for all_persona_descriptions
  * response parsing     — best-effort JSON extraction for the descriptions phase
  * manifest machinery  — example capture (single- & multi-GPU) and manifest.txt

Import direction is one-way: this module imports from core/ and the sibling
generate_text.py / evaluate_prompts.py only — never from
evaluate_self_recognition.py — so there is no import cycle.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.run_utils import git_commit, model_slug
from core.schemas import PersonaConfig, RunConfig
from core.self_recognition import ANSWER_PRIMER
from experiments.self_recognition.evaluate_prompts import (
    COMPARISON_TEMPLATE, DESCRIPTION_TEMPLATE, DETECTION_TEMPLATE,
    RECOGNITION_TEMPLATE, SCORING_TEMPLATE,
)
from experiments.self_recognition.evaluation_cases import (
    ALL_CASES as BINARY_CASES, CALIBRATION_BASE_CASES, CASE_REGISTRY,
    DEFAULT_CALIBRATION_BASE_CASE, OTHER_DESCRIPTION_STYLES, BinaryTrial,
    build_binary_prompt, build_case_trials, describe_other, expand_conditions,
    resolve_spec, resolve_specs,
)
from experiments.self_recognition.prompt_wordings import (
    DEFAULT_PROMPT_VERSION, PROMPT_VERSIONS,
)
from experiments.self_recognition.generate_text import DATA_DIR, task_root

logger = logging.getLogger(__name__)

# The default "groups" — the per-persona subfolders the eval reads
# ({persona}/{group}/summaries.json). In ARTICLE mode these are the XSUM/CNN
# datasets (the paper-replication default). In PROMPT mode they are the
# task-set names produced by generate_text.py (e.g. self_recognition_neutral),
# set via the eval config's `groups:` (or inferred from `task_sets:`). The
# classic paper phases need the source article and so run in article mode only;
# all_persona_descriptions needs only the generated text and works with any group.
DATASETS = ("xsum", "cnn")
# Groups for which a vendored <group>_articles.json exists (so the classic
# paper phases, which show the source article, can run).
ARTICLE_GROUPS = ("xsum", "cnn")

ARTICLES_DIR = DATA_DIR / "articles"

# Eval outputs land under results/text_evaluations/<task>/<run_dir>/.
TEXT_EVALUATIONS_DIR = Path(__file__).parent / "results" / "text_evaluations"

# The individually-selectable phases the eval can run. `measurements:` in the
# config lists any subset of these (each is one row-level `phase`):
#   pairwise_detection   pairwise self-recognition  ("which did you write?", 1/2)
#   pairwise_comparison  pairwise self-preference   ("which do you prefer?",  1/2)
#   recognition          individual self-recognition ("did you write this?", Yes/No)
#   scoring              individual self-preference  (Likert 1-5, prob-weighted mean)
#   all_persona_descriptions  multi-class persona-source classification (JSON)
ALL_PHASES = ("pairwise_detection", "pairwise_comparison", "recognition", "scoring",
              "all_persona_descriptions")
# Convenience family aliases accepted in `measurements:` — each expands to its
# member phases. The bare legacy names "detection"/"comparison" remain accepted
# as input shorthand for the (renamed) pairwise phases, so older configs still
# parse; only the emitted phase string changed. (Original family aliases were
# "pairwise"+"individual".)
MEASUREMENT_ALIASES = {
    "pairwise": ("pairwise_detection", "pairwise_comparison"),
    "individual": ("recognition", "scoring"),
    "detection": ("pairwise_detection",),
    "comparison": ("pairwise_comparison",),
}
DEFAULT_MEASUREMENTS = ("pairwise_detection", "pairwise_comparison",
                        "recognition", "scoring")

# Sentinel "evaluator" for the description-only condition of the persona-
# descriptions experiment: no persona is induced, the model sees only the
# candidate descriptions. Treated as its own work unit so it shards like any
# other evaluator. Chosen to never collide with a real persona name.
NEUTRAL_EVALUATOR = "__neutral__"

# Candidate-display ablation modes for all_persona_descriptions.
CANDIDATE_DISPLAY_MODES = ("full_prompt_anonymous_labels", "short_label_names")

# ── Binary 12-case eval ───────────────────────────────────────────────────────
# Every case definition (structure + wording) lives in evaluation_cases.py, the
# single source of truth; BINARY_CASES / OTHER_DESCRIPTION_STYLES / the trial and
# prompt builders are imported from there at the top of this file.


def expand_measurements(measurements) -> tuple[str, ...]:
    """Expand a `measurements:` list (phase names and/or family aliases) into the
    concrete set of phases to run, de-duplicated in canonical ALL_PHASES order.
    Raises ValueError on any unknown entry."""
    requested: set[str] = set()
    for m in measurements:
        if m in MEASUREMENT_ALIASES:
            requested.update(MEASUREMENT_ALIASES[m])
        elif m in ALL_PHASES:
            requested.add(m)
        else:
            valid = list(ALL_PHASES) + list(MEASUREMENT_ALIASES)
            raise ValueError(f"unknown measurement {m!r}; valid: {valid}")
    return tuple(p for p in ALL_PHASES if p in requested)


# ── Generation IO: locate & load the on-disk persona summaries ───────────────

def _load_articles(dataset: str) -> dict[str, str]:
    with open(ARTICLES_DIR / f"{dataset}_articles.json") as f:
        return json.load(f)


def _generations_root(run_config: RunConfig) -> Path:
    """Directory that directly contains the per-persona generation subfolders.

    Explicit `generations_filepath` in the config wins (so the same eval config
    can be pointed at any generation source); otherwise derive it from task +
    model: results/text_generations/<task>/<model_slug>/. Either way the eval
    expects {root}/{persona}/{dataset}/summaries.json.
    """
    if run_config.generations_filepath:
        return Path(run_config.generations_filepath)
    # If run_name is set, derive the generation folder automatically — same path
    # generate_text.py would have written to — so you don't need generations_filepath.
    subdir = run_config.output_subdir or run_config.run_name
    return task_root(run_config.task, run_config.model_name, subdir)


def _summaries_path(gen_root: Path, persona: str, dataset: str) -> Path:
    return gen_root / persona / dataset / "summaries.json"


def _load_summaries(gen_root: Path, persona: str, dataset: str) -> dict[str, str] | None:
    """Load one persona's summaries for a dataset, or None if not generated yet."""
    path = _summaries_path(gen_root, persona, dataset)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def check_generations_present(run_config: RunConfig, groups=DATASETS) -> None:
    """Pre-flight: every configured persona must have generations for every
    group before any evaluation runs. Raises FileNotFoundError listing the
    missing (persona, group) pairs so the user fixes the source up front
    rather than discovering it mid-run. `groups` are the per-persona subfolders
    expected under the generations root (datasets in article mode, task-set
    names in prompt mode)."""
    gen_root = _generations_root(run_config)
    missing = [
        (persona, g) for persona in run_config.personas for g in groups
        if not _summaries_path(gen_root, persona, g).exists()
    ]
    if missing:
        lines = "\n".join(f"  - {p}/{g}: {_summaries_path(gen_root, p, g)}"
                          for p, g in missing)
        raise FileNotFoundError(
            f"No generations found under {gen_root} for {len(missing)} "
            f"(persona, group) pair(s):\n{lines}\n"
            f"Generate them first (generate_text.py), point `generations_filepath` at "
            f"the right folder, fix `groups:`, or drop the persona(s) from the config."
        )


def _load_sources(run_config: RunConfig, gen_root: Path, dataset: str, start_index: int):
    """Load every configured persona's summaries for `dataset`, sliced to the
    same article window. Personas with no summaries on disk are skipped (warned);
    run() calls check_generations_present() up front so this normally never warns."""
    sources: dict[str, dict[str, str]] = {}
    for persona in run_config.personas:
        full = _load_summaries(gen_root, persona, dataset)
        if full is None:
            logger.warning(f"no summaries on disk for persona={persona} dataset={dataset}; skipping")
            continue
        keys = list(full)
        end = None if run_config.sample_size is None else start_index + run_config.sample_size
        sources[persona] = {k: full[k] for k in keys[start_index:end]}
    return sources


# ── Exact prompt/response rendering (for manifest example capture) ───────────

def _render_prompt_exact(backend, messages: list[dict], *, primer: str | None = None) -> str:
    """Full prompt string as the model tokenizes it (chat template + optional primer)."""
    text = backend.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if primer is not None:
        text += primer
    return text


def _decode_generation_exact(backend, messages: list[dict], *, max_new_tokens: int,
                           temperature: float = 0.0) -> tuple[str, str, str]:
    """Single-item generate returning (clean_response, prompt_exact, completion_exact).

    completion_exact preserves special/chat tokens; clean_response matches
    HFBackend.generate() (skip_special_tokens=True) and trials.jsonl.
    """
    import torch

    prompt_exact = _render_prompt_exact(backend, messages)
    input_ids = backend.tokenizer.encode(prompt_exact, return_tensors="pt").to(backend.input_device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output_ids = backend.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=backend.tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    clean = backend.tokenizer.decode(new_tokens, skip_special_tokens=True)
    completion_exact = backend.tokenizer.decode(new_tokens, skip_special_tokens=False)
    return clean, prompt_exact, completion_exact


# ── Manifest example capture ─────────────────────────────────────────────────

class ManifestExampleCollector:
    """Capture up to n examples per phase for manifest.txt (single- or multi-GPU)."""

    def __init__(self, n: int):
        self.n = n
        self.examples: dict[str, list[dict]] = {}

    def has_room(self, phase: str) -> bool:
        return len(self.examples.get(phase, [])) < self.n

    def maybe_add(self, phase: str, example: dict) -> None:
        if not self.has_room(phase):
            return
        self.examples.setdefault(phase, []).append(example)

    def to_json(self) -> dict:
        return self.examples

    @classmethod
    def from_json(cls, data: dict, n: int) -> "ManifestExampleCollector":
        col = cls(n)
        col.examples = {k: list(v) for k, v in (data or {}).items()}
        return col


def _merge_manifest_examples(run_dir: Path, num_shards: int, n_per_phase: int) -> dict[str, list]:
    """Merge per-shard example captures, keeping the first n_per_phase per phase."""
    merged: dict[str, list] = {}
    for i in range(num_shards):
        path = run_dir / "_shards" / f"manifest_examples_{i}.json"
        if not path.exists():
            continue
        shard_ex = json.loads(path.read_text())
        for phase, exs in shard_ex.items():
            bucket = merged.setdefault(phase, [])
            for ex in exs:
                if len(bucket) < n_per_phase:
                    bucket.append(ex)
    return merged


def _save_manifest_examples_shard(run_dir: Path, shard_index: int,
                                  collector: ManifestExampleCollector) -> None:
    path = run_dir / "_shards" / f"manifest_examples_{shard_index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collector.to_json(), indent=2))


# ── Eval options (config-derived knobs threaded to workers) ──────────────────

@dataclass
class EvalOptions:
    """Run-level knobs for which measurements run and how, read from the config
    by run() and threaded through to each shard worker so re-discovery in worker
    mode stays consistent.

      measurements              expanded tuple of phases to run (subset of
                                ALL_PHASES; family aliases already resolved).
      groups                    the per-persona subfolders to read
                                ({persona}/{group}/summaries.json): datasets in
                                article mode, task-set names in prompt mode.
      candidate_display_mode    all_persona_descriptions: how candidates are shown.
      description_max_new_tokens free-form JSON budget for the descriptions phase.
      seed                      base seed for the per-trial candidate shuffle.
      n_manifest_examples       verbatim prompt/response examples per phase in manifest.txt.
    """
    measurements: tuple[str, ...] = DEFAULT_MEASUREMENTS
    groups: tuple[str, ...] = DATASETS
    candidate_display_mode: str = "full_prompt_anonymous_labels"
    description_max_new_tokens: int = 256
    seed: int = 42
    n_manifest_examples: int = 3
    excl_self_condition: bool = False  # also run all_persona_descriptions with the
    # evaluator excluded from the candidate list and self-source trials skipped;
    # condition label: "active_persona_excl_self"
    # ── Binary 12-case eval (case1…case12) knobs ──
    # When cases_to_run is non-empty the run takes the BINARY path instead of the
    # legacy measurements path. Each case is self-contained: its evaluator-SP
    # state(s) and description treatment are intrinsic to the CaseSpec (see
    # evaluation_cases.py).
    cases_to_run: tuple[str, ...] = ()
    # Which prompt-wording version to render the cases in (see prompt_wordings.py).
    # "v1" = original "...under your current persona prompt" framing (reference for
    # prior runs); "v2" = natural first-person "did you write this?" framing.
    prompt_wording_version: str = DEFAULT_PROMPT_VERSION
    # Optional override of every describing case's description style (else each
    # case's own style; "not_applicable" for cases that describe nobody).
    description_style: str | None = None
    # case12 (calibration) mirrors this base case's structure/wording, restricted
    # to the calibration_personas subset (None = all configured personas).
    calibration_base_case: str = DEFAULT_CALIBRATION_BASE_CASE
    calibration_personas: tuple[str, ...] | None = None
    # The single cap on how many LOGICAL trials (text combinations × counterbalance)
    # each case uses: every case is randomly sampled down to this many (seeded by
    # sampling_seed). Cases that run in two evaluator-SP states (case5/case12) emit
    # this many trials PER state — i.e. each sampled example is measured both
    # active and neutral — so they produce 2× rows, not 2× examples.
    max_trials_per_case: int = 7200
    sampling_seed: int = 42
    save_logprobs: bool = True
    save_parquet: bool = False
    binary_free_generate: bool = False   # also free-generate a letter (raw_response
    # + parse_status); default off — the constrained A/B logit read is primary.
    binary_max_new_tokens: int = 5
    dry_run: bool = False
    # eval_dir — when set, all runs write into ONE stable collection directory
    #   (results/text_evaluations/<task>/<eval_dir>/) instead of a fresh timestamped
    #   folder, with one result file per case/condition SLICE. Lets you run a single
    #   case/condition per invocation and accumulate them in one place for analysis.
    # descriptions_filepath — optional persona_descriptions.json overriding the
    #   OTHER-persona descriptions (esp. the curated third_person_description) the
    #   eval shows; falls back to computing them from the persona system prompts.
    eval_dir: str | None = None
    descriptions_filepath: str | None = None
    # Activation capture (core.activation_capture / core.activation_store):
    #   collect_activations — capture residual-stream features at the spec'd named
    #     token positions, one unbatched forward per trial, into a sharded store.
    #   activations_dir — where to write (default: <run_dir>/activations); set a
    #     fixed path to accumulate one shared dataset across runs (joined by trial_id).
    #   activation_layers — override the captured layers (else depth-scaled defaults).
    collect_activations: bool = False
    activations_dir: str | None = None
    activation_layers: tuple[int, ...] | None = None
    activation_shard_size: int = 1000
    #   delete_local_activations — delete the local activation files after a
    #     successful R2 upload (ephemeral-pod friendly; no-op without R2).
    delete_local_activations: bool = False

    def runs(self, phase: str) -> bool:
        return phase in self.measurements

    def runs_any(self, *phases: str) -> bool:
        return any(p in self.measurements for p in phases)

    def runs_binary(self) -> bool:
        """Binary 12-case path active (cases_to_run non-empty)."""
        return bool(self.cases_to_run)

    def needs_articles(self) -> bool:
        """The classic paper phases show the source article; the descriptions and
        binary phases do not. Articles are loaded only when a classic phase runs."""
        if self.runs_binary():
            return False
        return self.runs_any("pairwise_detection", "pairwise_comparison",
                             "recognition", "scoring")


def _build_eval_options(exp_config: dict, run_config: RunConfig) -> EvalOptions:
    """Read the eval knobs from the raw config dict, validating choices.

    Two mutually-exclusive paths:
      * BINARY path — `cases_to_run:` is set (a subset of case1…case12). The run
        builds binary A/B trials; `measurements:` is ignored. Knobs:
        description_style (override), calibration_base_case, calibration_personas,
        max_trials_per_case, sampling_seed, save_logprobs, save_parquet,
        binary_free_generate, binary_max_new_tokens, dry_run.
      * LEGACY path — `measurements:` selects the paper / descriptions phases.

    `groups:` (both paths) are the per-persona subfolders to read (datasets in
    article mode, task-set names in prompt mode). Resolution order: explicit
    `groups:` → inferred from `task_sets:` → the xsum/cnn default.
    """
    groups = _resolve_groups(exp_config, run_config)

    cases = exp_config.get("cases_to_run")
    if cases:
        return _build_binary_eval_options(exp_config, run_config, groups, cases)

    measurements = exp_config.get("measurements") or list(DEFAULT_MEASUREMENTS)
    if isinstance(measurements, str):
        measurements = [measurements]
    phases = expand_measurements(measurements)  # validates + expands aliases
    if not phases:
        raise ValueError("measurements expanded to an empty set; nothing to run")
    mode = exp_config.get("candidate_display_mode", "full_prompt_anonymous_labels")
    if mode not in CANDIDATE_DISPLAY_MODES:
        raise ValueError(f"candidate_display_mode must be one of {list(CANDIDATE_DISPLAY_MODES)}, "
                         f"got {mode!r}")
    opts = EvalOptions(
        measurements=phases,
        groups=groups,
        candidate_display_mode=mode,
        description_max_new_tokens=int(exp_config.get("description_max_new_tokens", 256)),
        seed=int(run_config.seed),
        n_manifest_examples=int(exp_config.get("n_manifest_examples", 3)),
        excl_self_condition=bool(exp_config.get("excl_self_condition", False)),
        eval_dir=exp_config.get("eval_dir") or run_config.run_name,  # stable name wins over timestamp
    )
    # Classic paper phases show the source article, so they require article-backed
    # groups (xsum/cnn). Prompt-mode (task-set) groups support only descriptions.
    if opts.needs_articles():
        non_article = [g for g in groups if g not in ARTICLE_GROUPS]
        if non_article:
            raise ValueError(
                f"groups {non_article} have no source articles, but the requested "
                f"measurements include a classic paper phase that shows the article. "
                f"Use groups {list(ARTICLE_GROUPS)} for those phases, or restrict "
                f"`measurements:` to [all_persona_descriptions] for prompt-mode groups."
            )
    return opts


def _build_binary_eval_options(exp_config: dict, run_config: RunConfig,
                               groups: tuple[str, ...], cases) -> EvalOptions:
    """Build EvalOptions for the binary 12-case path, validating choices. Each
    case's evaluator-SP state and description treatment are intrinsic to its
    CaseSpec; the only knobs here are an optional description_style override and
    the case12 calibration base case / persona subset."""
    if isinstance(cases, str):
        cases = [cases]
    cases = tuple(str(c) for c in cases)
    bad = [c for c in cases if c not in BINARY_CASES]
    if bad:
        raise ValueError(f"cases_to_run has unknown case(s) {bad}; valid: {list(BINARY_CASES)}")

    description_style = exp_config.get("description_style")
    if description_style is not None:
        description_style = str(description_style)
        if description_style not in OTHER_DESCRIPTION_STYLES:
            raise ValueError(f"description_style {description_style!r} unknown; "
                             f"valid: {list(OTHER_DESCRIPTION_STYLES)}")

    calibration_base_case = str(exp_config.get("calibration_base_case", DEFAULT_CALIBRATION_BASE_CASE))
    if calibration_base_case not in CALIBRATION_BASE_CASES:
        raise ValueError(f"calibration_base_case {calibration_base_case!r} unknown; "
                         f"valid: {list(CALIBRATION_BASE_CASES)}")
    cal_personas = exp_config.get("calibration_personas")
    if isinstance(cal_personas, str):
        cal_personas = [cal_personas]
    cal_personas = tuple(str(p) for p in cal_personas) if cal_personas else None

    wording_version = str(exp_config.get("prompt_wording_version", DEFAULT_PROMPT_VERSION))
    if wording_version not in PROMPT_VERSIONS:
        raise ValueError(f"prompt_wording_version {wording_version!r} unknown; "
                         f"valid: {list(PROMPT_VERSIONS)}")

    return EvalOptions(
        measurements=(),
        groups=groups,
        seed=int(run_config.seed),
        n_manifest_examples=int(exp_config.get("n_manifest_examples", 3)),
        cases_to_run=cases,
        prompt_wording_version=wording_version,
        description_style=description_style,
        calibration_base_case=calibration_base_case,
        calibration_personas=cal_personas,
        max_trials_per_case=int(exp_config.get(
            "max_trials_per_case",
            exp_config.get("max_trials_per_case_condition", 7200))),  # legacy alias
        sampling_seed=int(exp_config.get("sampling_seed", run_config.seed)),
        save_logprobs=bool(exp_config.get("save_logprobs", True)),
        save_parquet=bool(exp_config.get("save_parquet", False)),
        binary_free_generate=bool(exp_config.get("binary_free_generate", False)),
        binary_max_new_tokens=int(exp_config.get("binary_max_new_tokens", 5)),
        dry_run=bool(exp_config.get("dry_run", False)),
        eval_dir=exp_config.get("eval_dir") or run_config.run_name,
        descriptions_filepath=exp_config.get("descriptions_filepath"),
        collect_activations=bool(exp_config.get("collect_activations", False)),
        activations_dir=exp_config.get("activations_dir"),
        activation_layers=(tuple(exp_config["activation_layers"])
                           if exp_config.get("activation_layers") else None),
        activation_shard_size=int(exp_config.get("activation_shard_size", 1000)),
        delete_local_activations=bool(exp_config.get("delete_local_activations", False)),
    )


def _resolve_groups(exp_config: dict, run_config: RunConfig) -> tuple[str, ...]:
    """The per-persona subfolders the eval reads. Resolution order:
      - explicit `groups:` list                                  → use as given;
      - `groups: all` (sentinel)                                 → auto-detect
        EVERY group present on disk under generations_filepath (see
        _detect_groups), so a run picks up whatever was generated;
      - inferred from `task_sets:` (the prompt-mode set names)   → those sets;
      - default (article mode)                                   → xsum/cnn.
    """
    groups = exp_config.get("groups")
    if groups:
        if isinstance(groups, str):
            if groups.lower() == "all":
                return _detect_groups(run_config)
            groups = [groups]
        elif isinstance(groups, list) and [str(g).lower() for g in groups] == ["all"]:
            return _detect_groups(run_config)
        return tuple(str(g) for g in groups)
    task_sets = exp_config.get("task_sets")
    if task_sets:
        from experiments.self_recognition.generate_text import _task_set_name
        return tuple(_task_set_name(e) for e in task_sets)
    return DATASETS


def _detect_groups(run_config: RunConfig) -> tuple[str, ...]:
    """Every group present on disk: the union of subfolders that contain a
    summaries.json across all configured personas under generations_filepath.
    Sorted for a stable order. Errors if nothing is found (wrong path / not
    generated yet) so the run fails fast with a clear message."""
    gen_root = _generations_root(run_config)
    found: set[str] = set()
    for persona in run_config.personas:
        pdir = gen_root / persona
        if not pdir.is_dir():
            continue
        for sub in pdir.iterdir():
            if (sub / "summaries.json").exists():
                found.add(sub.name)
    if not found:
        raise FileNotFoundError(
            f"groups: all found no <persona>/<group>/summaries.json under {gen_root}. "
            f"Check generations_filepath, or generate first."
        )
    return tuple(sorted(found))


# ── all_persona_descriptions: candidate framing ──────────────────────────────

def _labels_for(n: int) -> list[str]:
    """Answer labels A, B, C, … for n candidates (A–Z; n is small in practice)."""
    return [chr(ord("A") + i) for i in range(n)]


def _shuffle_candidates(candidate_names: list[str], opts: EvalOptions, *,
                        dataset: str, key: str, condition: str,
                        evaluator: str, source: str) -> dict[str, str]:
    """Deterministic per-trial {label: persona_name} mapping.

    Order is randomized per trial (so label position can't be exploited and
    position bias averages out across the dataset) but reproducible: seeded by
    the run seed plus the trial's identity. The mapping is logged on every row.
    """
    rng = random.Random(f"{opts.seed}|{dataset}|{key}|{condition}|{evaluator}|{source}")
    names = list(candidate_names)
    rng.shuffle(names)
    return dict(zip(_labels_for(len(names)), names))


def _render_candidates(mapping: dict[str, str], personas: dict[str, PersonaConfig],
                       mode: str) -> str:
    """Render the candidate block under labels A/B/C/D… per display mode.

    full_prompt_anonymous_labels — each label shows the candidate's full system
        prompt, with the persona NAME hidden (the main, name-agnostic condition).
    short_label_names — each label shows the candidate's short persona name (the
        semantic-hint ablation). Either way the {label: name} mapping is hidden
        from the model and recorded only in the row for analysis.
    """
    blocks = []
    for label, name in mapping.items():
        if mode == "short_label_names":
            blocks.append(f"[{label}] {name}")
        else:  # full_prompt_anonymous_labels (default)
            sp = (personas[name].system_prompt or "(no system prompt)").strip()
            blocks.append(f"[{label}] This assistant operated under the following "
                          f"system prompt:\n\"\"\"\n{sp}\n\"\"\"")
    return "\n\n".join(blocks)


def _build_description_user(text: str, mapping: dict[str, str],
                            personas: dict[str, PersonaConfig], mode: str) -> str:
    labels = list(mapping)
    prob_example = ", ".join(f'"{l}": <int>' for l in labels)
    return DESCRIPTION_TEMPLATE.format(
        text=text,
        candidates=_render_candidates(mapping, personas, mode),
        labels=", ".join(labels),
        prob_example=prob_example,
    )


# ── all_persona_descriptions: response parsing ───────────────────────────────

def _extract_json_object(raw: str | None) -> dict | None:
    """Best-effort parse of the first valid JSON object in a free-form response.

    Tolerates code fences and trailing prose: scans from the first '{' and tries
    progressively shorter spans ending at each '}', returning the first that
    json.loads cleanly. Returns None if nothing parses.
    """
    if not raw:
        return None
    start = raw.find("{")
    if start == -1:
        return None
    # Try the largest spans first (last '}' inward) so nested objects survive.
    end_positions = [i + 1 for i, ch in enumerate(raw) if ch == "}" and i > start]
    for end in reversed(end_positions):
        try:
            obj = json.loads(raw[start:end])
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _parse_description_response(raw: str | None, labels: list[str]) -> dict:
    """Parse the descriptions-phase JSON into normalized fields.

    Returns a dict with: probabilities ({label: prob} summing to 1, or None),
    raw_probabilities ({label: value} as emitted, or None), predicted_label,
    confidence, brief_reason, and parse_status ∈ {"ok","normalized","failed"}.

    "ok"        — JSON parsed and the emitted percentages summed to ~100.
    "normalized"— JSON parsed but the percentages were renormalized to sum to 1.
    "failed"    — no usable probabilities recovered.
    """
    out = {"probabilities": None, "raw_probabilities": None, "predicted_label": None,
           "confidence": None, "brief_reason": None, "parse_status": "failed"}
    obj = _extract_json_object(raw)
    if obj is None:
        return out

    out["brief_reason"] = obj.get("brief_reason")
    conf = obj.get("confidence")
    if isinstance(conf, (int, float)) and 0 <= conf <= 100:
        out["confidence"] = float(conf)

    probs = obj.get("probabilities")
    if not isinstance(probs, dict):
        return out
    # Keep only the valid labels; coerce numeric values, drop the rest.
    raw_vals: dict[str, float] = {}
    for lab in labels:
        v = probs.get(lab)
        if isinstance(v, (int, float)) and v >= 0:
            raw_vals[lab] = float(v)
    if not raw_vals or sum(raw_vals.values()) <= 0:
        return out
    out["raw_probabilities"] = raw_vals
    total = sum(raw_vals.values())
    out["probabilities"] = {lab: raw_vals.get(lab, 0.0) / total for lab in labels}
    # ~100 (allow integer rounding / a couple points of slack) → "ok".
    out["parse_status"] = "ok" if abs(total - 100.0) <= 2.0 else "normalized"

    ml = obj.get("most_likely")
    if isinstance(ml, str) and ml in out["probabilities"]:
        out["predicted_label"] = ml
    else:
        out["predicted_label"] = max(out["probabilities"], key=out["probabilities"].get)
    return out


# ── Manifest writing ─────────────────────────────────────────────────────────

def _write_manifest(run_dir: Path, run_config: RunConfig, evaluators: list[str],
                    personas: dict[str, PersonaConfig], opts: EvalOptions, run_id: str,
                    examples: dict[str, list] | None = None) -> None:
    lines = [
        "SELF-RECOGNITION EVALUATION — MANIFEST",
        "=" * 72,
        f"run_id:         {run_id}",
        f"generated_at:   {datetime.now(timezone.utc).isoformat()}",
        f"git_commit:     {git_commit()}",
        f"model_name:     {run_config.model_name}  (evaluator = generator)",
        f"model_slug:     {model_slug(run_config.model_name)}",
        f"adapter:        {run_config.adapter or '(none)'}",
        f"task:           {run_config.task}",
        f"generations:    {_generations_root(run_config)}",
        f"groups:         {list(opts.groups)}",
        f"sample_size:    {run_config.sample_size}",
        f"batch_size:     {run_config.batch_size}",
        f"measurements:   {list(opts.measurements)}",
        f"n_manifest_examples: {opts.n_manifest_examples}",
        f"source personas:    {run_config.personas}",
        f"evaluator personas: {evaluators}",
        "",
        "METHOD",
        "-" * 72,
        "Evaluator = model induced with persona E (system prompt). 'self' = E's own",
        "summaries; alternatives = other personas' summaries. Pairwise score is the",
        "ordering-bias-corrected averaged logprob 0.5*(P_fwd(self)+P_bwd(self)) over",
        "both option orders, via core.self_recognition (same code that reproduces the",
        "paper). Rows use the persona vocabulary (evaluator_persona / source_persona)",
        "consistent with the rest of the self_recognition experiment; the analysis",
        "notebook's appendix reads them.",
        "",
    ]
    if opts.runs_any("pairwise_detection", "pairwise_comparison", "recognition", "scoring"):
        lines += [
            "PROMPT TEMPLATES (paper Appendix A — only classic phases in this run)",
            "-" * 72,
        ]
        if opts.runs("pairwise_detection"):
            lines += ["[pairwise_detection user template]\n" + DETECTION_TEMPLATE, ""]
        if opts.runs("pairwise_comparison"):
            lines += ["[pairwise_comparison user template]\n" + COMPARISON_TEMPLATE, ""]
        if opts.runs("recognition"):
            lines += ["[recognition user template]\n" + RECOGNITION_TEMPLATE, ""]
        if opts.runs("scoring"):
            lines += ["[scoring user template]\n" + SCORING_TEMPLATE, ""]
        lines += [
            "Classic phases read the answer at the token after the assistant primer",
            f'"{ANSWER_PRIMER.strip()}" (see core.self_recognition.ANSWER_PRIMER).',
            "",
        ]
    if opts.runs("all_persona_descriptions"):
        lines += [
            "ALL_PERSONA_DESCRIPTIONS (multi-class persona-source classification)",
            "-" * 72,
            f"candidate_display_mode:     {opts.candidate_display_mode}",
            f"description_max_new_tokens: {opts.description_max_new_tokens}",
            f"shuffle_seed:               {opts.seed}",
            "conditions: active_persona (each evaluator persona induced) + "
            "description_only (neutral, no persona).",
            "Each persona summary is one generated text; the evaluator assigns a JSON",
            "probability distribution over a shuffled candidate set (labelled A/B/C/D…).",
            "The hidden {label: persona} mapping is randomized per trial and logged on",
            "every row. One row per evaluation (PersonaDescriptionEvalRecord).",
            "",
            "[all_persona_descriptions user template]\n" + DESCRIPTION_TEMPLATE, "",
        ]
    lines += [
        "EVALUATOR PERSONA SYSTEM PROMPTS (verbatim)",
        "-" * 72,
    ]
    for e in evaluators:
        lines += [f"[{e}]", (personas[e].system_prompt or "(none)").strip(), ""]
    lines += _format_manifest_examples(examples or {}, opts)
    (run_dir / "manifest.txt").write_text("\n".join(lines) + "\n")


def _format_manifest_examples(examples: dict[str, list], opts: EvalOptions) -> list[str]:
    """Render captured trials: exact chat-templated prompts and model outputs."""
    if not any(examples.get(p) for p in opts.measurements):
        return [
            "",
            "EXAMPLE TRIALS",
            "=" * 72,
            "(no examples captured — re-run with n_manifest_examples > 0 on a single GPU,",
            " or ensure workers wrote _shards/manifest_examples_*.json)",
            "",
        ]
    out: list[str] = ["", "EXAMPLE TRIALS (exact strings as tokenized / decoded)", "=" * 72, ""]
    phase_order = [p for p in ALL_PHASES if p in opts.measurements]
    for phase in phase_order:
        exs = examples.get(phase) or []
        out += [f"--- {phase.upper()} ({len(exs)} example(s)) ---", ""]
        if not exs:
            out += ["(no examples captured for this phase)", ""]
            continue
        for i, ex in enumerate(exs, 1):
            if phase == "all_persona_descriptions":
                out += [
                    f"example #{i} | group={ex.get('dataset')} | key={ex.get('key')} | "
                    f"condition={ex.get('condition')} | evaluator={ex.get('evaluator')} | "
                    f"source={ex.get('source')} | correct={ex.get('correct_label')} | "
                    f"mapping={ex.get('candidate_mapping')}",
                    "[prompt_exact — chat template + generation prompt, as tokenized]",
                    ex.get("prompt_exact", "").rstrip(),
                    "[completion_exact — new tokens only, skip_special_tokens=False]",
                    ex.get("completion_exact", "").rstrip(),
                    "[response_clean — as logged to trials.jsonl]",
                    ex.get("response_clean", "").rstrip(),
                    "",
                ]
            else:
                out += [
                    f"example #{i} | group={ex.get('dataset')} | key={ex.get('key')} | "
                    f"evaluator={ex.get('evaluator')}",
                ]
                if phase in ("pairwise_detection", "pairwise_comparison"):
                    out.append(
                        f"other_source={ex.get('other_source')} | order={ex.get('order')}"
                    )
                if phase in ("recognition", "scoring"):
                    out.append(f"source={ex.get('source')} | is_self={ex.get('is_self')}")
                out += [
                    "[prompt_exact — chat template + generation prompt + answer primer]",
                    ex.get("prompt_exact", "").rstrip(),
                    "[choice_probs — softmax over constrained next-token choices]",
                    str(ex.get("choice_probs")),
                    "",
                ]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Binary 12-case eval (case1…case12)
# ═══════════════════════════════════════════════════════════════════════════
# Plumbing for the binary path: load the generated texts into a lookup, enumerate
# + sample + expand trials (delegating the case logic to evaluation_cases),
# assemble the user prompt for a trial, parse a free-text A/B, and write the trial
# manifest, eval manifest, and optional Parquet. The measurement loop itself
# (build prompt → forward pass → record) lives in evaluate_self_recognition.py.

_AB_RE = re.compile(r"[AB]")


def _parse_ab_response(raw: str | None) -> tuple[str | None, str]:
    """Parse a free-text answer to a single letter. Returns (letter|None, status).

    Used only in the optional free-generation mode (binary_free_generate); the
    primary path reads constrained A/B logits and never needs this. Tolerates
    extra text by taking the first A/B character. status ∈ {"parsed","failed"}.
    """
    if not raw:
        return None, "failed"
    m = _AB_RE.search(raw.strip().upper())
    return (m.group(0), "parsed") if m else (None, "failed")


# ── Generation text lookup ───────────────────────────────────────────────────

def _load_text_map(run_config: RunConfig, gen_root: Path,
                   groups) -> dict[str, dict[str, dict[str, str]]]:
    """{group: {persona: {task_id: text}}} for every configured persona that has
    summaries on disk. Personas missing a group are simply absent there."""
    text_map: dict[str, dict[str, dict[str, str]]] = {}
    for g in groups:
        per_persona: dict[str, dict[str, str]] = {}
        for persona in run_config.personas:
            full = _load_summaries(gen_root, persona, g)
            if full is not None:
                per_persona[persona] = full
        text_map[g] = per_persona
    return text_map


def _group_personas(text_map_group: dict[str, dict[str, str]],
                    configured: list[str]) -> list[str]:
    """Configured personas that actually have summaries for this group (sorted)."""
    return sorted(p for p in configured if p in text_map_group)


def _common_keys(text_map_group: dict[str, dict[str, str]], personas: list[str],
                 sample_size: int | None) -> list[str]:
    """Task ids present for EVERY listed persona in this group (sorted, sliced to
    sample_size). Two-text cases need both authors to have the same task."""
    if not personas:
        return []
    common = set(text_map_group[personas[0]])
    for p in personas[1:]:
        common &= set(text_map_group[p])
    keys = sorted(common)
    return keys[:sample_size] if sample_size is not None else keys


def _generation_run_id(gen_root: Path) -> str:
    """Best-effort id for the source generation run: the run folder name (a
    timestamp) or the basename of generations_filepath."""
    return gen_root.name or str(gen_root)


# ── Trial enumeration over groups × cases (with sampling + condition expansion) ─

def enumerate_binary_trials(run_config: RunConfig, opts: EvalOptions,
                            text_map: dict) -> list[BinaryTrial]:
    """Every concrete binary trial for this run, in a stable order: for each group
    and case, enumerate+sample base trials then expand across the case's
    evaluator-SP state(s). base_trial_id embeds the group, so ids are unique
    across groups. case12 (calibration) is restricted to calibration_personas."""
    specs = resolve_specs(opts.cases_to_run, calibration_base_case=opts.calibration_base_case,
                          wording_version=opts.prompt_wording_version)
    cal_subset = set(opts.calibration_personas) if opts.calibration_personas else None
    trials: list[BinaryTrial] = []
    for group in opts.groups:
        group_personas = _group_personas(text_map.get(group, {}), run_config.personas)
        if len(group_personas) < 2:
            logger.warning(f"binary: <2 personas with summaries for group {group}; skipping")
            continue
        for case_id in opts.cases_to_run:
            spec = specs[case_id]
            personas = group_personas
            if case_id == "case12" and cal_subset is not None:
                personas = [p for p in group_personas if p in cal_subset]
            if len(personas) < 2:
                logger.warning(f"binary {case_id}: <2 eligible personas in group {group}; skipping")
                continue
            keys = _common_keys(text_map[group], personas, run_config.sample_size)
            if not keys:
                logger.warning(f"binary {case_id}: no common task keys in group {group}; skipping")
                continue
            base = build_case_trials(
                spec, group, keys, personas,
                sampling_seed=opts.sampling_seed,
                max_trials=opts.max_trials_per_case,
            )
            trials += expand_conditions(base, spec, description_style=opts.description_style)
    return trials


# ── Per-trial prompt assembly (texts looked up here, names never shown) ──────

def trial_text_fields(trial: BinaryTrial, text_map: dict) -> dict:
    """The text strings + stable generation ids this trial shows."""
    g = text_map[trial.group]
    if trial.case_id in ("case1", "case2"):
        src = trial.source_persona
        return {
            "single_text": g[src][trial.task_id], "text1": None, "text2": None,
            "text1_generation_id": f"{trial.group}/{src}/{trial.task_id}",
            "text2_generation_id": None,
        }
    t1s, t2s = trial.text1_source_persona, trial.text2_source_persona
    return {
        "single_text": None,
        "text1": g[t1s][trial.task_id], "text2": g[t2s][trial.task_id],
        "text1_generation_id": f"{trial.group}/{t1s}/{trial.task_id}",
        "text2_generation_id": f"{trial.group}/{t2s}/{trial.task_id}",
    }


def build_trial_prompt(trial: BinaryTrial, texts: dict,
                       personas: dict[str, PersonaConfig], *, spec,
                       descriptions: dict | None = None, return_spans: bool = False):
    """Assemble the user-turn prompt for one trial. `spec` is the trial's CaseSpec
    (which decides which persona(s) are described and in what style); persona
    names never appear.

    `descriptions` is the resolved {persona: {style: text}} map (see
    resolve_persona_descriptions). When None or missing an entry, the description
    is computed on the fly from the persona system prompt. With return_spans=True,
    also returns segment char-spans (for activation capture)."""
    style = trial.other_description_style

    def desc_for(name: str) -> str:
        if descriptions and name in descriptions and style in descriptions[name]:
            return descriptions[name][style]
        return describe_other(personas[name], style)

    desc_main = desc_1 = desc_2 = None
    if spec.describe == "other":
        desc_main = desc_for(trial.other_persona)
    elif spec.describe == "self":
        desc_main = desc_for(trial.evaluator_persona)
    elif spec.describe == "both":
        desc_1 = desc_for(trial.other_persona_1)
        desc_2 = desc_for(trial.other_persona_2)
    return build_binary_prompt(
        spec, answer_mapping=trial.answer_mapping,
        single_text=texts["single_text"], text1=texts["text1"], text2=texts["text2"],
        desc_main=desc_main, desc_1=desc_1, desc_2=desc_2,
        return_spans=return_spans,
    )


# ── Persona descriptions: the OTHER-prompt styles, stored as an artifact ──────
# The eval shows a persona in one of OTHER_DESCRIPTION_STYLES:
#   system_prompt_style       — the persona's system prompt verbatim (2nd person)
#   third_person_description  — a heuristic 3rd-person rewrite
#   redacted                  — secret-removed (cases 9 & 11); heuristic fallback
# Both the 3rd-person and (especially) the redacted variant are heuristic, so all
# variants per persona are stored in ONE persona_descriptions.json so the wording
# is auditable and can be hand-curated. The eval writes the variants it used into
# the output dir (provenance) and can LOAD an override file (descriptions_filepath).

def build_persona_descriptions(personas: dict[str, PersonaConfig]) -> dict[str, dict[str, str]]:
    """Every description variant for every persona, computed from its system
    prompt. The third_person_description and redacted variants are heuristic — a
    starting point to hand-edit in the stored file (the redacted one especially,
    since reliably stripping a hidden goal needs human judgment)."""
    return {
        name: {style: describe_other(p, style) for style in OTHER_DESCRIPTION_STYLES}
        for name, p in personas.items()
    }


def write_persona_descriptions(path: Path, descriptions: dict[str, dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(descriptions, indent=2, ensure_ascii=False))
    return path


def resolve_persona_descriptions(opts: "EvalOptions",
                                 personas: dict[str, PersonaConfig]) -> dict[str, dict[str, str]]:
    """The {persona: {style: text}} map the eval will use: computed from the
    persona prompts, then OVERRIDDEN per (persona, style) by descriptions_filepath
    if it is set (so a curated 3rd-person description wins, while anything the file
    omits still falls back to the computed text)."""
    descriptions = build_persona_descriptions(personas)
    if opts.descriptions_filepath:
        path = Path(opts.descriptions_filepath)
        if path.exists():
            override = json.loads(path.read_text())
            for name, styles in override.items():
                descriptions.setdefault(name, {}).update(styles or {})
        else:
            logger.warning(f"descriptions_filepath {path} not found; computing descriptions")
    return descriptions


# ── Collection-directory layout (accumulate separate case/condition runs) ────
# With opts.eval_dir set, every run writes into ONE stable directory, one result
# file per (case ∪ condition) SLICE, so you can run a single case/condition per
# invocation and gather them in the same place. Without it, the legacy per-run
# timestamped directory (single trials.jsonl) is used.

@dataclass
class BinaryLayout:
    deliverable: Path     # the JSONL rows for this run land here (append + resume)
    work: Path            # per-slice working dir (trial manifest, shard parts)
    trial_manifest: Path  # pre-run, recoverable trial list
    eval_manifest: Path   # per-run/per-slice manifest JSON
    shards_dir: Path      # multi-GPU per-shard outputs
    slice_tag: str | None  # None in legacy mode
    collection: bool


def binary_slice_tag(opts: "EvalOptions") -> str:
    """Filesystem-safe tag for a run's set of cases, used to name the result file
    in a collection dir so separate runs don't clobber. With atomic cases the
    evaluator-SP state and description treatment are intrinsic to each case, so the
    case set is the slice (e.g. "case7" or "case3+case7"); a description_style
    override is appended when set."""
    tag = "+".join(opts.cases_to_run)
    if opts.description_style:
        tag += "__" + opts.description_style.replace("_description", "").replace("_style", "")
    return tag


def binary_layout(rd: Path, opts: "EvalOptions") -> BinaryLayout:
    """Resolve the on-disk layout for a binary run (collection vs legacy)."""
    if opts.eval_dir:
        slice_tag = binary_slice_tag(opts)
        work = rd / "_work" / slice_tag
        return BinaryLayout(
            deliverable=rd / f"{slice_tag}.jsonl",
            work=work,
            trial_manifest=work / "trial_manifest.jsonl",
            eval_manifest=rd / "manifests" / f"{slice_tag}.eval_manifest.json",
            shards_dir=work / "_shards",
            slice_tag=slice_tag,
            collection=True,
        )
    return BinaryLayout(
        deliverable=rd / "trials.jsonl",
        work=rd,
        trial_manifest=rd / "trial_manifest.jsonl",
        eval_manifest=rd / "eval_manifest.json",
        shards_dir=rd / "_shards",
        slice_tag=None,
        collection=False,
    )


def append_shard_outputs(shards_dir: Path, deliverable: Path, num_shards: int) -> None:
    """Append each shard's rows to the deliverable (append preserves any rows from
    a prior resumed run; workers already skipped done trial_ids, so no dupes)."""
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    with open(deliverable, "a") as out:
        for i in range(num_shards):
            sp = shards_dir / f"eval_{i}.jsonl"
            if sp.exists():
                out.write(sp.read_text())


# ── Trial manifest (pre-run, recoverable) + resume ───────────────────────────

def write_trial_manifest(path: Path, trials: list[BinaryTrial]) -> Path:
    """Persist the exact trial list to `path` BEFORE evaluation, so the run is
    recoverable if it crashes. One JSON object (asdict) per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in trials:
            f.write(json.dumps(asdict(t)) + "\n")
    return path


def read_trial_manifest(path: Path) -> list[BinaryTrial]:
    """Reconstruct the trial list a launcher wrote (workers read their slice)."""
    out: list[BinaryTrial] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(BinaryTrial(**json.loads(line)))
    return out


def load_done_trial_ids(path: Path) -> set[str]:
    """trial_ids already present in an output JSONL (for resume). Missing → empty."""
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tid = json.loads(line).get("trial_id")
            except Exception:
                continue
            if tid:
                done.add(tid)
    return done


# ── Manifests + Parquet ──────────────────────────────────────────────────────

def binary_trial_counts(trials: list[BinaryTrial]) -> dict:
    """Trial counts keyed by "<case>|sp=<0/1>|desc=<style>" for the manifest."""
    c = Counter(
        f"{t.case_id}|sp={int(bool(t.eval_system_prompt_enabled))}|desc={t.other_description_style}"
        for t in trials
    )
    return dict(sorted(c.items()))


def write_binary_manifest(manifest_path: Path, run_config: RunConfig, opts: EvalOptions,
                          trials: list[BinaryTrial], run_id: str, gen_run_id: str,
                          slice_tag: str | None = None,
                          deliverable: Path | None = None) -> Path:
    """Eval manifest JSON: config snapshot, model, input generation file, counts
    by case-condition, sampling seed, timestamp. `slice_tag`/`deliverable` record
    where this run's rows landed (collection mode)."""
    counts = binary_trial_counts(trials)
    manifest = {
        "experiment": "self_recognition_binary_eval",
        "run_id": run_id,
        "generation_run_id": gen_run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": run_config.model_name,
        "model_slug": model_slug(run_config.model_name),
        "adapter": run_config.adapter,
        "task": run_config.task,
        "generations_filepath": str(_generations_root(run_config)),
        "eval_dir": opts.eval_dir,
        "slice_tag": slice_tag,
        "results_file": (str(deliverable) if deliverable else None),
        "config": {
            "cases_to_run": list(opts.cases_to_run),
            "prompt_wording_version": opts.prompt_wording_version,
            "description_style": opts.description_style,
            "calibration_base_case": opts.calibration_base_case,
            "calibration_personas": (list(opts.calibration_personas)
                                     if opts.calibration_personas else None),
            "max_trials_per_case": opts.max_trials_per_case,
            "sampling_seed": opts.sampling_seed,
            "save_logprobs": opts.save_logprobs,
            "binary_free_generate": opts.binary_free_generate,
            "descriptions_filepath": opts.descriptions_filepath,
            "groups": list(opts.groups),
            "sample_size": run_config.sample_size,
            "personas": list(run_config.personas),
            "batch_size": run_config.batch_size,
        },
        "n_trials_total": len(trials),
        "n_trials_by_case_condition": counts,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def maybe_write_parquet(trials_jsonl: Path, opts: EvalOptions) -> Path | None:
    """Write a Parquet copy of trials.jsonl when save_parquet is set. Best-effort:
    logs and returns None if pandas/pyarrow can't write it."""
    if not opts.save_parquet:
        return None
    try:
        from core.results_logger import load_results
        df = load_results(trials_jsonl)
        out = trials_jsonl.with_suffix(".parquet")
        df.to_parquet(out)
        return out
    except Exception as e:  # missing pyarrow, etc.
        logger.warning(f"could not write Parquet ({e}); JSONL is authoritative")
        return None


# ── Dry-run preview + validations ────────────────────────────────────────────

def preview_binary_trials(trials: list[BinaryTrial], text_map: dict,
                          personas: dict[str, PersonaConfig], specs: dict,
                          n: int = 2) -> None:
    """Print a short preview of assembled prompts per case + balance validations.
    `specs` is {case_id: CaseSpec} for the run. Used by dry_run; no model loaded."""
    by_case: dict[str, list[BinaryTrial]] = {}
    for t in trials:
        by_case.setdefault(t.case_id, []).append(t)

    print("\n" + "=" * 72)
    print(f"BINARY EVAL DRY RUN — {len(trials)} trial(s) across {len(by_case)} case(s)")
    print("=" * 72)
    for case_id in sorted(by_case):
        cts = by_case[case_id]
        n_correct_A = sum(t.correct_answer == "A" for t in cts)
        print(f"\n### {case_id} ({cts[0].case_type}) — {len(cts)} trials | "
              f"correct A/B = {n_correct_A}/{len(cts) - n_correct_A}")
        if cts[0].text_order is not None:
            order_counts = Counter(t.text_order for t in cts)
            print(f"    text_order balance: {dict(order_counts)}")
        for t in cts[:n]:
            texts = trial_text_fields(t, text_map)
            prompt = build_trial_prompt(t, texts, personas, spec=specs[t.case_id])
            print(f"\n  --- trial_id={t.trial_id}")
            print(f"      sp_enabled={t.eval_system_prompt_enabled} "
                  f"desc_style={t.other_description_style} correct={t.correct_answer}")
            preview = prompt if len(prompt) < 700 else prompt[:700] + " …[truncated]"
            print("      " + preview.replace("\n", "\n      "))
    print("\n" + "=" * 72 + "\n")


def validate_binary_trials(trials: list[BinaryTrial]) -> list[str]:
    """Lightweight invariants for dry-run: unique trial ids, A/B answers only,
    A/B balance per case, text-position balance for two-text cases. Returns a
    list of human-readable problems (empty = all good).

    Balance is checked with a COUNT-AWARE band: the enumeration is exactly 50/50
    by construction, so a deviation only flags a logic bug — but sampling a case
    down to a small cap adds binomial noise (~1/√n), so the band widens for small
    samples to avoid false alarms on tiny dev runs."""
    problems: list[str] = []
    ids = [t.trial_id for t in trials]
    if len(set(ids)) != len(ids):
        problems.append(f"duplicate trial_id(s): {len(ids) - len(set(ids))}")
    if any(t.correct_answer not in ("A", "B") for t in trials):
        problems.append("correct_answer not in {A, B} for some trial")
    # Balance is a property of the logical (base) trials; the condition expansion
    # replicates each one deterministically, so dedupe to base trials first (else
    # the effective sample size is overcounted and the band is too tight).
    by_case: dict[str, dict[str, BinaryTrial]] = {}
    for t in trials:
        by_case.setdefault(t.case_id, {}).setdefault(t.base_trial_id, t)
    for case_id, base in by_case.items():
        cts = list(base.values())
        n = len(cts)
        # ~3 binomial sigma around 0.5, floored to a sane minimum band.
        tol = max(0.15, 3 * (0.25 / n) ** 0.5) if n else 1.0
        frac = sum(t.correct_answer == "A" for t in cts) / n if n else 0
        if abs(frac - 0.5) > tol:
            problems.append(f"{case_id}: correct-answer A fraction {frac:.2f} not ~0.5 "
                            f"(±{tol:.2f}, n_base={n})")
        if any(t.text_order for t in cts):  # two-text cases counterbalance text order
            orders = Counter(t.text_order for t in cts)
            if len(orders) == 2:
                lo, hi = sorted(orders.values())
                if hi and lo / hi < 0.5:
                    problems.append(f"{case_id}: text_order imbalance {dict(orders)}")
    return problems
