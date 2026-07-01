"""Typed schemas for the PSM introspection framework."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersonaConfig:
    """Configuration for a persona."""
    name: str
    system_prompt: str
    description: str = ""
    # Coarse grouping label (e.g. "Suppression - lexical", "Near-twin (pair A)").
    # Used by analysis to bucket personas; carried through from the persona file.
    category: str = ""
    # TODO: Add vector-steering params when implemented
    # TODO: Add fine-tuned adapter path when implemented


@dataclass
class TaskItem:
    """A single task/question to present to the model."""
    task_id: str
    prompt: str
    task_set: str  # e.g. "factual", "opinion", "borderline"
    format: str = "constrained"  # "constrained" or "free_form"
    choices: list[str] = field(default_factory=list)  # for multiple choice
    expected_answer: Optional[str] = None  # ground truth if known
    metadata: dict = field(default_factory=dict)


@dataclass
class RunConfig:
    """Configuration for an experiment run."""
    experiment_name: str
    model_name: str
    personas: list[str]  # persona names to use
    # Task sets to load. Each entry is either a set name (str) for the whole set,
    # or a single-key {set_name: [task_id, ...]} mapping to take specific
    # questions. See core.task_loader.load_tasks.
    task_sets: list
    sample_size: Optional[int] = None  # None = use all tasks
    seed: int = 42
    max_new_tokens: int = 256
    temperature: float = 0.0  # greedy by default for reproducibility
    output_dir: str = "results/raw"
    # Source-reporter matrix specific
    pause_cue: str = "\n[PAUSE: Before answering, note your current inclination.]\n"
    # Base model few-shot mode: "fixed" (same examples every question) or
    # "random" (sample from pool, different each question). Only used by
    # confidence_entropy_base experiment.
    few_shot_mode: str = "fixed"
    # If True, replace the generic "Answer: " terminal cue with a
    # persona-specific suffix like "Answer (as a chemist): ". Only used by
    # confidence_entropy_base experiment.
    use_persona_suffixes: bool = False
    # Confidence legend style for base model prompts:
    #   "bins"    — "S: <5%, T: 5-10%, ... Z: >90%" (percentage anchors)
    #   "ordinal" — "S, T, U, V, W, X, Y, Z (S = lowest, Z = highest)"
    # Only used by confidence_entropy_base experiment.
    confidence_legend: str = "bins"
    # Activation probing: direct task file path (relative to tasks/ dir)
    task_file: Optional[str] = None
    # Patchscope: path to patchscope-specific config (relative to config/ dir)
    patchscope_config: Optional[str] = None
    # Model dtype: "bfloat16", "float16", "float32", or None (auto: float16 on CUDA)
    model_dtype: Optional[str] = None
    # Inference batch size. >1 batches recognition/generation calls through the
    # backend's *_batch methods (padded forward passes). 1 = unbatched (legacy).
    batch_size: int = 1
    # Number of GPUs for data-parallel sharding of a self-recognition run.
    #   None  → auto-detect (use all visible CUDA devices)
    #   1     → single-process (no sharding)
    #   N>1   → split the workload across N worker processes, one per GPU
    num_gpus: Optional[int] = None
    # Optional OpenRouter judge
    openrouter_model: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    # Optional HuggingFace LoRA adapter id/path. When set, the model is treated
    # as a fine-tuned variant: the adapter's short name labels the run directory
    # and figures (e.g. Llama_8b_<adapter>), and the adapter is loaded onto the
    # base model via PEFT in the HF backend.
    adapter: Optional[str] = None
    # self_recognition text-generation/eval: the generation task name. Names the
    # top-level output folder under results/text_generations/<task>/ and
    # results/text_evaluations/<task>/. Default "articles" (the paper-replication
    # XSUM/CNN summaries).
    task: str = "articles"
    # self_recognition eval only: explicit directory containing the per-persona
    # generations to evaluate ({generations_filepath}/{persona}/{dataset}/summaries.json).
    # When None, the eval derives it from task + model_slug. (Named *_filepath to
    # stay distinct from the `task` label, which is reused elsewhere.)
    generations_filepath: Optional[str] = None
    # self_recognition text generation (generate_text.py):
    #   conceal_persona — append a "don't reveal your role/identity" instruction to
    #     the user turn so the persona writes in its own voice WITHOUT explicit
    #     giveaways. Important for the binary recognition eval, which tests
    #     style-level self-recognition (not literal "As a chemist," matching). The
    #     persona system prompt itself is left untouched. Default off (paper-faithful).
    #   strip_self_refs — post-hoc safety net: strip a leading "As a <role>," clause
    #     from the stored text (raw output is always kept separately). Applies to
    #     BOTH article and prompt modes. Default on.
    conceal_persona: bool = False
    strip_self_refs: bool = True
    # A single stable name that ties all phases of one experiment together.
    # When set, it flows as:
    #   generation → output_subdir (folder name under text_generations/.../):
    #                  results/text_generations/<task>/<model_slug>/<run_name>/
    #   eval       → eval_dir (stable collection folder under text_evaluations/):
    #                  results/text_evaluations/<task>/<run_name>/
    #   R2         → prefix for both activation stores:
    #                  runs/<run_name>/activations/       (generation phase)
    #                  runs/<run_name>/eval_activations/  (eval phase)
    # This means `download_run_activations(run_id=run_name, ...)` fetches both.
    # output_subdir / eval_dir take precedence if set explicitly; run_name is the
    # fallback so you can set one stable name and override individual phases if needed.
    run_name: Optional[str] = None
    output_subdir: Optional[str] = None  # overrides run_name for generation folder
    # Activation capture during GENERATION (core.activation_capture/_store). One
    # extra forward per generated text captures the spec's generation-phase named
    # positions into a sharded store keyed by text_id ("{group}/{persona}/{task_id}").
    collect_activations: bool = False
    activations_dir: Optional[str] = None
    activation_layers: Optional[list] = None
    activation_shard_size: int = 1000
    # When True, delete the local activation files after a SUCCESSFUL R2 upload, so
    # an ephemeral pod doesn't accumulate them and pulling results stays light. Only
    # ever deletes what was confirmed uploaded (the sync raises on failure first);
    # a no-op when R2 isn't configured. Resume across pods then relies on R2, not
    # local disk — fine when each run does a NEW case set (disjoint trial_ids).
    delete_local_activations: bool = False


@dataclass
class ParsedResponse:
    """Parsed output from a model response."""
    raw_text: str
    answer: Optional[str] = None  # e.g. "A", "B", "C", "D"
    confidence: Optional[int] = None  # 1-5
    refuse: Optional[bool] = None
    parse_success: bool = False


@dataclass
class SourceStateMetrics:
    """Metrics measured from the source persona's internal state at the pause point."""
    top1_answer: Optional[str] = None
    top1_prob: Optional[float] = None
    entropy: Optional[float] = None
    logit_gap: Optional[float] = None  # gap between top-1 and top-2 logits
    confidence_proxy_bin: Optional[int] = None  # 1-5 derived from top1_prob or logit_gap
    answer_probs: Optional[dict[str, float]] = None  # prob for each choice option


@dataclass
class TrialRecord:
    """One trial of an experiment. Written as one JSONL line."""
    experiment: str
    model: str
    task_id: str
    task_set: str
    # Persona info
    source_persona: Optional[str] = None
    reporter_persona: Optional[str] = None
    # For cross-persona prediction
    predicted_answer: Optional[str] = None
    predicted_confidence: Optional[int] = None
    predicted_refuse: Optional[bool] = None
    actual_answer: Optional[str] = None
    actual_confidence: Optional[int] = None
    actual_refuse: Optional[bool] = None
    # For source-reporter matrix
    source_metrics: Optional[dict] = None  # serialized SourceStateMetrics
    reporter_answer: Optional[str] = None
    reporter_confidence: Optional[int] = None
    reporter_refuse: Optional[bool] = None
    used_kv_cache: bool = False
    # Scores
    choice_match: Optional[bool] = None
    confidence_error: Optional[int] = None
    refuse_match: Optional[bool] = None
    # Meta
    raw_response: str = ""
    error: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class ConfidenceEntropyRecord:
    """One trial of the confidence-vs-entropy experiment. Written as one JSONL line.

    Each trial consists of 4 independent prompts to the model:
    1. Open-ended: free-form reasoning about the MCQ
    2. Forced-choice: constrained to output just a letter (A/B/C/D)
       → this is where we measure option_probs, entropy, etc.
    3. Confidence open-ended: model explains its confidence level in free text
    4. Stated confidence: model rates its confidence on a categorical scale (S-Z)
       → this is where we measure confidence_option_probs
    """
    experiment: str
    model: str
    persona_name: str
    question_id: str
    domain: str
    source_dataset: str
    correct_answer: Optional[str] = None
    # Prompt 1: open-ended reasoning
    open_ended_response: str = ""
    # Prompt 2: forced-choice answer (letter only)
    forced_choice_answer: Optional[str] = None  # normalized to uppercase A-E
    forced_answer_validity: Optional[bool] = None  # True if answer parsed cleanly
    is_correct: Optional[bool] = None
    forced_choice_raw: str = ""
    # Logprob-based metrics from forced-choice prompt (over answer options only)
    option_probs: Optional[dict[str, float]] = None  # softmax probs (sum to 1 over choices)
    option_logits: Optional[dict[str, float]] = None  # raw pre-softmax logits
    answer_option_entropy: Optional[float] = None
    chosen_answer_probability: Optional[float] = None
    margin_between_top_two: Optional[float] = None
    answer_ranking: Optional[list[str]] = None  # choices sorted by prob descending
    # Prompt 3: confidence open-ended reasoning
    confidence_open_response: str = ""
    # Prompt 4: stated confidence
    stated_confidence_letter: Optional[str] = None  # S-Z
    stated_confidence_midpoint: Optional[float] = None  # 0.025-0.95
    stated_confidence_raw: str = ""
    confidence_answer_validity: Optional[bool] = None  # True if confidence letter parsed cleanly
    # Stated confidence logprob-based metrics
    confidence_option_probs: Optional[dict[str, float]] = None  # softmax probs
    confidence_option_logits: Optional[dict[str, float]] = None  # raw pre-softmax logits
    # Reproducibility
    system_prompt: str = ""
    temperature: float = 0.0
    # Exact prompts sent to the model (for auditing/debugging)
    mc_prompt_text: Optional[str] = None  # full MC prompt (base model: raw text, instruct: None)
    confidence_prompt_text: Optional[str] = None  # full confidence prompt
    # Meta
    error: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class SelfRecognitionRecord:
    """One trial of the persona self-recognition experiment.

    A single record covers a generation row or one of several evaluation
    "modes", discriminated by `phase`. Fields not relevant to the row's phase
    are left as None. The recognition modes and their characteristic fields:

      phase="generation"
        Source text production. Fields: source_persona, target_persona,
        source_pov, generated_text(_raw), token_length, generated_text_id.

      phase="individual"  (a.k.a. individual_yes_no)
        Evaluator shown one text, asked YES/NO "did you write this?".
        Choice comes from constrained first-token probs over {YES, NO}.
        Fields: evaluator_persona, candidate_a_source/target/text,
        parsed_choice ("YES"/"NO"), model_answer (= parsed_choice),
        choice_probs, prob_yes, prob_no, is_self, is_correct,
        and (optionally) logprob_yes / logprob_no when the backend exposes
        token logprobs (None otherwise).

      phase="individual_probability"
        Evaluator shown one text, asked for an integer 0-100 estimate of the
        probability that it wrote the text. Fields: evaluator_persona,
        candidate_a_source/target/text, parsed_probability (0-100 or None),
        is_self (ground-truth self/nonself), raw_response, generated_text_id.

      phase="paired"  (a.k.a. paired_ab / paired_ab_counterbalanced)
        Evaluator shown candidate A and candidate B, picks which it wrote.
        Choice from constrained first-token probs over {A, B}. A/B order is
        counterbalanced by default (each unordered pair run as "ab" and "ba");
        `pair_order` records the order. Fields: candidate_a/b_source/target/text,
        pair_order, parsed_choice ("A"/"B"), model_answer (= parsed_choice),
        choice_probs, prob_a, prob_b. When the evaluator authored exactly one
        candidate (has_ground_truth): correct_answer ("A"/"B"),
        position_of_self ("A"/"B"), chose_self (bool), is_correct.

      phase="paired" with confidence enabled  (a.k.a. paired_ab_confidence)
        Same as paired, but the generation uses a "CHOICE: A/B\nCONFIDENCE:
        0-100" format. Here parsed_choice / model_answer come from the CHOICE
        line (the answer the stated confidence refers to), parsed_confidence
        (0-100 or None) is populated, and the constrained prob_a / prob_b are
        left None (the answer letter is not the first generated token).

    Validation invariants (enforced in __post_init__ when the field is set):
      parsed_probability / parsed_confidence ∈ {None} ∪ [0, 100];
      parsed_choice / model_answer ∈ {None, "A", "B", "YES", "NO"};
      correct_answer / position_of_self ∈ {None, "A", "B"};
      chose_self ∈ {None, True, False}.
    """
    experiment: str
    phase: str  # "generation" | "individual" | "paired"
    model: str
    task_id: str
    run_id: str
    # Which task set this question came from, e.g. "neutral" or "misaligned".
    task_set: Optional[str] = None
    # Generation phase
    source_persona: Optional[str] = None
    # Persona the model was asked to *write as* (3rd-person POV). For 1st-person
    # rows, equals source_persona. For 3rd-person rows where the chemist is
    # asked to "write as a 5 year old", source_persona="chemist" and
    # target_persona="five_year_old".
    target_persona: Optional[str] = None
    # "1st_person" or "3rd_person" — POV under which the source text was produced.
    source_pov: Optional[str] = None
    generated_text: Optional[str] = None
    generated_text_raw: Optional[str] = None  # before preprocessing
    token_length: Optional[int] = None
    # Evaluation phase (shared)
    evaluator_persona: Optional[str] = None
    candidate_a_source: Optional[str] = None
    candidate_a_target: Optional[str] = None  # 3rd-person target for candidate A
    candidate_a_text: Optional[str] = None
    candidate_b_source: Optional[str] = None  # paired only
    candidate_b_target: Optional[str] = None  # paired only, 3rd-person target for B
    candidate_b_text: Optional[str] = None    # paired only
    pair_order: Optional[str] = None          # "ab" or "ba" (paired)
    # Model output
    parsed_choice: Optional[str] = None       # "YES"/"NO" (individual) or "A"/"B" (paired)
    choice_probs: Optional[dict[str, float]] = None
    # Normalized scalar probabilities (softmax over the two choices), duplicated
    # from choice_probs for easy column access in analysis. Only the pair
    # relevant to the phase is populated (yes/no for individual, a/b for paired).
    prob_yes: Optional[float] = None
    prob_no: Optional[float] = None
    prob_a: Optional[float] = None
    prob_b: Optional[float] = None
    raw_response: Optional[str] = None
    # Explicit parsed answer for the row's mode. For individual / paired this
    # mirrors parsed_choice; kept as a separate, explicitly-named field so
    # downstream analysis has a stable column independent of how the choice was
    # decoded. ∈ {None, "A", "B", "YES", "NO"}.
    model_answer: Optional[str] = None
    # individual_probability: integer 0-100 estimate of P(self wrote text), or
    # None if the response could not be parsed.
    parsed_probability: Optional[int] = None
    # paired (confidence mode): integer 0-100 stated confidence in the choice,
    # or None if absent / unparseable.
    parsed_confidence: Optional[int] = None
    # Ground-truth self/nonself for single-candidate modes (individual,
    # individual_probability): True iff the evaluator persona authored the text
    # (evaluator == inducing persona). None when undefined.
    is_self: Optional[bool] = None
    # paired ground-truth helpers (set only when the evaluator authored exactly
    # one of the two candidates). correct_answer/position_of_self are the letter
    # holding the evaluator's own text; chose_self is whether it was picked.
    correct_answer: Optional[str] = None
    position_of_self: Optional[str] = None
    chose_self: Optional[bool] = None
    # Optional constrained-choice logprobs (full-vocab log-softmax at the chosen
    # first token). Populated for individual YES/NO when the backend exposes
    # them and the run enables yes_no_logprobs; None otherwise.
    logprob_yes: Optional[float] = None
    logprob_no: Optional[float] = None
    # Stable identifier for the single candidate text in individual modes:
    # "{task_id}|{source_persona}|{target_persona}". None for paired/generation.
    generated_text_id: Optional[str] = None
    # Two notions of "correct":
    #   author_is_correct — evaluator recognizes text generated under its own
    #     system prompt (authorship by induced persona).
    #   style_is_correct  — evaluator recognizes text written in its style
    #     (the 3rd-person target persona). In 1st-person these coincide.
    author_is_correct: Optional[bool] = None
    style_is_correct: Optional[bool] = None
    has_author_ground_truth: bool = False
    has_style_ground_truth: bool = False
    # Back-compat alias for author_is_correct / has_author_ground_truth.
    is_correct: Optional[bool] = None
    has_ground_truth: bool = False
    # Meta
    prompt_text: Optional[str] = None         # original task prompt
    timestamp: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the recognition-mode invariants before the row is logged.

        Permissive by design: each check only fires when its field is set, so
        generation rows, error rows, and legacy rows (which leave the new
        fields at None) all pass untouched. Raises ValueError on a genuine
        contract violation, which would indicate a parsing/logging bug.
        """
        def _in_range(name: str, val) -> None:
            if val is not None and not (0 <= val <= 100):
                raise ValueError(f"{name} must be None or within [0, 100], got {val!r}")

        _in_range("parsed_probability", self.parsed_probability)
        _in_range("parsed_confidence", self.parsed_confidence)

        for name, allowed in (
            ("parsed_choice", {"A", "B", "YES", "NO"}),
            ("model_answer", {"A", "B", "YES", "NO"}),
            ("correct_answer", {"A", "B"}),
            ("position_of_self", {"A", "B"}),
        ):
            val = getattr(self, name)
            if val is not None and val not in allowed:
                raise ValueError(f"{name} must be None or one of {sorted(allowed)}, got {val!r}")

        if self.chose_self is not None and not isinstance(self.chose_self, bool):
            raise ValueError(f"chose_self must be None or bool, got {self.chose_self!r}")


@dataclass
class PaperReplicationRecord:
    """One trial of the Panickssery et al. (2024) self-recognition replication.

    Faithful re-implementation of "LLM Evaluators Recognize and Favor Their Own
    Generations" (arXiv:2404.13076). One underlying model acts as the evaluator;
    candidate summaries come from several distinct *sources* (human / gpt4 / gpt35
    / llama / claude). A record is discriminated by `phase`:

      phase="detection"   (pairwise self-recognition, Table 5 / Fig 2-left)
        Evaluator shown its own summary vs `other_source`'s summary, asked which
        it wrote. Run in both orders; `score` is the order-averaged probability
        the evaluator assigned to its OWN summary — 0.5·(P_fwd(self) + P_bwd(self)),
        the paper's ordering-bias-corrected self-recognition score.

      phase="comparison"  (pairwise self-preference, Table 5 / Fig 4-left)
        Same structure with the "which do you prefer?" prompt; `score` is the
        order-averaged probability of preferring its OWN summary (self-preference).

      phase="recognition" (individual yes/no self-recognition, Table 10 / Fig 2-right)
        Evaluator shown a single summary from `target_source`, asked yes/no
        "did you write this?". `score` = renormalized P("Yes"); `ground_truth`
        = int(target_source == evaluator self).

      phase="scoring"     (individual Likert self-preference, Table 10 / Fig 4-right)
        Evaluator rates a single `target_source` summary 1-5; `score` is the
        probability-weighted mean rating (∑ i·P(i)). `ground_truth` as above.

    Pairwise rows populate `other_source` + `probs_forward`/`probs_backward`;
    individual rows populate `target_source` + `probs`. `score` is the headline
    scalar for the row's phase. Per the paper, summaries are sampled at temp 0
    and the evaluator's confidence comes from constrained first-token probs.
    """
    experiment: str
    phase: str  # "detection" | "comparison" | "recognition" | "scoring"
    model: str          # evaluator model_name (e.g. meta-llama/Llama-2-7b-chat-hf)
    dataset: str        # "xsum" | "cnn"
    run_id: str
    key: str            # article id (shared across all sources)
    evaluator: str      # the evaluator's own source label (its "self"), e.g. "llama"
    # Pairwise (detection / comparison): the alternative source compared against.
    other_source: Optional[str] = None
    probs_forward: Optional[dict[str, float]] = None   # {"1":p,"2":p}; self is option 1
    probs_backward: Optional[dict[str, float]] = None  # {"1":p,"2":p}; self is option 2
    # Individual (recognition / scoring): the single source whose text was shown.
    target_source: Optional[str] = None
    probs: Optional[dict[str, float]] = None  # {"Yes","No"} or {"1".."5"}
    # Headline scalar for the row's phase (see class docstring).
    score: Optional[float] = None
    # Individual phases: 1 iff target_source is the evaluator's own text.
    ground_truth: Optional[int] = None
    raw_response: Optional[str] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PersonaRecognitionEvalRecord:
    """One trial of the paper-faithful persona self-recognition evaluation
    (experiments/self_recognition/evaluate_self_recognition.py).

    Same four measurements as PaperReplicationRecord (Panickssery et al. 2024),
    but in the PERSONA vocabulary used across the self_recognition experiment:
    `evaluator_persona` is the inducing/evaluator persona ("self"); `source_persona`
    is the non-evaluator persona in question — the alternative author for the
    pairwise phases, or the author of the single shown summary for the individual
    phases. One vocabulary so this and the existing persona analyses share names.

      phase="pairwise_detection"   pairwise self-recognition  (probs_forward/backward, 1/2)
      phase="pairwise_comparison"  pairwise self-preference   (probs_forward/backward, 1/2)
      phase="recognition"          individual yes/no          (probs, score=P(Yes))
      phase="scoring"              individual Likert 1-5       (probs, score=prob-weighted mean)

    Pairwise score is the ordering-bias-corrected 0.5*(P_fwd(self)+P_bwd(self)).
    `ground_truth` (individual) = int(source_persona == evaluator_persona).
    """
    experiment: str
    phase: str  # "pairwise_detection" | "pairwise_comparison" | "recognition" | "scoring"
    model: str
    dataset: str        # "xsum" | "cnn"
    run_id: str
    key: str            # article id (shared across all sources)
    evaluator_persona: str            # the inducing/evaluator persona ("self")
    source_persona: Optional[str] = None  # the other persona (pairwise alt / individual author)
    probs_forward: Optional[dict[str, float]] = None
    probs_backward: Optional[dict[str, float]] = None
    probs: Optional[dict[str, float]] = None
    score: Optional[float] = None
    ground_truth: Optional[int] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PersonaDescriptionEvalRecord:
    """One trial of the multi-class persona-source classification eval
    (the `all_persona_descriptions` measurement in
    experiments/self_recognition/evaluate_self_recognition.py).

    Tests whether an *active* evaluator persona classifies its own generations
    better than other evaluator personas classify that same source persona's
    generations. The evaluator is shown one generated text plus a shuffled set
    of candidate personas labelled A/B/C/D… and asked to assign probabilities
    (summing to 100) over which candidate produced it.

    Two conditions, discriminated by `condition`:
      "active_persona"     — the evaluator persona's system prompt is active.
                             `evaluator_persona` is set; `is_self` =
                             (evaluator_persona == source_persona).
      "description_only"   — no evaluator persona is active (neutral baseline).
                             `evaluator_persona` is None; `is_self` is None.

    Candidate display (`candidate_display_mode`):
      "full_prompt_anonymous_labels" — A/B/C/D labels, each shown with the full
                             candidate persona system prompt (the main condition).
      "short_label_names"  — A/B/C/D labels, each shown with the candidate's
                             short persona name (semantic-hint ablation).

    `candidate_mapping` is the hidden {label: persona_name} assignment for this
    trial (order randomized per trial); `correct_label` is the label mapped to
    `source_persona`. `probabilities` is the normalized distribution over labels
    (sums to 1.0); `raw_probabilities` is what the model emitted before
    normalization. `parse_status` ∈ {"ok", "normalized", "failed"}.
    """
    experiment: str
    phase: str          # always "all_persona_descriptions"
    model: str
    dataset: str        # the "group": "xsum"/"cnn" (article mode) or the task-set
                        # name (prompt mode). `task_set` mirrors it under a clearer
                        # name; both are kept so older readers still find `dataset`.
    run_id: str
    key: str            # article id / task_id (the text's source task)
    condition: str      # "active_persona" | "description_only"
    source_persona: str  # who actually generated the text
    candidate_display_mode: str
    candidate_mapping: dict          # {label: persona_name}
    correct_label: str               # label mapped to source_persona
    # The generation group the text came from — the task-set name (prompt mode)
    # or the dataset (article mode). Same value as `dataset`, under a clearer name
    # so analysis can break results down by task set. Defaults to `dataset` when
    # not passed explicitly (see __post_init__).
    task_set: Optional[str] = None
    evaluator_persona: Optional[str] = None  # None for description_only
    # Parsed model output
    probabilities: Optional[dict] = None       # normalized {label: prob}, sums to 1
    raw_probabilities: Optional[dict] = None    # {label: value} as emitted
    predicted_label: Optional[str] = None       # argmax of probabilities
    correct_probability: Optional[float] = None  # normalized prob on correct_label
    confidence: Optional[float] = None           # model's stated confidence (0-100)
    brief_reason: Optional[str] = None
    is_correct: Optional[bool] = None            # predicted_label == correct_label
    is_self: Optional[bool] = None               # evaluator_persona == source_persona
    parse_status: str = "failed"                 # "ok" | "normalized" | "failed"
    raw_response: Optional[str] = None
    text: Optional[str] = None                   # the generated text shown
    timestamp: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        # `task_set` mirrors the group; default it from `dataset` so every row
        # carries the clearer name even when the caller only set `dataset`.
        if self.task_set is None:
            self.task_set = self.dataset


@dataclass
class PatchscopeRecord:
    """One trial of the patchscope activation interpretation experiment.

    References:
      Patchscopes: Ghandeharioun et al., arXiv:2401.06102
      SelfIE: Chen et al., arXiv:2403.10949
    """
    experiment: str
    template_name: str
    model: str
    question_id: str
    source_persona: str
    reporter_persona: str
    condition: str  # "real", "text_only_baseline", "shuffled"
    source_layer: int
    injection_layer: int
    injection_mode: str  # "replace" or "add"
    # Which variant under interpretation_templates.<template_name> was used (YAML prompt_style).
    interpretation_prompt_style: str = ""
    # Reporter system prompt (raw, before chat template)
    reporter_system_prompt: str = ""
    # Full prompt sent to model (exact text including special tokens, system prompt, MCQ, etc.)
    reporter_interpretation_prompt: str = ""
    # Decoding mode: "logits" or "generate"
    reporter_decode_mode: str = "generate"
    # Reporter results
    reporter_generated_text: str = ""
    reporter_parsed_answer: Optional[str] = None
    reporter_parse_success: bool = False
    # Reporter logit-mode results (constrained single-token decode)
    reporter_choice_probs: Optional[dict[str, float]] = None   # softmax over valid choices
    reporter_choice_logits: Optional[dict[str, float]] = None   # raw logits for valid choices
    reporter_choice_logprobs: Optional[dict[str, float]] = None  # log-softmax over full vocab at choice tokens
    reporter_total_choice_prob: Optional[float] = None           # sum of full-vocab probs on choice tokens
    reporter_predicted: Optional[str] = None                     # argmax choice
    reporter_is_correct: Optional[bool] = None                   # reporter_parsed_answer matches ground truth
    # Reporter relevancy scores (SelfIE Section 3.2)
    reporter_relevancy_scores: Optional[list[float]] = None
    reporter_mean_relevancy: Optional[float] = None
    # Ground truth (from source's direct answer under that persona)
    source_last_prefill_answer: Optional[str] = None          # from prefill logits (canonical)
    source_generated_answer: Optional[str] = None       # from actual generation during decode (canonical)
    source_answer_probs: Optional[dict[str, float]] = None
    # Source confidence metrics
    source_chosen_prob: Optional[float] = None       # P(source's top answer)
    source_margin: Optional[float] = None            # top1_prob - top2_prob
    source_answer_entropy: Optional[float] = None    # entropy of source_answer_probs
    # Reporter confidence metrics
    reporter_chosen_prob: Optional[float] = None     # P(reporter's predicted answer)
    reporter_margin: Optional[float] = None          # top1_prob - top2_prob
    # Source extraction site metadata
    source_extraction_mode: Optional[str] = None            # "prefill_<position>" or "during_generation"
    source_extraction_token_index: Optional[int] = None     # 0-based token position
    source_extraction_token_id: Optional[int] = None        # token ID at that position
    source_extraction_token_text: Optional[str] = None      # decoded token string
    source_extraction_token_offset: Optional[int] = None   # index within multi-token sweep (0 = anchor / first gen token)
    # Original question data
    question_text: Optional[str] = None
    question_options: Optional[dict[str, str]] = None
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    expected_disagreement: Optional[str] = None      # from task JSON
    neutral_reference_answer: Optional[str] = None   # ground truth from task JSON
    # Meta
    error: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class PersonaBinaryEvalRecord:
    """One binary (A/B) self-recognition trial — the 12-case design defined in
    experiments/self_recognition/evaluation_cases.py (the single source of truth
    for case structure AND wording).

    Every trial has exactly two answer options (A and B), so the chance baseline
    is always 50%. A persona of interest E is shown one or two texts (generated
    earlier by generate_text.py under various persona prompts) and must pick A or
    B according to the case's question. Each case is self-contained: it fixes its
    own evaluator-system-prompt state(s) (active E vs neutral), how the other
    persona(s) are described (none / full secret / redacted), and the exact
    wording. The 12 cases span single-text self-recognition (1–2), paired
    self-recognition with/without the other's secret (3, 4, 7, 11), the
    deactivated-self probe (6), neutral two-persona classification ceilings/floors
    (5, 8, 9, 10), and a calibration ceiling over obvious-style personas (12). See
    the CASE_REGISTRY in evaluation_cases.py for each case's full spec.

    Naming follows the repo: `run_id` is the EVAL run id (the spec's eval_run_id);
    `generation_run_id` identifies the source generation run; `group` is the
    dataset / task-set the texts came from; `task_id` is the per-group item key
    (Text1/Text2 in two-text cases share it).

    Signal: `prob_A`/`prob_B` are the 2-way softmax over {A, B} at the answer
    position; `logprob_A`/`logprob_B` are the full-vocab log-softmax there (None
    when logprobs are unavailable). `predicted_answer` is argmax over {A, B} (or
    the parsed letter in free-generation mode). `parse_status` ∈ {"ok",
    "constrained", "parsed", "failed"}.

    `base_trial_id` identifies the logical trial independent of the prompt-condition
    variant (eval_system_prompt_enabled, other_description_style); `trial_id` is
    unique per concrete instance. `sampled_from_full_case` records whether the
    case was sampled down to `max_trials_per_case`.
    """
    experiment: str
    phase: str                       # always "binary_recognition"
    case_id: str                     # "case1" … "case12"
    case_type: str                   # "test" | "control" | "baseline" | "calibration"
    model: str
    run_id: str                      # the eval run id (spec: eval_run_id)
    trial_id: str
    base_trial_id: str
    group: str                       # dataset ("xsum"/"cnn") or task-set name
    task_id: str                     # per-group item key (article id / task id)
    evaluator_persona: str           # the persona of interest E (induced only when SP on)
    eval_system_prompt_enabled: bool
    other_description_style: str     # "system_prompt_style" | "third_person_description" | "redacted" | "not_applicable"
    correct_answer: str              # "A" | "B"
    generation_run_id: Optional[str] = None
    # Persona roles (which are set depends on the case)
    source_persona: Optional[str] = None     # single-text cases (1–2): author of the text
    other_persona: Optional[str] = None       # self_other cases: the non-E persona O
    other_persona_1: Optional[str] = None      # two-described-persona cases (5/8/9): persona_1
    other_persona_2: Optional[str] = None      # two-described-persona cases (5/8/9): persona_2
    # Texts shown (single_text for the single-text cases; text1/text2 for paired cases)
    single_text: Optional[str] = None
    text1: Optional[str] = None
    text2: Optional[str] = None
    text1_source_persona: Optional[str] = None
    text2_source_persona: Optional[str] = None
    text1_generation_id: Optional[str] = None  # stable id: "{group}/{src}/{task_id}"
    text2_generation_id: Optional[str] = None
    # Answer framing (hidden from the model, recorded for analysis)
    candidate_mapping: Optional[dict] = None   # role → persona name (e.g. {"current": E, "other": O})
    answer_mapping: Optional[dict] = None       # {"A": <meaning>, "B": <meaning>}
    text_order: Optional[str] = None            # e.g. "E_first" / "O_first" / "P1_first" / "P2_first"
    answer_order: Optional[str] = None          # e.g. "A=current" / "A=text1"
    # Model output
    predicted_answer: Optional[str] = None      # "A" | "B" | None
    is_correct: Optional[bool] = None
    prob_A: Optional[float] = None              # 2-way softmax over {A, B}
    prob_B: Optional[float] = None
    prob_correct: Optional[float] = None
    logprob_A: Optional[float] = None           # full-vocab log-softmax (None if unavailable)
    logprob_B: Optional[float] = None
    # Exact prompt strings
    prompt_text: Optional[str] = None           # the user-turn prompt
    system_prompt_text: Optional[str] = None    # the evaluator system prompt (None when disabled)
    raw_response: Optional[str] = None
    parse_status: str = "failed"                # "ok" | "constrained" | "parsed" | "failed"
    # Sampling provenance
    sampled_from_full_case: Optional[bool] = None
    sampling_seed: Optional[int] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PersonaCrossModelEvalRecord:
    """One binary (A/B) CROSS-MODEL self-recognition trial.

    Generalizes the paired "which did you write?" probe so the two candidate texts
    can come from DIFFERENT models, not just different personas. The evaluator is a
    (model, persona) pair; its SELF text is the one generated by that same
    (evaluator_model, evaluator_persona). Each trial pairs that self text against
    one FOIL text (same task) whose source differs from the evaluator along one or
    both axes, captured by `foil_same_model` / `foil_same_persona`:

      diff_model_same_persona  — foil is the OTHER model, SAME persona  → isolates
                                 MODEL-identity recognition (e.g. Llama-neutral vs
                                 Qwen-neutral).
      same_model_diff_persona  — foil is the SAME model, a DIFFERENT persona →
                                 within-model persona recognition.
      diff_model_diff_persona  — foil differs on BOTH axes (easiest).

    Chance is always 0.5. Text order (self first/second) and answer letter (A=self
    or A=foil) are counterbalanced across trials, so raw accuracy is position- and
    letter-bias free. `eval_system_prompt_enabled` selects the active (persona
    induced) vs neutral (no system prompt) condition — the neutral condition is the
    baseline that isolates any active-state contribution.

    Signal fields mirror PersonaBinaryEvalRecord: `prob_A`/`prob_B` are the 2-way
    softmax over {A, B} at the answer position; `logprob_A`/`logprob_B` the
    full-vocab log-softmax there. `prob_correct` is the mass on the correct letter.
    """
    experiment: str
    phase: str                        # always "cross_model_recognition"
    model: str                        # the EVALUATOR model (kept named `model` for uniform loading)
    run_id: str
    trial_id: str
    base_trial_id: str
    group: str
    task_id: str
    foil_type: str                    # "diff_model_same_persona" | "same_model_diff_persona" | "diff_model_diff_persona"
    # Evaluator identity (the "self")
    evaluator_model: str
    evaluator_persona: str
    eval_system_prompt_enabled: bool  # True = active persona, False = neutral baseline
    condition: str                    # "active" | "neutral" (mirrors the flag, for convenient grouping)
    # Foil identity + derived same/diff axes
    foil_model: str
    foil_persona: str
    foil_same_model: bool
    foil_same_persona: bool
    # Which source sits in each text slot (resolved from text_order)
    text1_model: str
    text1_persona: str
    text2_model: str
    text2_persona: str
    text1_is_self: bool
    text1: str
    text2: str
    text1_generation_id: str          # "{model_slug}/{persona}/{group}/{task_id}"
    text2_generation_id: str
    # Answer framing (hidden from the model, recorded for analysis)
    answer_mapping: dict              # {"A": "self"/"foil", "B": ...}
    text_order: str                   # "self_first" | "foil_first"
    answer_order: str                 # "A=self" | "A=foil"
    correct_answer: str               # "A" | "B"
    generation_run_id: Optional[str] = None
    # Model output
    predicted_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    prob_A: Optional[float] = None
    prob_B: Optional[float] = None
    prob_correct: Optional[float] = None
    logprob_A: Optional[float] = None
    logprob_B: Optional[float] = None
    # Exact prompt strings
    prompt_text: Optional[str] = None
    system_prompt_text: Optional[str] = None
    raw_response: Optional[str] = None
    parse_status: str = "constrained"
    # Sampling provenance
    sampled_from_full_cell: Optional[bool] = None
    sampling_seed: Optional[int] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None
