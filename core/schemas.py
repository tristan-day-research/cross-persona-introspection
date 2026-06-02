"""Typed schemas for the PSM introspection framework."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersonaConfig:
    """Configuration for a persona."""
    name: str
    system_prompt: str
    description: str = ""
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
    # Inference batch size (activation_probing only; >1 requires padding support)
    batch_size: int = 1
    # Optional OpenRouter judge
    openrouter_model: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    # Optional HuggingFace LoRA adapter id/path. When set, the model is treated
    # as a fine-tuned variant: the adapter's short name labels the run directory
    # and figures (e.g. Llama_8b_<adapter>). Loading the adapter itself is not
    # yet wired into the backend — this field only drives naming/labelling.
    adapter: Optional[str] = None


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
