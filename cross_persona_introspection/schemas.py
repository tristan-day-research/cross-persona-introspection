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
    task_sets: list[str]  # task set names to load
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
    # Reporter system prompt (raw, before chat template)
    reporter_system_prompt: str = ""
    # Full prompt sent to model (exact text including special tokens, system prompt, MCQ, etc.)
    interpretation_prompt: str = ""
    # Decoding mode: "logits" or "generate"
    decode_mode: str = "generate"
    # Results
    generated_text: str = ""
    reporter_parsed_answer: Optional[str] = None
    parse_success: bool = False
    # Logit-mode results (constrained single-token decode)
    choice_probs: Optional[dict[str, float]] = None   # softmax over valid choices
    choice_logits: Optional[dict[str, float]] = None   # raw logits for valid choices
    choice_logprobs: Optional[dict[str, float]] = None  # log-softmax over full vocab at choice tokens
    total_choice_prob: Optional[float] = None           # sum of full-vocab probs on choice tokens
    predicted: Optional[str] = None                     # argmax choice
    is_correct: Optional[bool] = None                   # predicted == source_direct_answer
    # Relevancy scores (SelfIE Section 3.2)
    relevancy_scores: Optional[list[float]] = None
    mean_relevancy: Optional[float] = None
    # Ground truth (from source's direct answer under that persona)
    source_direct_answer: Optional[str] = None          # from prefill logits (canonical)
    source_generated_answer: Optional[str] = None       # from actual generation during decode (canonical)
    source_answer_probs: Optional[dict[str, float]] = None
    # Source confidence metrics
    source_chosen_prob: Optional[float] = None       # P(source's top answer)
    source_margin: Optional[float] = None            # top1_prob - top2_prob
    # Reporter confidence metrics
    reporter_chosen_prob: Optional[float] = None     # P(reporter's predicted answer)
    reporter_margin: Optional[float] = None          # top1_prob - top2_prob
    # Extraction site metadata
    extraction_mode: Optional[str] = None            # "prefill_<position>" or "during_generation"
    extraction_token_index: Optional[int] = None     # 0-based token position
    extraction_token_id: Optional[int] = None        # token ID at that position
    extraction_token_text: Optional[str] = None      # decoded token string
    # Original question data
    question_text: Optional[str] = None
    question_options: Optional[dict[str, str]] = None
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    # Meta
    error: Optional[str] = None
    timestamp: Optional[str] = None
