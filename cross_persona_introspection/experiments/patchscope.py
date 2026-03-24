"""Patchscope activation interpretation experiment.

Implements activation patching as described in Patchscopes (Ghandeharioun et al.,
ICML 2024, arXiv:2401.06102): extract a hidden-state activation from a source
forward pass, inject it into a separate interpretation forward pass, and decode
what the model generates.

Also borrows the relevancy score from SelfIE (Chen et al., ICML 2024,
arXiv:2403.10949): for each generated token, the difference in probability
with vs. without the injected activation.

Additional references:
  - Anthropic introspection: transformer-circuits.pub/2025/introspection
  - Song et al., arXiv:2509.13316

All tunable parameters live in config/patchscope.yaml — nothing is hardcoded here.
"""

import json
import logging
import random
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from cross_persona_introspection.experiments.base import BaseExperiment
from cross_persona_introspection.schemas import (
    PatchscopeRecord,
    PersonaConfig,
    RunConfig,
)

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
TASKS_DIR = Path(__file__).parent.parent.parent / "tasks"


# ── Helpers ──────────────────────────────────────────────────────────────


def _load_patchscope_config(config_filename: str) -> dict:
    """Load patchscope-specific YAML config."""
    path = CONFIG_DIR / config_filename
    if not path.exists():
        raise FileNotFoundError(f"Patchscope config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _load_questions(
    task_file: str,
    sample_size: Optional[int],
    seed: int,
    categories: Optional[list[str]] = None,
    samples_per_category: Optional[int] = None,
) -> list[dict]:
    """Load questions from JSON task file with optional category filtering and sampling.

    Args:
        task_file: JSON filename in tasks/ directory.
        sample_size: Global cap on total questions (applied last, from experiments.yaml).
        seed: Random seed for reproducible sampling.
        categories: If non-empty, only keep questions whose category_id is in this list.
        samples_per_category: If set, sample this many questions from each category.
    """
    path = TASKS_DIR / task_file
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    with open(path) as f:
        questions = json.load(f)

    # Filter by category
    if categories:
        questions = [q for q in questions if q.get("category_id") in categories]
        logger.info(f"Category filter {categories}: {len(questions)} questions remain")

    # Sample per category
    if samples_per_category is not None:
        rng = random.Random(seed)
        by_cat: dict[str, list[dict]] = {}
        for q in questions:
            by_cat.setdefault(q.get("category_id", "unknown"), []).append(q)
        sampled = []
        for cat_id in sorted(by_cat.keys()):
            pool = by_cat[cat_id]
            n = min(samples_per_category, len(pool))
            sampled.extend(rng.sample(pool, n))
        questions = sampled
        logger.info(f"Sampled {samples_per_category}/category: {len(questions)} questions total")

    # Global sample_size cap (from experiments.yaml)
    if sample_size is not None and sample_size < len(questions):
        rng = random.Random(seed)
        questions = rng.sample(questions, sample_size)

    return questions


def _model_short_name(model_name: str) -> str:
    """Extract a short name for file naming, e.g. 'meta-llama/Llama-3.1-8B-Instruct' -> 'llama31-8b'."""
    name = model_name.split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = name.strip("-")
    # Shorten common patterns
    name = name.replace("instruct", "").replace("chat", "").strip("-")
    if len(name) > 30:
        name = name[:30].rstrip("-")
    return name


def _resolve_layers(spec, num_layers: int) -> list[int]:
    """Resolve a layer specification to a list of layer indices.

    spec can be:
      "all"    — [0, 1, ..., num_layers-1]
      "middle" — middle third of layers
      int      — single layer
      list     — explicit list
    """
    if spec == "all":
        return list(range(num_layers))
    elif spec == "middle":
        third = num_layers // 3
        return list(range(third, 2 * third))
    elif isinstance(spec, int):
        return [spec]
    elif isinstance(spec, list):
        return [int(x) for x in spec]
    else:
        raise ValueError(f"Unknown layer spec: {spec}")


def _get_transformer_layers(model):
    """Return the nn.ModuleList of transformer layers for supported architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama, Mistral, Qwen
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2
    else:
        raise ValueError(
            "Cannot find transformer layers. "
            "Checked: model.model.layers, model.transformer.h. "
            "Add support for this architecture in _get_transformer_layers()."
        )


def _get_placeholder_token_id(tokenizer, configured_token: str = "auto") -> int:
    """Resolve the placeholder token to a single token ID.

    Args:
        tokenizer: HuggingFace tokenizer.
        configured_token: From config — "auto", "?", "<unk>", or any literal string.

    Patchscopes uses "?" (single token at position i*).
    SelfIE uses unk_token (multiple filler tokens).
    """
    if configured_token == "auto":
        # Try "?" first (Patchscopes default)
        ids = tokenizer.encode("?", add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
        # Fall back to unk
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id != tokenizer.eos_token_id:
            return tokenizer.unk_token_id
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
            return tokenizer.pad_token_id
        return 1

    # Explicit token string from config
    if configured_token == "<unk>" and tokenizer.unk_token_id is not None:
        return tokenizer.unk_token_id
    if configured_token == "<pad>" and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id

    # Try encoding the literal string
    ids = tokenizer.encode(configured_token, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]

    # Multi-token — warn and use first token
    logger.warning(
        f"Placeholder token '{configured_token}' encodes to {len(ids)} tokens "
        f"({ids}); using first token ID {ids[0]}"
    )
    return ids[0] if ids else 1


def find_token_position(tokenizer, text: str, word: str, strategy: str = "last") -> int:
    """Find the token position of a word in tokenized text.

    Args:
        tokenizer: HuggingFace tokenizer.
        text: Full input text.
        word: Word to find (e.g., "CEO").
        strategy: "last" uses the last subtoken of the word, "first" uses the first.

    Returns:
        Token position (0-indexed).
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Find contiguous span that decodes to the target word
    word_ids = tokenizer.encode(word, add_special_tokens=False)

    # Search for the subsequence
    for i in range(len(token_ids) - len(word_ids) + 1):
        if token_ids[i : i + len(word_ids)] == word_ids:
            if strategy == "last":
                return i + len(word_ids) - 1
            else:
                return i

    # Fallback: search by decoded string matching
    for i, tok_str in enumerate(tokens):
        if word.lower() in tok_str.lower().strip():
            return i

    raise ValueError(
        f"Could not find '{word}' in tokenized text. "
        f"Tokens: {list(enumerate(tokens))}"
    )


def _shuffle_options(question: dict, seed: int, qid: str, persona: str) -> tuple[dict, dict[str, str]]:
    """Shuffle option order for a question, returning (shuffled_question, remap).

    Returns:
        shuffled_question: question dict with options reordered (new labels A-D)
        remap: dict mapping shuffled label -> original label, e.g. {"A": "C", "B": "A", ...}
    """
    rng = random.Random(seed + hash(qid + persona))
    original_labels = ["A", "B", "C", "D"]
    shuffled_labels = original_labels.copy()
    rng.shuffle(shuffled_labels)
    # shuffled_labels[i] is the original label that goes into position i
    # e.g. shuffled_labels = ["C", "A", "D", "B"] means new A = old C
    options = question.get("options", {})
    new_options = {}
    remap = {}  # new_label -> original_label
    for new_pos, new_label in enumerate(original_labels):
        orig_label = shuffled_labels[new_pos]
        if orig_label in options:
            new_options[new_label] = options[orig_label]
            remap[new_label] = orig_label
    shuffled_q = dict(question)
    shuffled_q["options"] = new_options
    return shuffled_q, remap


def _format_options_lines(options: dict) -> str:
    """Format options as one per line: A) ..., B) ..., matching source prompt format."""
    return "\n".join(f"{k}) {v}" for k, v in options.items())


def _format_question_for_source(question: dict) -> str:
    """Format a question dict into a user-facing MCQ string for the source pass."""
    lines = [question["question_text"], ""]
    for letter in ["A", "B", "C", "D"]:
        if letter in question.get("options", {}):
            lines.append(f"{letter}) {question['options'][letter]}")
    lines.append("")
    lines.append(
        "Choose the single best answer. "
        "Answer with exactly one letter: A, B, C, or D."
    )
    return "\n".join(lines)


# ── Constrained answer parsing ───────────────────────────────────────────

# Expected keywords per template — search the full response for these
_TEMPLATE_KEYWORDS: dict[str, list[str]] = {
    "answer_extraction": ["A", "B", "C", "D"],
    "persona_probe": ["CONSERVATIVE", "PROGRESSIVE"],
    "agreement_probe": ["AGREE", "DISAGREE"],
    "self_recognition": ["SELF", "OTHER"],
}


def _parse_constrained(template_name: str, generated_text: str) -> tuple[Optional[str], bool]:
    """Parse a constrained response by searching for expected keywords.

    Returns (parsed_answer, parse_success). Searches the full response text
    for known keywords rather than blindly taking the first word, since models
    often wrap answers in preamble like "Based on the representation, ...".
    """
    keywords = _TEMPLATE_KEYWORDS.get(template_name)
    if not keywords:
        # Unknown template — fall back to first alphabetic token
        cleaned = re.sub(r"[^A-Za-z\s]", "", generated_text.strip()).upper()
        first = cleaned.split()[0] if cleaned.split() else None
        return (first, first is not None)

    text_upper = generated_text.upper()

    # For answer_extraction, look for standalone A/B/C/D (not inside words)
    if template_name == "answer_extraction":
        # First try: response is just a letter
        stripped = generated_text.strip().upper()
        if stripped and stripped[0] in "ABCD" and (len(stripped) == 1 or not stripped[1].isalpha()):
            return (stripped[0], True)
        # Second try: find last standalone letter (models often explain then answer)
        for match in reversed(list(re.finditer(r'\b([ABCD])\b', text_upper))):
            return (match.group(1), True)
        return (None, False)

    # For other templates: find the first matching keyword in the text
    # Sort by position in text (earliest match wins)
    found = []
    for kw in keywords:
        # Use word boundary to avoid matching substrings (e.g. "DISAGREE" contains "AGREE")
        match = re.search(r'\b' + kw + r'\b', text_upper)
        if match:
            found.append((match.start(), kw))

    if found:
        found.sort(key=lambda x: x[0])
        return (found[0][1], True)

    return (None, False)


# ── Core patching functions ──────────────────────────────────────────────


def extract_activation(
    model,
    tokenizer,
    device,
    messages: list[dict],
    layer_idx: int,
    token_position: str | int = "last",
    raw_text: Optional[str] = None,
) -> torch.Tensor:
    """Run a source forward pass and capture the hidden state at (layer, position).

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        device: Device for input tensors.
        messages: Chat messages for the source prompt.
        layer_idx: Which transformer layer to hook (0-indexed, excluding embedding).
        token_position: "last" or an integer index.
        raw_text: If provided, use this as input text directly (skip chat template).

    Returns:
        Activation tensor of shape (d_model,).
    """
    if raw_text is not None:
        input_text = raw_text
    else:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # add_special_tokens=False because apply_chat_template already includes BOS
    input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(device)

    captured = {}
    transformer_layers = _get_transformer_layers(model)

    def hook_fn(module, input, output):
        # output is typically a tuple; first element is hidden states
        hidden = output[0] if isinstance(output, tuple) else output
        if token_position == "last":
            pos = -1
        else:
            pos = int(token_position)
        captured["activation"] = hidden[0, pos, :].detach().clone()

    handle = transformer_layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    return captured["activation"]


def inject_and_generate(
    model,
    tokenizer,
    device,
    messages: list[dict],
    activation: torch.Tensor,
    injection_layer: int,
    placeholder_positions: list[int],
    mode: str = "replace",
    alpha: float = 1.0,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    do_sample: bool = False,
    raw_text: Optional[str] = None,
) -> str:
    """Run the interpretation forward pass with activation injection, then generate.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        device: Device for input tensors.
        messages: Chat messages for the interpretation prompt (contains placeholders).
        activation: Source activation tensor of shape (d_model,).
        injection_layer: Which layer to inject at.
        placeholder_positions: Token indices where placeholders are in the input.
        mode: "replace" or "add".
        alpha: Scaling factor for "add" mode.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        do_sample: Whether to sample.
        raw_text: If provided, use this as input text directly (skip chat template).

    Returns:
        Generated text string.
    """
    if raw_text is not None:
        input_text = raw_text
    else:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # add_special_tokens=False because apply_chat_template already includes BOS
    input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids)

    transformer_layers = _get_transformer_layers(model)
    act = activation.to(device)

    def injection_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        for pos in placeholder_positions:
            if pos < hidden.shape[1]:
                if mode == "replace":
                    hidden[0, pos, :] = act
                elif mode == "add":
                    hidden[0, pos, :] = hidden[0, pos, :] + alpha * act
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = transformer_layers[injection_layer].register_forward_hook(injection_hook)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,  # KV cache ON — hook only fires during prefill
            )
    finally:
        handle.remove()

    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Template → valid choices mapping ─────────────────────────────────────

TEMPLATE_CHOICES: dict[str, list[str]] = {
    "answer_extraction": ["A", "B", "C", "D"],
    "persona_probe": ["Conservative", "Progressive"],
    "agreement_probe": ["Agree", "Disagree"],
    "self_recognition": ["Self", "Other"],
}


def _resolve_choice_token_ids(
    tokenizer, choices: list[str]
) -> dict[str, int]:
    """Map each choice string to a single token ID.

    Tries the bare string first, then space-prefixed (e.g. " A" vs "A"),
    since models often predict a leading space after a prompt.
    Returns {choice_label: token_id}.
    """
    token_ids = {}
    for choice in choices:
        for variant in [choice, f" {choice}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids[choice] = ids[0]
                break
        if choice not in token_ids:
            # Multi-token — use first token and warn
            ids = tokenizer.encode(choice, add_special_tokens=False)
            if ids:
                token_ids[choice] = ids[0]
                logger.warning(
                    f"Choice '{choice}' is multi-token ({ids}); using first token {ids[0]}"
                )
    return token_ids


def inject_and_extract_logits(
    model,
    tokenizer,
    device,
    messages: list[dict],
    activation: Optional[torch.Tensor],
    injection_layer: int,
    placeholder_positions: list[int],
    choice_token_ids: dict[str, int],
    mode: str = "replace",
    alpha: float = 1.0,
    raw_text: Optional[str] = None,
) -> dict:
    """Run interpretation forward pass with activation injection, extract logits over choices.

    Single forward pass, no generation. Returns the probability distribution
    over the constrained choice set at the next-token position.

    If activation is None (text_only_baseline), runs without injection.

    Returns:
        {"probs": {choice: float}, "logits": {choice: float}, "predicted": str}
    """
    if raw_text is not None:
        input_text = raw_text
    else:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # add_special_tokens=False because apply_chat_template already includes BOS
    input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(device)

    transformer_layers = _get_transformer_layers(model)

    handle = None
    if activation is not None:
        act = activation.to(device)

        def injection_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for pos in placeholder_positions:
                if pos < hidden.shape[1]:
                    if mode == "replace":
                        hidden[0, pos, :] = act
                    elif mode == "add":
                        hidden[0, pos, :] = hidden[0, pos, :] + alpha * act
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        handle = transformer_layers[injection_layer].register_forward_hook(injection_hook)

    try:
        with torch.no_grad():
            outputs = model(input_ids)
    finally:
        if handle is not None:
            handle.remove()

    # Next-token logits (last position)
    next_logits = outputs.logits[0, -1, :]  # (vocab_size,)

    # Extract logits for valid choices only
    choice_logit_values = {}
    for choice, tid in choice_token_ids.items():
        choice_logit_values[choice] = next_logits[tid].item()

    # Softmax over choices only (constrained distribution)
    logit_tensor = torch.tensor(list(choice_logit_values.values()))
    probs = F.softmax(logit_tensor, dim=0)
    choice_probs = {
        choice: probs[i].item()
        for i, choice in enumerate(choice_logit_values.keys())
    }

    predicted = max(choice_probs, key=choice_probs.get)

    return {
        "probs": choice_probs,
        "logits": choice_logit_values,
        "predicted": predicted,
    }


def compute_relevancy_scores(
    model,
    tokenizer,
    device,
    messages: list[dict],
    activation: torch.Tensor,
    injection_layer: int,
    placeholder_positions: list[int],
    mode: str,
    alpha: float,
    generated_token_ids: list[int],
    max_tokens: int = 64,
) -> list[float]:
    """Compute SelfIE relevancy scores for generated tokens.

    relevancy(t_i) = P(t_i | WITH injection) - P(t_i | WITHOUT injection)

    This requires two forward passes per token position — one with injection,
    one without. We do this for the first max_tokens generated tokens.

    Returns:
        List of relevancy scores, one per generated token (up to max_tokens).
    """
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # add_special_tokens=False because apply_chat_template already includes BOS
    base_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(device)

    transformer_layers = _get_transformer_layers(model)
    act = activation.to(device)
    scores = []

    n_tokens = min(len(generated_token_ids), max_tokens)

    for i in range(n_tokens):
        # Build input: base prompt + generated tokens so far
        if i > 0:
            prefix_gen = torch.tensor(
                [generated_token_ids[:i]], device=device
            )
            full_ids = torch.cat([base_ids, prefix_gen], dim=1)
        else:
            full_ids = base_ids

        target_token_id = generated_token_ids[i]

        # Forward WITH injection
        def injection_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for pos in placeholder_positions:
                if pos < hidden.shape[1]:
                    if mode == "replace":
                        hidden[0, pos, :] = act
                    elif mode == "add":
                        hidden[0, pos, :] = hidden[0, pos, :] + alpha * act
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        handle = transformer_layers[injection_layer].register_forward_hook(injection_hook)
        try:
            with torch.no_grad():
                out_with = model(full_ids)
        finally:
            handle.remove()

        logits_with = out_with.logits[0, -1, :]
        prob_with = F.softmax(logits_with, dim=-1)[target_token_id].item()

        # Forward WITHOUT injection
        with torch.no_grad():
            out_without = model(full_ids)

        logits_without = out_without.logits[0, -1, :]
        prob_without = F.softmax(logits_without, dim=-1)[target_token_id].item()

        scores.append(prob_with - prob_without)

    return scores


# ── Experiment class ─────────────────────────────────────────────────────


class PatchscopeExperiment(BaseExperiment):
    """Patchscope activation interpretation experiment.

    Implements the full experiment matrix:
      source_persona × evaluator_persona × template × condition
    where condition ∈ {real, text_only_baseline, shuffled}.
    """

    def __init__(self, config: RunConfig, all_personas: dict[str, PersonaConfig]):
        super().__init__(config)
        self.all_personas = all_personas
        self.ps_config: dict = {}
        self.backend = None
        self.questions: list[dict] = []
        self.records: list[PatchscopeRecord] = []
        self._sample_prompts: dict[str, dict] = {}  # template -> {prompt, response}
        self._errors: list[str] = []
        self._run_elapsed: float = 0.0
        # Output paths — set early so incremental saves work
        self._jsonl_path: Optional[Path] = None
        self._log_path: Optional[Path] = None
        self._base_name: str = ""
        self._last_flush: int = 0  # records count at last flush

    def setup(self) -> None:
        import transformers
        import torch as _torch
        from cross_persona_introspection.backends.hf_backend import HFBackend

        # Suppress "Both max_new_tokens and max_length seem to have been set" warnings
        transformers.logging.set_verbosity_error()

        # Load patchscope-specific config
        if not self.config.patchscope_config:
            raise ValueError("patchscope experiment requires 'patchscope_config' in experiment config")
        self.ps_config = _load_patchscope_config(self.config.patchscope_config)

        # Load model
        dtype_map = {
            "bfloat16": _torch.bfloat16,
            "float16": _torch.float16,
            "float32": _torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.model_dtype) if self.config.model_dtype else None
        logger.info(f"Loading model: {self.config.model_name}  dtype={self.config.model_dtype or 'auto'}")
        self.backend = HFBackend(self.config.model_name, device="auto", torch_dtype=torch_dtype)

        # Load questions
        task_file = self.config.task_file
        if not task_file:
            raise ValueError("patchscope experiment requires 'task_file' in experiment config")
        self.questions = _load_questions(
            task_file,
            sample_size=self.config.sample_size,
            seed=self.config.seed,
            categories=self.ps_config.get("categories") or None,
            samples_per_category=self.ps_config.get("samples_per_category"),
        )
        logger.info(f"Loaded {len(self.questions)} questions from {task_file}")

    def _init_output_paths(self) -> None:
        """Set up output file paths once, early, so incremental saves work."""
        if self._jsonl_path is not None:
            return  # already initialised
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = _model_short_name(self.config.model_name)
        if self.ps_config.get("layer_sweep", {}).get("enabled", False):
            exp_type = "layer_sweep"
        else:
            exp_type = "matrix"
        self._base_name = f"patchscope_{model_short}_{timestamp}_{exp_type}"
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = out_dir / f"{self._base_name}.jsonl"
        self._log_path = out_dir / f"{self._base_name}.txt"

    def _flush_results(self) -> None:
        """Write all records collected so far to JSONL (overwrite) and update the log."""
        if self._jsonl_path is None:
            self._init_output_paths()
        # Write full JSONL (overwrite — keeps file consistent if interrupted)
        with open(self._jsonl_path, "w") as f:
            for record in self.records:
                f.write(json.dumps(asdict(record), default=str) + "\n")
        # Update companion log
        self._write_log(self._log_path, self._base_name)
        self._last_flush = len(self.records)
        logger.info(
            f"  Checkpoint: flushed {len(self.records)} records to {self._jsonl_path.name}"
        )

    def run(self) -> None:
        assert self.backend is not None
        model = self.backend.model
        tokenizer = self.backend.tokenizer
        device = self.backend.input_device
        ps = self.ps_config

        self._init_output_paths()
        run_start = time.monotonic()

        num_layers = model.config.num_hidden_layers
        extraction_cfg = ps["extraction"]
        injection_cfg = ps["injection"]
        gen_cfg = ps["generation"]
        relevancy_cfg = ps["relevancy"]
        controls_cfg = ps["controls"]
        templates = ps["interpretation_templates"]

        # Filter templates if enabled_templates is set
        enabled = ps.get("enabled_templates", [])
        if enabled:
            templates = {k: v for k, v in templates.items() if k in enabled}
            logger.info(f"Template filter: running only {list(templates.keys())}")
        else:
            logger.info(f"Running all templates: {list(templates.keys())}")

        base_prompt = ps["interpretation_base_prompt"]

        source_persona_names = ps["source_personas"]
        evaluator_persona_names = ps["evaluator_personas"]

        # Resolve extraction layers
        # Priority: layer_pairs > layer_sweep > extraction.layers + injection.layer
        explicit_pairs = ps.get("layer_pairs")
        if explicit_pairs:
            # Explicit pairs mode — source_layers is the unique set of source layers
            source_layers = sorted(set(int(p[0]) for p in explicit_pairs))
            # injection_layers will be looked up per source_layer below
            injection_layers = sorted(set(int(p[1]) for p in explicit_pairs))
            _pair_map: dict[int, list[int]] = {}
            for p in explicit_pairs:
                _pair_map.setdefault(int(p[0]), []).append(int(p[1]))
            logger.info(f"Using explicit layer_pairs: {explicit_pairs}")
        elif ps.get("layer_sweep", {}).get("enabled", False):
            sweep = ps["layer_sweep"]
            source_layers = [int(x) for x in sweep["source_layers"]]
            injection_layers = [int(x) for x in sweep["injection_layers"]]
            _pair_map = None
        else:
            source_layers = _resolve_layers(extraction_cfg["layers"], num_layers)
            injection_layers = [int(injection_cfg["layer"])]
            _pair_map = None

        injection_mode = injection_cfg["mode"]
        injection_alpha = float(injection_cfg["alpha"])

        # Resolve prompt style (top-level prompt_style takes precedence)
        style = ps.get("prompt_style") or injection_cfg.get("placeholder_style", "patchscopes")
        configured_placeholder = injection_cfg.get("placeholder_token", "?")
        if style in ("patchscopes", "identity"):
            num_placeholders = 1
        elif style == "selfie":
            num_placeholders = 5
        else:  # "custom"
            num_placeholders = int(injection_cfg["num_placeholders"])

        placeholder_token_id = _get_placeholder_token_id(tokenizer, configured_placeholder)
        placeholder_token = tokenizer.decode([placeholder_token_id])

        control_sample_rate = float(controls_cfg.get("control_sample_rate", 1.0))
        control_rng = random.Random(self.config.seed + 7)  # separate seed for control sampling

        # Pre-resolve choice token IDs for templates using logits decode_mode
        all_choice_token_ids: dict[str, dict[str, int]] = {}
        for tmpl_name, tmpl_cfg in templates.items():
            if tmpl_cfg.get("decode_mode", "generate") == "logits" and tmpl_name in TEMPLATE_CHOICES:
                choices = TEMPLATE_CHOICES[tmpl_name]
                all_choice_token_ids[tmpl_name] = _resolve_choice_token_ids(
                    tokenizer, choices
                )
                logger.info(
                    f"Logits mode — {tmpl_name} choices: "
                    f"{all_choice_token_ids[tmpl_name]}"
                )
        logger.info(
            f"Placeholder token: id={placeholder_token_id} repr={repr(placeholder_token)} "
            f"× {num_placeholders}"
        )

        # Pre-compute source activations and direct answers for all (source_persona, question, layer)
        # This avoids redundant forward passes when sweeping evaluators/templates.
        logger.info("=== Phase 1: Extracting source activations ===")
        # activations[source_persona][question_idx][layer] = tensor(d_model,)
        activations: dict[str, dict[int, dict[int, torch.Tensor]]] = {}
        # direct_answers[source_persona][question_idx] = {"answer": str, "probs": dict}
        direct_answers: dict[str, dict[int, dict]] = {}
        # shuffle_maps[(qid, sp_name)] = (shuffled_question, remap)
        shuffle_maps: dict[tuple[str, str], tuple[dict, dict[str, str]]] = {}
        # Sample source prompts for .txt log
        self._sample_source_prompts: dict[str, dict] = {}

        for sp_name in source_persona_names:
            activations[sp_name] = {}
            direct_answers[sp_name] = {}

            sp = self.all_personas[sp_name]

            for qi, question in enumerate(tqdm(
                self.questions, desc=f"  Extract [{sp_name}]", leave=True
            )):
                try:
                    qid = question.get("question_id", f"q{qi}")

                    # Shuffle option order to control for label-position bias
                    shuffled_q, remap = _shuffle_options(
                        question, self.config.seed, qid, sp_name
                    )
                    shuffle_maps[(qid, sp_name)] = (shuffled_q, remap)
                    inverse_remap = {v: k for k, v in remap.items()}  # orig -> shuffled

                    user_msg = _format_question_for_source(shuffled_q)
                    messages = []
                    if sp.system_prompt:
                        messages.append({"role": "system", "content": sp.system_prompt})
                    messages.append({"role": "user", "content": user_msg})

                    # Extract activations at all needed layers
                    activations[sp_name][qi] = {}
                    for layer in source_layers:
                        act = extract_activation(
                            model, tokenizer, device, messages,
                            layer_idx=layer,
                            token_position=extraction_cfg["token_position"],
                        )
                        activations[sp_name][qi][layer] = act

                    # Get source's direct answer via logits (same method as reporter)
                    probs, logits_dict = self.backend.get_choice_probs_and_logits(
                        messages, ["A", "B", "C", "D"]
                    )
                    shuffled_answer = max(probs, key=probs.get)
                    # Remap back to canonical (original) label
                    canonical_answer = remap.get(shuffled_answer, shuffled_answer)
                    # Remap probs/logits keys back to canonical
                    canonical_probs = {remap.get(k, k): v for k, v in probs.items()}
                    canonical_logits = {remap.get(k, k): v for k, v in logits_dict.items()}
                    direct_answers[sp_name][qi] = {
                        "answer": canonical_answer,
                        "probs": canonical_probs,
                        "logits": canonical_logits,
                        "raw": f"[logits] {canonical_answer}",
                        "shuffled_answer": shuffled_answer,
                        "option_order": [remap.get(l, l) for l in ["A", "B", "C", "D"]],
                    }

                    # Capture sample source prompt for .txt log
                    if sp_name not in self._sample_source_prompts:
                        self._sample_source_prompts[sp_name] = {
                            "persona": sp_name,
                            "question_id": qid,
                            "prompt_text": tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            ),
                            "answer": canonical_answer,
                            "shuffled_answer": shuffled_answer,
                            "option_order": [remap.get(l, l) for l in ["A", "B", "C", "D"]],
                        }

                except Exception as e:
                    msg = f"Extract error: {sp_name} q{qi} {question.get('question_id', '?')}: {e}"
                    logger.error(msg)
                    self._errors.append(msg)
                    activations.setdefault(sp_name, {})[qi] = {}
                    direct_answers.setdefault(sp_name, {})[qi] = {
                        "answer": None, "probs": {}, "logits": {}, "raw": ""
                    }

        # Phase 2: Run interpretation matrix
        logger.info("=== Phase 2: Running interpretation matrix ===")

        # Build conditions list
        conditions = ["real"]
        if controls_cfg.get("text_only_baseline", False):
            conditions.append("text_only_baseline")
        if controls_cfg.get("shuffled_activation", False):
            conditions.append("shuffled")

        total_cells = (
            len(source_persona_names) * len(self.questions) * len(source_layers)
            * len(injection_layers) * len(evaluator_persona_names)
            * len(templates) * len(conditions)
        )
        logger.info(f"Total matrix cells: {total_cells}")

        # ── Baseline cache ──────────────────────────────────────────────
        # text_only_baseline results depend only on (question, evaluator, template)
        # — not on source_persona, source_layer, or injection_layer (no activation
        # is injected). Cache them to avoid redundant forward passes.
        # Key: (question_id, evaluator_persona, template_name)
        # Value: dict with logits/generate results
        _baseline_cache: dict[tuple, dict] = {}

        cell_count = 0
        for sp_name in source_persona_names:
            for qi, question in enumerate(self.questions):
                qid = question.get("question_id", f"q{qi}")

                # Skip questions where extraction failed entirely
                if not activations.get(sp_name, {}).get(qi, {}):
                    logger.warning(
                        f"Skipping {sp_name} q={qid}: extraction produced no activations"
                    )
                    continue

                for src_layer in source_layers:
                    # Get the real activation (may be empty if extraction failed for this layer)
                    real_act = activations.get(sp_name, {}).get(qi, {}).get(src_layer)
                    if real_act is None:
                        logger.warning(
                            f"Skipping {sp_name} q={qid} layer={src_layer}: no activation"
                        )
                        continue

                    # Get a shuffled activation (from a different question)
                    shuffled_act = None
                    if "shuffled" in conditions and len(self.questions) > 1:
                        other_qi = (qi + 1) % len(self.questions)
                        shuffled_act = activations.get(sp_name, {}).get(other_qi, {}).get(src_layer)

                    # When using explicit pairs, only iterate injection layers
                    # paired with this source layer
                    _inj_layers = _pair_map[src_layer] if _pair_map else injection_layers
                    for inj_layer in _inj_layers:
                        for eval_name in evaluator_persona_names:
                            eval_persona = self.all_personas[eval_name]

                            for tmpl_name, tmpl_cfg in templates.items():
                                # Select prompt style matching placeholder_style
                                prompt_style = style if style in ("patchscopes", "selfie", "identity") else "selfie"
                                prompt_template = tmpl_cfg.get(prompt_style)
                                if not prompt_template:
                                    # Fall back to whichever exists
                                    prompt_template = tmpl_cfg.get("patchscopes") or tmpl_cfg.get("selfie", "")

                                # Build the placeholder string
                                placeholder_str = (placeholder_token + " ") * num_placeholders
                                placeholder_str = placeholder_str.strip()

                                # Use a sentinel to track placeholder position through
                                # chat template wrapping. This avoids matching "?" in
                                # question text when looking for the injection target.
                                _SENTINEL = "\x00PH\x00"

                                # Use shuffled options matching the source pass
                                _shuffle_key = (qid, sp_name)
                                if _shuffle_key in shuffle_maps:
                                    _q_for_template = shuffle_maps[_shuffle_key][0]
                                else:
                                    _q_for_template = question
                                _opts = _q_for_template.get("options", {})

                                options_str = ", ".join(
                                    f"{k}) {v}" for k, v in _opts.items()
                                )
                                options_formatted_str = _format_options_lines(_opts)

                                user_content = prompt_template.strip()
                                user_content = user_content.replace("{placeholder}", _SENTINEL)
                                user_content = user_content.replace(
                                    "{question_text}", question.get("question_text", "")
                                )
                                user_content = user_content.replace("{options_formatted}", options_formatted_str)
                                user_content = user_content.replace("{options}", options_str)
                                user_content = user_content.replace(
                                    "{statement}", question.get("question_text", "")
                                )

                                if prompt_style == "identity":
                                    # Identity: raw text, no chat template
                                    # Find sentinel position, then replace with actual placeholder
                                    _ph_char_pos = user_content.find(_SENTINEL)
                                    user_content = user_content.replace(_SENTINEL, placeholder_str)
                                    interp_text = user_content
                                    interp_messages = [{"role": "user", "content": user_content}]
                                    interp_ids = tokenizer.encode(
                                        interp_text, return_tensors="pt", add_special_tokens=False
                                    )
                                else:
                                    # Prepend base prompt if non-empty (selfie style)
                                    if base_prompt.strip():
                                        user_content = base_prompt.strip() + "\n\n" + user_content

                                    interp_messages_sentinel = []
                                    if eval_persona.system_prompt:
                                        interp_messages_sentinel.append({
                                            "role": "system",
                                            "content": eval_persona.system_prompt,
                                        })
                                    interp_messages_sentinel.append({
                                        "role": "user",
                                        "content": user_content,
                                    })

                                    # Build interp_text with sentinel to find char position
                                    interp_text_sentinel = tokenizer.apply_chat_template(
                                        interp_messages_sentinel, tokenize=False, add_generation_prompt=True
                                    )
                                    _ph_char_pos = interp_text_sentinel.find(_SENTINEL)

                                    # Now replace sentinel with actual placeholder
                                    user_content = user_content.replace(_SENTINEL, placeholder_str)
                                    interp_messages = []
                                    if eval_persona.system_prompt:
                                        interp_messages.append({
                                            "role": "system",
                                            "content": eval_persona.system_prompt,
                                        })
                                    interp_messages.append({
                                        "role": "user",
                                        "content": user_content,
                                    })

                                    interp_text = tokenizer.apply_chat_template(
                                        interp_messages, tokenize=False, add_generation_prompt=True
                                    )
                                    interp_ids = tokenizer.encode(
                                        interp_text, return_tensors="pt", add_special_tokens=False
                                    )

                                # Find placeholder token by mapping char offset to token index.
                                # Only matches the intended answer slot, not "?" in question text.
                                placeholder_positions = []
                                if _ph_char_pos >= 0:
                                    _cumulative = 0
                                    for ti, tid in enumerate(interp_ids[0].tolist()):
                                        _tok_str = tokenizer.decode([tid])
                                        _tok_start = _cumulative
                                        _cumulative += len(_tok_str)
                                        # Check if this token overlaps the placeholder span
                                        if (_tok_start < _ph_char_pos + len(placeholder_str)
                                                and _cumulative >= _ph_char_pos
                                                and placeholder_token in _tok_str.strip()):
                                            placeholder_positions.append(ti)

                                if not placeholder_positions:
                                    logger.warning(
                                        f"No placeholder positions found in prompt for "
                                        f"{tmpl_name}! Injection will be skipped."
                                    )

                                for condition in conditions:
                                    # Skip control conditions based on sampling rate
                                    if condition != "real" and control_sample_rate < 1.0:
                                        if control_rng.random() > control_sample_rate:
                                            continue
                                    cell_count += 1
                                    _da = direct_answers[sp_name][qi]
                                    record = PatchscopeRecord(
                                        experiment="patchscope",
                                        template_name=tmpl_name,
                                        model=self.config.model_name,
                                        question_id=qid,
                                        source_persona=sp_name,
                                        evaluator_persona=eval_name,
                                        condition=condition,
                                        source_layer=src_layer,
                                        injection_layer=inj_layer,
                                        injection_mode=injection_mode,
                                        source_direct_answer=_da["answer"],
                                        source_answer_probs=_da["probs"],
                                        question_text=question.get("question_text", ""),
                                        question_options=question.get("options"),
                                    )

                                    try:
                                        # Pick activation for this condition
                                        if condition == "text_only_baseline":
                                            act_for_condition = None
                                        elif condition == "shuffled":
                                            if shuffled_act is None:
                                                record.error = "no shuffled activation available"
                                                record.timestamp = datetime.now(timezone.utc).isoformat()
                                                self.records.append(record)
                                                continue
                                            act_for_condition = shuffled_act
                                        else:
                                            act_for_condition = real_act

                                        record.evaluator_system_prompt = eval_persona.system_prompt or ""

                                        # Get shuffle remap for answer remapping
                                        _shuffle_key = (qid, sp_name)
                                        _remap = shuffle_maps.get(_shuffle_key, (None, {}))[1]
                                        _inverse_remap = {v: k for k, v in _remap.items()} if _remap else {}

                                        # Decide: logits mode or generate mode
                                        use_logits = (
                                            tmpl_cfg.get("decode_mode", "generate") == "logits"
                                            and tmpl_name in all_choice_token_ids
                                        )

                                        # Pass raw_text for identity style to skip chat template
                                        _raw = interp_text if prompt_style == "identity" else None

                                        # For text_only_baseline: build prompt WITHOUT placeholder
                                        # so the model doesn't see "?" — it just predicts the next token
                                        # after "Answer: " naturally
                                        _raw_baseline = None
                                        if condition == "text_only_baseline" and _ph_char_pos >= 0:
                                            _raw_baseline = interp_text[:_ph_char_pos].rstrip()
                                            # Ensure it ends with a space after "Answer:"
                                            if not _raw_baseline.endswith(" "):
                                                _raw_baseline += " "

                                        # Log the actual prompt used for this condition
                                        if condition == "text_only_baseline" and _raw_baseline:
                                            record.interpretation_prompt = _raw_baseline
                                        else:
                                            record.interpretation_prompt = interp_text

                                        # ── Baseline deduplication ──
                                        # text_only_baseline depends only on (question, evaluator, template),
                                        # not on source_persona/layer. Reuse cached results.
                                        _baseline_key = (qid, eval_name, tmpl_name, sp_name)
                                        if condition == "text_only_baseline" and _baseline_key in _baseline_cache:
                                            cached = _baseline_cache[_baseline_key]
                                            record.decode_mode = cached["decode_mode"]
                                            record.generated_text = cached["generated_text"]
                                            record.parsed_answer = cached.get("parsed_answer")
                                            record.parse_success = cached.get("parse_success", False)
                                            record.choice_probs = cached.get("choice_probs")
                                            record.choice_logits = cached.get("choice_logits")
                                            record.predicted = cached.get("predicted")
                                            if tmpl_name == "answer_extraction" and record.source_direct_answer:
                                                record.is_correct = (
                                                    record.predicted == record.source_direct_answer
                                                )
                                            record.timestamp = datetime.now(timezone.utc).isoformat()
                                            self.records.append(record)
                                            cell_count += 1
                                            continue

                                        if use_logits:
                                            # ── Logits mode: single forward pass, no generation ──
                                            record.decode_mode = "logits"
                                            # For baseline, use prompt without placeholder
                                            _effective_raw = _raw_baseline if condition == "text_only_baseline" else _raw
                                            result = inject_and_extract_logits(
                                                model, tokenizer, device,
                                                interp_messages, act_for_condition,
                                                injection_layer=inj_layer,
                                                placeholder_positions=placeholder_positions,
                                                choice_token_ids=all_choice_token_ids[tmpl_name],
                                                mode=injection_mode,
                                                alpha=injection_alpha,
                                                raw_text=_effective_raw,
                                            )

                                            # Remap shuffled labels back to canonical
                                            if _remap and tmpl_name == "answer_extraction":
                                                canonical_probs = {_remap.get(k, k): v for k, v in result["probs"].items()}
                                                canonical_logits = {_remap.get(k, k): v for k, v in result["logits"].items()}
                                                canonical_predicted = _remap.get(result["predicted"], result["predicted"])
                                            else:
                                                canonical_probs = result["probs"]
                                                canonical_logits = result["logits"]
                                                canonical_predicted = result["predicted"]

                                            record.choice_probs = canonical_probs
                                            record.choice_logits = canonical_logits
                                            record.predicted = canonical_predicted
                                            record.parsed_answer = canonical_predicted
                                            record.parse_success = True
                                            record.generated_text = f"[logits] {canonical_predicted}"

                                            # Compute is_correct for answer_extraction
                                            if tmpl_name == "answer_extraction" and record.source_direct_answer:
                                                record.is_correct = (
                                                    canonical_predicted == record.source_direct_answer
                                                )

                                        else:
                                            # ── Generate mode: multi-token generation ──
                                            record.decode_mode = "generate"
                                            if act_for_condition is None:
                                                gen_text = self.backend.generate(
                                                    interp_messages,
                                                    max_new_tokens=gen_cfg["max_new_tokens"],
                                                    temperature=gen_cfg["temperature"],
                                                    do_sample=gen_cfg.get("do_sample", False),
                                                )
                                            else:
                                                gen_text = inject_and_generate(
                                                    model, tokenizer, device,
                                                    interp_messages, act_for_condition,
                                                    injection_layer=inj_layer,
                                                    placeholder_positions=placeholder_positions,
                                                    mode=injection_mode,
                                                    alpha=injection_alpha,
                                                    max_new_tokens=gen_cfg["max_new_tokens"],
                                                    temperature=gen_cfg["temperature"],
                                                    do_sample=gen_cfg.get("do_sample", False),
                                                    raw_text=_raw,
                                                )
                                            record.generated_text = gen_text

                                            # Parse answers from generated text if template has known choices
                                            if tmpl_name in TEMPLATE_CHOICES:
                                                record.parsed_answer, record.parse_success = (
                                                    _parse_constrained(tmpl_name, gen_text)
                                                )

                                            # Relevancy scores (optional, expensive)
                                            if (
                                                relevancy_cfg.get("enabled", False)
                                                and condition == "real"
                                                and real_act is not None
                                            ):
                                                gen_token_ids = tokenizer.encode(
                                                    gen_text, add_special_tokens=False
                                                )
                                                if gen_token_ids:
                                                    rel_scores = compute_relevancy_scores(
                                                        model, tokenizer, device,
                                                        interp_messages, real_act,
                                                        injection_layer=inj_layer,
                                                        placeholder_positions=placeholder_positions,
                                                        mode=injection_mode,
                                                        alpha=injection_alpha,
                                                        generated_token_ids=gen_token_ids,
                                                        max_tokens=relevancy_cfg.get("max_tokens", 64),
                                                    )
                                                    record.relevancy_scores = rel_scores
                                                    record.mean_relevancy = (
                                                        sum(rel_scores) / len(rel_scores)
                                                        if rel_scores else None
                                                    )

                                        # Cache baseline results for deduplication
                                        if condition == "text_only_baseline" and _baseline_key not in _baseline_cache:
                                            _baseline_cache[_baseline_key] = {
                                                "decode_mode": record.decode_mode,
                                                "generated_text": record.generated_text,
                                                "parsed_answer": record.parsed_answer,
                                                "parse_success": record.parse_success,
                                                "choice_probs": record.choice_probs,
                                                "choice_logits": record.choice_logits,
                                                "predicted": record.predicted,
                                            }

                                        # Capture sample prompt (first of each template)
                                        sample_key = f"{tmpl_name}_{condition}"
                                        if sample_key not in self._sample_prompts:
                                            self._sample_prompts[sample_key] = {
                                                "template": tmpl_name,
                                                "condition": condition,
                                                "source_persona": sp_name,
                                                "evaluator_persona": eval_name,
                                                "source_layer": src_layer,
                                                "injection_layer": inj_layer,
                                                "interp_prompt_text": interp_text,
                                                "generated_text": record.generated_text,
                                                "question_id": qid,
                                            }

                                    except Exception as e:
                                        msg = (
                                            f"Interpret error: {sp_name}->{eval_name} "
                                            f"{tmpl_name} {condition} q={qid}: {e}"
                                        )
                                        logger.error(msg)
                                        self._errors.append(msg)
                                        record.error = str(e)

                                    record.timestamp = datetime.now(timezone.utc).isoformat()
                                    self.records.append(record)

                                    if cell_count % 50 == 0:
                                        logger.info(f"  Progress: {cell_count}/{total_cells} cells")

                                    # Incremental save every 100 new records
                                    if len(self.records) - self._last_flush >= 20:
                                        self._run_elapsed = time.monotonic() - run_start
                                        self._flush_results()

        # ── Phase 3: Source overrides (raw text, specific word extraction) ──
        source_overrides = ps.get("source_overrides", [])
        if source_overrides:
            logger.info(f"=== Phase 3: Source overrides ({len(source_overrides)} entries) ===")

            for so_entry in source_overrides:
                so_name = so_entry["name"]
                so_raw_text = so_entry["raw_text"]
                so_extract_word = so_entry["extract_word"]
                so_strategy = so_entry.get("subtoken_strategy", "last")
                so_expected = so_entry.get("expected_contains", "")

                try:
                    so_token_pos = find_token_position(
                        tokenizer, so_raw_text, so_extract_word, so_strategy
                    )
                except ValueError as e:
                    msg = f"Source override '{so_name}': {e}"
                    logger.error(msg)
                    self._errors.append(msg)
                    continue

                logger.info(
                    f"  Override '{so_name}': '{so_raw_text}', "
                    f"extract '{so_extract_word}' at position {so_token_pos}"
                )

                # Extract activations at all source layers
                so_activations: dict[int, torch.Tensor] = {}
                for layer in source_layers:
                    try:
                        act = extract_activation(
                            model, tokenizer, device,
                            messages=[],
                            layer_idx=layer,
                            token_position=so_token_pos,
                            raw_text=so_raw_text,
                        )
                        so_activations[layer] = act
                    except Exception as e:
                        msg = f"Extract error: override '{so_name}' layer={layer}: {e}"
                        logger.error(msg)
                        self._errors.append(msg)

                if not so_activations:
                    continue

                # Run through templates × layer pairs (same loop as Phase 2
                # but with source_persona="source_override", no evaluator sweep,
                # only "real" condition)
                for src_layer in source_layers:
                    real_act = so_activations.get(src_layer)
                    if real_act is None:
                        continue

                    _inj_layers = _pair_map[src_layer] if _pair_map else injection_layers
                    for inj_layer in _inj_layers:
                        for tmpl_name, tmpl_cfg in templates.items():
                            prompt_style = style if style in ("patchscopes", "selfie", "identity") else "selfie"
                            prompt_template = tmpl_cfg.get(prompt_style)
                            if not prompt_template:
                                prompt_template = tmpl_cfg.get("patchscopes") or tmpl_cfg.get("selfie", "")

                            placeholder_str = (placeholder_token + " ") * num_placeholders
                            placeholder_str = placeholder_str.strip()

                            user_content = prompt_template.strip()
                            user_content = user_content.replace("{placeholder}", placeholder_str)
                            user_content = user_content.replace("{question_text}", so_raw_text)
                            user_content = user_content.replace("{options}", "")
                            user_content = user_content.replace("{statement}", so_raw_text)

                            if prompt_style == "identity":
                                interp_text = user_content
                                interp_messages = [{"role": "user", "content": user_content}]
                                interp_ids = tokenizer.encode(
                                    interp_text, return_tensors="pt", add_special_tokens=False
                                )
                            else:
                                if base_prompt.strip():
                                    user_content = base_prompt.strip() + "\n\n" + user_content
                                interp_messages = [{"role": "user", "content": user_content}]
                                interp_text = tokenizer.apply_chat_template(
                                    interp_messages, tokenize=False, add_generation_prompt=True
                                )
                                interp_ids = tokenizer.encode(
                                    interp_text, return_tensors="pt", add_special_tokens=False
                                )

                            placeholder_positions = [
                                i for i, tid in enumerate(interp_ids[0].tolist())
                                if placeholder_token in tokenizer.decode([tid]).strip()
                            ]
                            if not placeholder_positions:
                                placeholder_positions = (
                                    interp_ids[0] == placeholder_token_id
                                ).nonzero(as_tuple=True)[0].tolist()

                            _raw = interp_text if prompt_style == "identity" else None

                            record = PatchscopeRecord(
                                experiment="patchscope",
                                template_name=tmpl_name,
                                model=self.config.model_name,
                                question_id=f"override_{so_name}",
                                source_persona="source_override",
                                evaluator_persona="source_override",
                                condition="real",
                                source_layer=src_layer,
                                injection_layer=inj_layer,
                                injection_mode=injection_mode,
                                source_direct_answer=None,
                                source_answer_probs=None,
                                question_text=so_raw_text,
                                question_options=None,
                            )

                            try:
                                record.interpretation_prompt = interp_text

                                use_logits = (
                                    tmpl_cfg.get("decode_mode", "generate") == "logits"
                                    and tmpl_name in all_choice_token_ids
                                )

                                if use_logits:
                                    record.decode_mode = "logits"
                                    result = inject_and_extract_logits(
                                        model, tokenizer, device,
                                        interp_messages, real_act,
                                        injection_layer=inj_layer,
                                        placeholder_positions=placeholder_positions,
                                        choice_token_ids=all_choice_token_ids[tmpl_name],
                                        mode=injection_mode,
                                        alpha=injection_alpha,
                                        raw_text=_raw,
                                    )
                                    record.choice_probs = result["probs"]
                                    record.choice_logits = result["logits"]
                                    record.predicted = result["predicted"]
                                    record.parsed_answer = result["predicted"]
                                    record.parse_success = True
                                    record.generated_text = f"[logits] {result['predicted']}"
                                else:
                                    record.decode_mode = "generate"
                                    gen_text = inject_and_generate(
                                        model, tokenizer, device,
                                        interp_messages, real_act,
                                        injection_layer=inj_layer,
                                        placeholder_positions=placeholder_positions,
                                        mode=injection_mode,
                                        alpha=injection_alpha,
                                        max_new_tokens=gen_cfg["max_new_tokens"],
                                        temperature=gen_cfg["temperature"],
                                        do_sample=gen_cfg.get("do_sample", False),
                                        raw_text=_raw,
                                    )
                                    record.generated_text = gen_text

                                # Check expected output
                                found = (
                                    so_expected.lower() in record.generated_text.lower()
                                    if so_expected else None
                                )
                                status = "✓" if found else "✗" if found is False else " "
                                logger.info(
                                    f"    {so_name} L{src_layer:>2}→{inj_layer:>2} "
                                    f"{tmpl_name}: [{status}] '{record.generated_text.strip()[:60]}'"
                                )

                                # Capture sample prompt
                                sample_key = f"override_{so_name}_{tmpl_name}"
                                if sample_key not in self._sample_prompts:
                                    self._sample_prompts[sample_key] = {
                                        "template": tmpl_name,
                                        "condition": "real (source_override)",
                                        "source_persona": "source_override",
                                        "evaluator_persona": "source_override",
                                        "source_layer": src_layer,
                                        "injection_layer": inj_layer,
                                        "interp_prompt_text": interp_text,
                                        "generated_text": record.generated_text,
                                        "question_id": f"override_{so_name}",
                                    }

                            except Exception as e:
                                msg = (
                                    f"Override interpret error: {so_name} "
                                    f"{tmpl_name} L{src_layer}→{inj_layer}: {e}"
                                )
                                logger.error(msg)
                                self._errors.append(msg)
                                record.error = str(e)

                            record.timestamp = datetime.now(timezone.utc).isoformat()
                            self.records.append(record)

                # Incremental save after each override entry
                self._run_elapsed = time.monotonic() - run_start
                self._flush_results()

        self._run_elapsed = time.monotonic() - run_start
        self._flush_results()
        logger.info(f"Completed {len(self.records)} records in {self._run_elapsed:.1f}s")

    def evaluate(self) -> dict:
        """Compute aggregate metrics from collected records."""
        if not self.records:
            return {"status": "no records"}

        total = len(self.records)
        errors = sum(1 for r in self.records if r.error)
        constrained = [r for r in self.records if r.parse_success]

        metrics = {
            "total_records": total,
            "errors": errors,
            "parsed_constrained": len(constrained),
        }

        # Per-condition accuracy for answer_extraction template
        for condition in ["real", "text_only_baseline", "shuffled"]:
            answer_recs = [
                r for r in constrained
                if r.template_name == "answer_extraction"
                and r.condition == condition
                and r.source_direct_answer is not None
            ]
            if answer_recs:
                correct = sum(
                    1 for r in answer_recs
                    if r.parsed_answer and r.parsed_answer[0] == r.source_direct_answer
                )
                metrics[f"answer_extraction_{condition}_accuracy"] = round(
                    correct / len(answer_recs), 4
                )
                metrics[f"answer_extraction_{condition}_n"] = len(answer_recs)

        # Persona probe accuracy
        for condition in ["real", "text_only_baseline", "shuffled"]:
            probe_recs = [
                r for r in constrained
                if r.template_name == "persona_probe" and r.condition == condition
            ]
            if probe_recs:
                correct = 0
                for r in probe_recs:
                    expected = (
                        "CONSERVATIVE" if "conservative" in r.source_persona
                        else "PROGRESSIVE"
                    )
                    if r.parsed_answer and expected.startswith(r.parsed_answer[:4]):
                        correct += 1
                metrics[f"persona_probe_{condition}_accuracy"] = round(
                    correct / len(probe_recs), 4
                )
                metrics[f"persona_probe_{condition}_n"] = len(probe_recs)

        # Self-recognition accuracy
        for condition in ["real"]:
            self_recs = [
                r for r in constrained
                if r.template_name == "self_recognition" and r.condition == condition
            ]
            if self_recs:
                correct = 0
                for r in self_recs:
                    is_self = r.source_persona == r.evaluator_persona
                    expected = "SELF" if is_self else "OTHER"
                    if r.parsed_answer and r.parsed_answer == expected:
                        correct += 1
                metrics[f"self_recognition_{condition}_accuracy"] = round(
                    correct / len(self_recs), 4
                )
                metrics[f"self_recognition_{condition}_n"] = len(self_recs)

        # Mean relevancy (if computed)
        rel_recs = [r for r in self.records if r.mean_relevancy is not None]
        if rel_recs:
            metrics["mean_relevancy"] = round(
                sum(r.mean_relevancy for r in rel_recs) / len(rel_recs), 6
            )

        return metrics

    def save_results(self) -> str:
        """Final save — run() already flushes incrementally, this is the last write."""
        self._init_output_paths()
        self._flush_results()
        return str(self._jsonl_path)

    def _write_log(self, log_path: Path, base_name: str) -> None:
        """Write the detailed companion .txt log file."""
        sep = "=" * 70
        thin = "-" * 70
        lines = []

        elapsed = getattr(self, "_run_elapsed", 0.0)
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        lines += [
            sep,
            "PATCHSCOPE ACTIVATION INTERPRETATION RUN LOG",
            sep,
            f"Generated  : {datetime.now(timezone.utc).isoformat()}",
            f"Run name   : {base_name}",
            f"Total time : {elapsed_str}  ({elapsed:.1f}s)",
            f"Records    : {len(self.records)}",
            f"Errors     : {sum(1 for r in self.records if r.error)}",
            "",
        ]

        # ── Matrix dimensions
        ps = self.ps_config
        n_questions = len(self.questions)
        n_src = len(ps.get("source_personas", []))
        n_eval = len(ps.get("evaluator_personas", []))
        n_templates = len(ps.get("interpretation_templates", {}))
        conditions = ["real"]
        if ps.get("controls", {}).get("text_only_baseline"):
            conditions.append("text_only_baseline")
        if ps.get("controls", {}).get("shuffled_activation"):
            conditions.append("shuffled")
        n_conditions = len(conditions)

        if ps.get("layer_sweep", {}).get("enabled"):
            n_src_layers = len(ps["layer_sweep"]["source_layers"])
            n_inj_layers = len(ps["layer_sweep"]["injection_layers"])
        else:
            num_model_layers = self.backend.model.config.num_hidden_layers if self.backend else "?"
            n_src_layers = len(_resolve_layers(ps["extraction"]["layers"], num_model_layers)) if isinstance(num_model_layers, int) else "?"
            n_inj_layers = 1

        total = n_questions * n_src * n_src_layers * n_inj_layers * n_eval * n_templates * n_conditions
        lines += [sep, "MATRIX DIMENSIONS", thin]
        lines += [
            f"  questions           : {n_questions}",
            f"  source_personas     : {n_src}  {ps.get('source_personas', [])}",
            f"  extraction_layers   : {n_src_layers}",
            f"  injection_layers    : {n_inj_layers}",
            f"  evaluator_personas  : {n_eval}  {ps.get('evaluator_personas', [])}",
            f"  templates           : {n_templates}  {list(ps.get('interpretation_templates', {}).keys())}",
            f"  conditions          : {n_conditions}  {conditions}",
            f"  total cells         : {total}",
            "",
        ]

        # ── Run parameters
        lines += [sep, "RUN PARAMETERS (from experiments.yaml)", thin]
        cfg = self.config
        lines += [
            f"experiment_name   : {cfg.experiment_name}",
            f"model_name        : {cfg.model_name}",
            f"model_dtype       : {cfg.model_dtype or 'auto'}",
            f"task_file         : {cfg.task_file}",
            f"sample_size       : {cfg.sample_size}",
            f"seed              : {cfg.seed}",
            f"patchscope_config : {cfg.patchscope_config}",
            "",
        ]

        # ── Patchscope parameters
        lines += [sep, "PATCHSCOPE PARAMETERS (from patchscope.yaml)", thin]
        lines += [yaml.dump(self.ps_config, default_flow_style=False)]

        # ── Model info
        lines += [sep, "MODEL INFO", thin]
        if self.backend:
            model = self.backend.model
            lines += [
                f"num_hidden_layers : {model.config.num_hidden_layers}",
                f"hidden_size       : {model.config.hidden_size}",
                f"vocab_size        : {model.config.vocab_size}",
                f"architecture      : {model.config.architectures}",
                "",
            ]

        # ── Personas (verbatim system prompts)
        lines += [sep, "PERSONA SYSTEM PROMPTS (verbatim)", thin]
        all_persona_names = set(
            self.ps_config.get("source_personas", [])
            + self.ps_config.get("evaluator_personas", [])
        )
        for pname in sorted(all_persona_names):
            persona = self.all_personas.get(pname)
            lines.append(f"\n[{pname}]")
            if persona and persona.system_prompt:
                for sline in persona.system_prompt.strip().splitlines():
                    lines.append(f"  {sline}")
            else:
                lines.append("  (no system prompt)")
        lines.append("")

        # ── Source prompt examples
        if hasattr(self, '_sample_source_prompts') and self._sample_source_prompts:
            lines += [sep, "SOURCE PROMPT EXAMPLES (verbatim, including special tokens)", thin]
            for sp_name, sample in sorted(self._sample_source_prompts.items()):
                lines += [
                    f"\n--- {sp_name} ---",
                    f"  question_id   : {sample['question_id']}",
                    f"  option_order  : {sample.get('option_order', 'ABCD')}",
                    f"  source_answer : {sample['answer']} (shuffled: {sample.get('shuffled_answer', '?')})",
                    "",
                    "  ── FULL SOURCE PROMPT (exact tokens sent to model) ──",
                ]
                for pline in sample["prompt_text"].splitlines():
                    lines.append(f"  {pline}")
                lines.append("")
            lines.append("")

        # ── Sample reporter prompts (verbatim, one per template+condition)
        lines += [sep, "SAMPLE REPORTER PROMPTS AND RESPONSES (verbatim, including special tokens)", thin]
        for sample_key, sample in sorted(self._sample_prompts.items()):
            lines += [
                f"\n--- {sample_key} ---",
                f"  template          : {sample['template']}",
                f"  condition         : {sample['condition']}",
                f"  source_persona    : {sample['source_persona']}",
                f"  evaluator_persona : {sample['evaluator_persona']}",
                f"  source_layer      : {sample['source_layer']}",
                f"  injection_layer   : {sample['injection_layer']}",
                f"  question_id       : {sample['question_id']}",
                "",
                "  ── FULL PROMPT (exact tokens sent to model) ──",
            ]
            for pline in sample["interp_prompt_text"].splitlines():
                lines.append(f"  {pline}")
            lines += [
                "",
                "  ── MODEL RESPONSE (verbatim) ──",
            ]
            for rline in sample["generated_text"].splitlines():
                lines.append(f"  {rline}")
            lines.append("")
        lines.append("")

        # ── Source override results (Phase 3)
        override_records = [r for r in self.records if r.source_persona == "source_override"]
        if override_records:
            lines += [sep, "SOURCE OVERRIDE RESULTS (raw text, specific word extraction)", thin]
            current_qid = None
            for r in override_records:
                if r.question_id != current_qid:
                    current_qid = r.question_id
                    lines.append(f"\n  {r.question_id}: \"{r.question_text}\"")
                lines.append(
                    f"    L{r.source_layer:>2} → {r.injection_layer:>2}  "
                    f"{r.template_name:<20s}  '{r.generated_text.strip()[:70]}'"
                )
            lines.append("")

        # ── Results summary
        lines += [sep, "RESULTS SUMMARY", thin]
        metrics = self.evaluate()
        for k, v in metrics.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

        # ── Errors
        lines += [sep, f"ERRORS AND WARNINGS ({len(self._errors)} total)", thin]
        if self._errors:
            for err in self._errors:
                lines.append(f"  {err}")
        else:
            lines.append("  None")
        lines.append("")

        lines.append(sep)
        log_path.write_text("\n".join(lines), encoding="utf-8")
