"""Patchscope helper utilities.

Config loading, prompt formatting, answer parsing, model introspection,
placeholder resolution, and interpretation-prompt construction used by
PatchscopeExperiment and patchscope_source_overrides.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent.parent
CONFIG_DIR = ROOT_DIR / "config"
TASKS_DIR = ROOT_DIR / "tasks"


# ── Config & data loading ─────────────────────────────────────────────────


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


# ── Layer & model utilities ───────────────────────────────────────────────


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


# ── Placeholder & token utilities ─────────────────────────────────────────


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


# ── Prompt formatting ─────────────────────────────────────────────────────


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


# ── Constrained answer parsing ────────────────────────────────────────────

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


# ── Template → valid choices mapping ──────────────────────────────────────

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


# ── Interpretation prompt construction ─────────────────────────────────


def build_interpretation_prompt(
    tokenizer,
    tmpl_cfg: dict,
    prompt_style: str,
    base_prompt: str,
    placeholder_token: str,
    num_placeholders: int,
    question: dict,
    reporter_system_prompt: str | None,
    options_override: dict | None = None,
) -> tuple[str, list[dict], list[int]]:
    """Build an interpretation prompt and locate placeholder token positions.

    This is the shared prompt-construction logic used by both the main
    experiment matrix (Phase 2) and source overrides (Phase 3).

    Uses a null-byte sentinel to track placeholder character position through
    chat-template wrapping, then maps that position back to token indices.

    Args:
        tokenizer: HuggingFace tokenizer.
        tmpl_cfg: Single template config dict (has keys like ``"patchscopes"``,
            ``"selfie"``, ``"identity"``).
        prompt_style: One of ``"patchscopes"``, ``"selfie"``, ``"identity"``.
        base_prompt: Interpretation base prompt prepended in selfie/patchscopes mode.
        placeholder_token: Decoded placeholder token string (e.g. ``"?"``).
        num_placeholders: How many placeholder tokens to insert.
        question: Question dict with ``question_text`` and ``options``.
        reporter_system_prompt: Reporter persona system prompt (may be None/empty).
        options_override: If given, use these options instead of ``question["options"]``.

    Returns:
        ``(interp_text, interp_messages, placeholder_positions)`` where
        *interp_text* is the fully rendered string (with special tokens for
        non-identity styles), *interp_messages* is the chat message list, and
        *placeholder_positions* is a list of token indices.
    """
    _SENTINEL = "\x00PH\x00"

    prompt_template = tmpl_cfg.get(prompt_style)
    if not prompt_template:
        prompt_template = tmpl_cfg.get("patchscopes") or tmpl_cfg.get("selfie", "")

    placeholder_str = (placeholder_token + " ") * num_placeholders
    placeholder_str = placeholder_str.strip()

    opts = options_override if options_override is not None else question.get("options", {})
    options_str = ", ".join(f"{k}) {v}" for k, v in opts.items())
    options_formatted_str = _format_options_lines(opts)

    user_content = prompt_template.strip()
    user_content = user_content.replace("{placeholder}", _SENTINEL)
    user_content = user_content.replace("{question_text}", question.get("question_text", ""))
    user_content = user_content.replace("{options_formatted}", options_formatted_str)
    user_content = user_content.replace("{options}", options_str)
    user_content = user_content.replace("{statement}", question.get("question_text", ""))

    if prompt_style == "identity":
        _ph_char_pos = user_content.find(_SENTINEL)
        user_content = user_content.replace(_SENTINEL, placeholder_str)
        interp_text = user_content
        interp_messages = [{"role": "user", "content": user_content}]
        interp_ids = tokenizer.encode(interp_text, return_tensors="pt", add_special_tokens=False)
    else:
        if base_prompt.strip():
            user_content = base_prompt.strip() + "\n\n" + user_content

        # Build with sentinel first to find char position
        sentinel_messages = []
        if reporter_system_prompt:
            sentinel_messages.append({"role": "system", "content": reporter_system_prompt})
        sentinel_messages.append({"role": "user", "content": user_content})

        sentinel_text = tokenizer.apply_chat_template(
            sentinel_messages, tokenize=False, add_generation_prompt=True
        )
        _ph_char_pos = sentinel_text.find(_SENTINEL)

        # Replace sentinel with actual placeholder
        user_content = user_content.replace(_SENTINEL, placeholder_str)
        interp_messages = []
        if reporter_system_prompt:
            interp_messages.append({"role": "system", "content": reporter_system_prompt})
        interp_messages.append({"role": "user", "content": user_content})

        interp_text = tokenizer.apply_chat_template(
            interp_messages, tokenize=False, add_generation_prompt=True
        )
        interp_ids = tokenizer.encode(interp_text, return_tensors="pt", add_special_tokens=False)

    # Map char offset → token indices
    placeholder_positions = []
    if _ph_char_pos >= 0:
        cumulative = 0
        for ti, tid in enumerate(interp_ids[0].tolist()):
            tok_str = tokenizer.decode([tid])
            tok_start = cumulative
            cumulative += len(tok_str)
            if (tok_start < _ph_char_pos + len(placeholder_str)
                    and cumulative >= _ph_char_pos
                    and placeholder_token in tok_str.strip()):
                placeholder_positions.append(ti)

    if not placeholder_positions:
        logger.warning("No placeholder positions found in prompt! Injection will be skipped.")

    return interp_text, interp_messages, placeholder_positions
