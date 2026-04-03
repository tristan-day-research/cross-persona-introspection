"""Patchscope helper utilities.

Config loading, prompt formatting, answer parsing, model introspection,
placeholder resolution, and interpretation-prompt construction used by
PatchscopeExperiment and patchscope_source_overrides.
"""

import hashlib
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
    """Extract a short name for result files (e.g. ``ps_l8b_<timestamp>_...``).

    Llama 8B family checkpoints (3.x / Instruct / etc.) map to ``l8b``; others get a
    compact slug from the final path segment.
    """
    lower = model_name.lower()
    if "llama" in lower and "8b" in lower.replace(" ", ""):
        return "l8b"
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


def _resolve_layer_config(
    ps_config: dict, extraction_cfg: dict, injection_cfg: dict, num_layers: int,
) -> tuple[list[int], list[int], dict[int, list[int]] | None]:
    """Parse layer_pairs / layer_sweep / simple layer config from patchscope YAML.

    Three modes (checked in priority order):
      1. ``layer_pairs`` — explicit (source, injection) pairs.
      2. ``layer_sweep``  — Cartesian product of source × injection layer lists.
      3. Fall back to ``extraction.layers`` + ``injection.layer``.

    Returns:
        ``(source_layers, injection_layers, pair_map_or_None)``
        *pair_map* is a dict mapping each source layer to its paired injection
        layers (only set in mode 1; ``None`` otherwise).
    """
    explicit_pairs = ps_config.get("layer_pairs")
    if explicit_pairs:
        source_layers = sorted(set(int(p[0]) for p in explicit_pairs))
        injection_layers = sorted(set(int(p[1]) for p in explicit_pairs))
        pair_map: dict[int, list[int]] = {}
        for p in explicit_pairs:
            pair_map.setdefault(int(p[0]), []).append(int(p[1]))
        logger.info(f"Using explicit layer_pairs: {explicit_pairs}")
        return source_layers, injection_layers, pair_map

    if ps_config.get("layer_sweep", {}).get("enabled", False):
        sweep = ps_config["layer_sweep"]
        source_layers = [int(x) for x in sweep["source_layers"]]
        injection_layers = [int(x) for x in sweep["injection_layers"]]
        return source_layers, injection_layers, None

    source_layers = _resolve_layers(extraction_cfg["layers"], num_layers)
    injection_layers = [int(injection_cfg["layer"])]
    return source_layers, injection_layers, None


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


def _get_placeholder_token_id(tokenizer, configured_token: str) -> int:
    """Encode the configured placeholder token to a single token ID.

    Args:
        tokenizer: HuggingFace tokenizer.
        configured_token: Literal token string from config (e.g. "___", "?").
            Special values "<unk>" and "<pad>" resolve to the tokenizer's
            built-in IDs.

    Raises:
        ValueError: If the token encodes to zero or multiple token IDs.
    """
    if configured_token == "<unk>":
        if tokenizer.unk_token_id is None:
            raise ValueError("Config uses <unk> but tokenizer has no unk_token_id")
        return tokenizer.unk_token_id
    if configured_token == "<pad>":
        if tokenizer.pad_token_id is None:
            raise ValueError("Config uses <pad> but tokenizer has no pad_token_id")
        return tokenizer.pad_token_id

    ids = tokenizer.encode(configured_token, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]

    raise ValueError(
        f"Placeholder token '{configured_token}' encodes to {len(ids)} token(s) "
        f"({ids}); it must encode to exactly 1 token. "
        f"Fix injection.placeholder_token in your patchscope config."
    )


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
    # Stable across processes (unlike built-in hash randomization).
    mix = hashlib.sha256(f"{seed}\0{qid}\0{persona}".encode()).digest()
    rng = random.Random(int.from_bytes(mix[:8], "big"))
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


def _options_formatted_abcd(options: dict) -> str:
    """Options as lines A)–D) in fixed order (only labels present in *options*)."""
    lines = []
    for letter in ["A", "B", "C", "D"]:
        if letter in options:
            lines.append(f"{letter}) {options[letter]}")
    return "\n".join(lines)


_IDEOLOGY_PERSONAS = frozenset({"persona_conservative", "persona_progressive"})


def reporter_sample_opposing_qualifies(
    source_persona: str,
    reporter_persona: str,
    policy: str,
) -> bool:
    """Whether a matrix cell should fill the .txt log *oppose* sample bucket.

    *policy*:
      - ``any_mismatch`` — any cell with source != reporter.
      - ``cross_ideology`` — only conservative vs progressive (both names in the pair).
    """
    if source_persona == reporter_persona:
        return False
    if policy == "any_mismatch":
        return True
    if policy == "cross_ideology":
        return (
            source_persona in _IDEOLOGY_PERSONAS
            and reporter_persona in _IDEOLOGY_PERSONAS
        )
    raise ValueError(
        f"Unknown reporting.opposing_sample_policy {policy!r}; "
        "use 'cross_ideology' or 'any_mismatch'."
    )


def format_source_pass_user_message(
    question: dict,
    user_message_template: str,
) -> str:
    """Build the Phase 1 source user message using ``source_pass.user_message_template`` from config.

    Placeholders: ``{question_text}``, ``{options_formatted}`` (A–D lines).
    """
    template = user_message_template.rstrip(" \t")  # strip trailing spaces/tabs but keep newlines
    if not template.strip():
        raise ValueError(
            "source_pass.user_message_template is empty. Set it in patchscope.yaml."
        )
    options_formatted = _options_formatted_abcd(question.get("options", {}))
    return template.format(
        question_text=question.get("question_text", ""),
        options_formatted=options_formatted,
    )


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


def normalize_prompt_styles(ps_config: dict) -> list[str]:
    """Expand ``prompt_style`` to a non-empty list of variant keys.

    YAML may set ``prompt_style`` to a single string (backward compatible) or
    to a list of strings. Each entry must match a key on every enabled
    ``interpretation_templates.*`` block (e.g. ``patchscopes``, ``selfie``,
    or custom keys like ``what_is_the_definition_of``).
    """
    raw = ps_config.get("prompt_style", "patchscopes")
    if isinstance(raw, list):
        out = [str(s).strip() for s in raw if str(s).strip()]
        if not out:
            raise ValueError("prompt_style is a list but contains no non-empty strings")
        return out
    s = str(raw).strip()
    if not s:
        raise ValueError("prompt_style is empty")
    return [s]


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
    *,
    use_chat_template: bool = True,
) -> tuple[str, list[dict], list[int]]:
    """Build an interpretation prompt and locate placeholder token positions.

    This is the shared prompt-construction logic used by both the main
    experiment matrix (Phase 2) and source overrides (Phase 3).

    Uses a null-byte sentinel to find the placeholder in the user string, then
    maps that character span to token indices.

    Args:
        tokenizer: HuggingFace tokenizer.
        tmpl_cfg: Single template config dict (has keys like ``"patchscopes"``,
            ``"selfie"``, ``"identity"``).
        prompt_style: One of ``"patchscopes"``, ``"selfie"``, ``"identity"``.
        base_prompt: Interpretation base prompt prepended in selfie/patchscopes mode.
        placeholder_token: Decoded placeholder token string (e.g. ``"?"``).
        num_placeholders: How many placeholder tokens to insert.
        question: Question dict with ``question_text`` and ``options``.
        reporter_system_prompt: Reporter persona system prompt; whitespace-only is treated as absent.
            Templates may include ``{persona_prompt}`` (same text); if present, it is substituted
            and the persona is not also prepended (plain mode) or duplicated as a separate system
            message (chat template mode).
        options_override: If given, use these options instead of ``question["options"]``.
        use_chat_template: If True (default), non-identity styles are wrapped with
            ``apply_chat_template`` and the reporter system prompt goes in the system
            role. ``identity`` style always skips the chat template regardless of this
            flag. If False, ALL styles skip the chat template: the reporter system
            prompt (if any) is prepended as plain text followed by a blank line, keeping
            the model in raw continuation mode.

    Returns:
        ``(interp_text, interp_messages, placeholder_positions)`` where
        *interp_text* is the string passed to the model: plain text when
        ``prompt_style == "identity"`` or ``use_chat_template=False``; output of
        ``apply_chat_template`` otherwise.
        *interp_messages* is the chat message list (system + user when configured).
        *placeholder_positions* are token indices into *interp_text* as encoded by
        the tokenizer with ``add_special_tokens=False``.
    """
    _SENTINEL = "\x00PH\x00"

    reporter_system = (reporter_system_prompt or "").strip()

    prompt_template = tmpl_cfg.get(prompt_style)
    if not prompt_template:
        raise ValueError(
            f"No prompt template for style '{prompt_style}' in template config. "
            f"Available keys: {[k for k in tmpl_cfg if k != 'decode_mode']}"
        )

    placeholder_str = (placeholder_token + " ") * num_placeholders
    placeholder_str = placeholder_str.strip()

    opts = options_override if options_override is not None else question.get("options", {})
    options_str = ", ".join(
        f"{k}) {opts[k]}" for k in ["A", "B", "C", "D"] if k in opts
    )
    options_formatted_str = _options_formatted_abcd(opts)

    raw_template = prompt_template.strip()
    has_persona_slot = "{persona_prompt}" in raw_template

    user_content = raw_template
    user_content = user_content.replace("{persona_prompt}", reporter_system)
    user_content = user_content.replace("{placeholder}", _SENTINEL)
    user_content = user_content.replace("{question_text}", question.get("question_text", ""))
    user_content = user_content.replace("{options_formatted}", options_formatted_str)
    user_content = user_content.replace("{options}", options_str)
    user_content = user_content.replace("{statement}", question.get("question_text", ""))

    if prompt_style == "identity" or not use_chat_template:
        # Raw continuation mode: no chat template, no special tokens.
        # Legacy: prepend reporter system when the template has no {persona_prompt} slot.
        if reporter_system and not has_persona_slot:
            user_content = reporter_system + "\n\n" + user_content
        _ph_char_pos = user_content.find(_SENTINEL)
        user_content = user_content.replace(_SENTINEL, placeholder_str)
        interp_text = user_content
        interp_messages = [{"role": "user", "content": user_content}]
        interp_ids = tokenizer.encode(interp_text, return_tensors="pt", add_special_tokens=False)
    else:
        if base_prompt.strip():
            user_content = base_prompt.strip() + "\n\n" + user_content

        sentinel_messages = []
        if reporter_system and not has_persona_slot:
            sentinel_messages.append({"role": "system", "content": reporter_system})
        sentinel_messages.append({"role": "user", "content": user_content})

        sentinel_text = tokenizer.apply_chat_template(
            sentinel_messages, tokenize=False, add_generation_prompt=True
        )
        _ph_char_pos = sentinel_text.find(_SENTINEL)

        user_content = user_content.replace(_SENTINEL, placeholder_str)
        interp_messages = []
        if reporter_system and not has_persona_slot:
            interp_messages.append({"role": "system", "content": reporter_system})
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


# ── Position validation ────────────────────────────────────────────────


def validate_placeholder_positions(
    tokenizer,
    interp_text: str,
    placeholder_positions: list[int],
    placeholder_token: str,
    tmpl_name: str = "",
    raise_on_error: bool = True,
) -> bool:
    """Verify that placeholder_positions actually point to placeholder tokens.

    Decodes the token at each position and checks that it contains the
    expected placeholder token.  Logs an annotated token map showing exactly
    where the placeholders are.

    Call this once per template during the first cell to catch bugs early.

    Args:
        tokenizer: HuggingFace tokenizer.
        interp_text: The fully rendered interpretation prompt string.
        placeholder_positions: Token indices where injection will happen.
        placeholder_token: The expected placeholder string (e.g. "?").
        tmpl_name: Template name for log messages.
        raise_on_error: If True, raise ValueError on mismatch.

    Returns:
        True if all positions are valid.
    """
    token_ids = tokenizer.encode(interp_text, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Build annotated token map
    position_set = set(placeholder_positions)
    annotated = []
    for i, tok in enumerate(tokens):
        marker = " <<<PATCH" if i in position_set else ""
        annotated.append(f"  [{i:3d}] {repr(tok)}{marker}")

    # Only log full map at DEBUG; log summary at INFO
    token_map_str = "\n".join(annotated)
    logger.debug(f"Token map for template={tmpl_name}:\n{token_map_str}")

    # Validate each position
    errors = []
    for pos in placeholder_positions:
        if pos >= len(tokens):
            errors.append(f"position {pos} is out of range (prompt has {len(tokens)} tokens)")
        elif placeholder_token not in tokens[pos].strip():
            errors.append(
                f"position {pos} is {repr(tokens[pos])}, expected placeholder {repr(placeholder_token)}"
            )

    if errors:
        context_lines = []
        for pos in placeholder_positions:
            start = max(0, pos - 2)
            end = min(len(tokens), pos + 3)
            context_lines.append(
                "  ... " + " ".join(
                    f"[{repr(tokens[i])}]" if i != pos else f">>>{repr(tokens[i])}<<<"
                    for i in range(start, end)
                ) + " ..."
            )
        error_msg = (
            f"Placeholder validation FAILED for template={tmpl_name}:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\nContext:\n" + "\n".join(context_lines)
        )
        if raise_on_error:
            raise ValueError(error_msg)
        logger.error(error_msg)
        return False

    if not placeholder_positions:
        msg = f"No placeholder positions for template={tmpl_name} — injection will be a no-op"
        if raise_on_error:
            raise ValueError(msg)
        logger.error(msg)
        return False

    # Log a concise summary
    pos_tokens = [f"[{p}]={repr(tokens[p].strip())}" for p in placeholder_positions]
    logger.info(f"Placeholder validation OK for template={tmpl_name}: {', '.join(pos_tokens)}")
    return True


def resolve_extraction_token_index(
    tokenizer,
    source_text: str,
    token_position: str | int,
    *,
    boundary_marker: str | None = None,
) -> tuple[int, dict]:
    """Map ``extraction.token_position`` to a concrete 0-based token index.

    Supported *token_position* values:

    - ``"last"`` — final token of *source_text* (after chat template + generation prompt).
    - ``"last_before_assistant"`` — last token whose span ends **before** the first occurrence
      of *boundary_marker* (default Llama 3--style ``"<|eot_id|><|start_header_id|>assistant"``).  Use this
      to read out hidden state at the end of the **user** turn instead of after assistant
      scaffolding (often ``\\n\\n`` as the literal ``last`` token).
    - A non-negative ``int`` — explicit index.

    Returns:
        ``(pos, meta)`` where *meta* may include ``boundary_marker``, ``boundary_char_index``.

    Raises:
        ValueError: unknown spec, marker not found, or tokenizer cannot resolve offsets.
    """
    token_ids = tokenizer.encode(source_text, add_special_tokens=False)
    n_tok = len(token_ids)
    meta: dict = {}

    if isinstance(token_position, int):
        pos = token_position
        if not (0 <= pos < n_tok):
            raise ValueError(f"extraction token index {pos} out of range for {n_tok} tokens")
        return pos, meta

    if not isinstance(token_position, str):
        raise ValueError(f"extraction.token_position must be int or str, got {type(token_position)}")

    spec = token_position.strip().lower().replace("-", "_")

    if spec == "last":
        return n_tok - 1, meta

    if spec in ("last_before_assistant", "before_assistant"):
        marker = boundary_marker or "<|eot_id|><|start_header_id|>assistant"
        meta["boundary_marker"] = marker
        idx_char = source_text.find(marker)
        if idx_char < 0:
            raise ValueError(
                f"extraction: boundary marker {marker!r} not found in templated source text. "
                "Set extraction.assistant_boundary_marker to match your chat template, "
                "or use token_position 'last' / an integer index."
            )
        meta["boundary_char_index"] = idx_char
        pos = _last_token_index_ending_before_char(tokenizer, source_text, idx_char)
        meta["resolved_via"] = "last_before_assistant"
        return pos, meta

    if spec.isdigit():
        pos = int(spec)
        if not (0 <= pos < n_tok):
            raise ValueError(f"extraction token index {pos} out of range for {n_tok} tokens")
        return pos, meta

    if spec in ("during_generation", "while_generating", "on_generation", "generated"):
        raise ValueError(
            f"extraction.token_position {token_position!r} selects activations from tokens the model "
            "generates after the prompt (post-prefill decode), not a prefill index. "
            "Phase-1 uses extract_activations_during_decode when this is set — ensure "
            "extraction.autoregressive.decode_steps >= 1 and that the experiment enables "
            "generation-time capture (readout or token_position). "
            "This error means resolve_extraction_token_index was called for a prefill-only path; "
            "check configuration."
        )

    raise ValueError(
        f"Unknown extraction.token_position {token_position!r}. "
        "Use 'last', 'last_before_assistant', a non-negative integer, or for generated-token "
        "activations use readout: during_generation (or token_position: during_generation)."
    )


_GEN_READOUT_MODES = frozenset({"autoregressive", "during_generation", "while_generating"})
_GEN_TOKEN_POSITION_SPECS = frozenset(
    {"during_generation", "while_generating", "on_generation", "generated"}
)


def _norm_extraction_key(s: str) -> str:
    return s.strip().lower().replace("-", "_")


def extraction_uses_generation_time_capture(extraction_cfg: dict) -> bool:
    """True when Phase-1 activations should come from post-prefill decode (generated tokens)."""
    readout = _norm_extraction_key(str(extraction_cfg.get("readout") or "prefill"))
    if readout == "both":
        return False  # "both" mode handled separately
    if readout in _GEN_READOUT_MODES:
        return True
    tp = extraction_cfg.get("token_position", "last")
    if isinstance(tp, str) and _norm_extraction_key(tp) in _GEN_TOKEN_POSITION_SPECS:
        return True
    return False


def extraction_uses_both_modes(extraction_cfg: dict) -> bool:
    """True when Phase-1 should extract from both prefill position AND during generation."""
    readout = _norm_extraction_key(str(extraction_cfg.get("readout") or "prefill"))
    return readout == "both"


def _last_token_index_ending_before_char(
    tokenizer,
    text: str,
    char_boundary: int,
) -> int:
    """Largest token index i such that token i's char span ends at or before *char_boundary*."""
    try:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, ValueError, NotImplementedError):
        enc = None

    if enc is not None and enc.get("offset_mapping"):
        best = -1
        for i, (_s, e) in enumerate(enc["offset_mapping"]):
            if e <= char_boundary:
                best = i
        if best >= 0:
            return best

    # Slow-tokenizer fallback: approximate by cumulative decoded length
    ids = tokenizer.encode(text, add_special_tokens=False)
    cumulative = 0
    best = -1
    for i, tid in enumerate(ids):
        piece = tokenizer.decode([tid])
        end = cumulative + len(piece)
        if end <= char_boundary:
            best = i
        cumulative = end
    if best < 0:
        raise ValueError(
            "Could not map extraction boundary to a token index (no token ends before the marker). "
            "Try a Fast tokenizer or set an explicit integer token_position."
        )
    return best


def describe_source_extraction_site(
    tokenizer,
    source_text: str,
    token_position: str | int,
    *,
    boundary_marker: str | None = None,
) -> dict:
    """Return structured info about where ``extract_activations_multi_layer`` reads hidden states.

    Uses :func:`resolve_extraction_token_index` so ``last_before_assistant`` matches extraction.

    Returns:
        Dict with keys ``token_position_spec``, ``token_index``, ``n_tokens``, ``token_id``,
        ``token_decoded_repr``, optional ``boundary_marker`` / ``boundary_char_index``, and
        ``error`` if resolution fails.
    """
    try:
        pos, meta = resolve_extraction_token_index(
            tokenizer, source_text, token_position, boundary_marker=boundary_marker,
        )
    except ValueError as e:
        return {
            "token_position_spec": token_position,
            "error": str(e),
            "n_tokens": len(tokenizer.encode(source_text, add_special_tokens=False)),
        }

    token_ids = tokenizer.encode(source_text, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    base = {
        "token_position_spec": token_position,
        "token_index": pos,
        "n_tokens": len(tokens),
        **meta,
    }
    if not (0 <= pos < len(tokens)):
        return {**base, "error": f"index {pos} out of range for {len(tokens)} tokens"}

    return {
        **base,
        "token_id": int(token_ids[pos]),
        "token_decoded_repr": repr(tokens[pos]),
        "token_decoded_strip": tokens[pos].strip(),
    }


def validate_extraction_position(
    tokenizer,
    source_text: str,
    token_position: str | int,
    tmpl_name: str = "",
    *,
    boundary_marker: str | None = None,
) -> None:
    """Log which token the extraction hook will capture.

    Resolves ``last_before_assistant`` the same way as extraction.  Never raises.
    """
    try:
        pos, _meta = resolve_extraction_token_index(
            tokenizer, source_text, token_position, boundary_marker=boundary_marker,
        )
    except ValueError as e:
        logger.warning(f"Extraction position for {tmpl_name}: {e}")
        return

    token_ids = tokenizer.encode(source_text, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    if 0 <= pos < len(tokens):
        # Show a window around the extraction position
        start = max(0, pos - 3)
        end = min(len(tokens), pos + 4)
        context = " ".join(
            f">>>{repr(tokens[i])}<<<" if i == pos else repr(tokens[i])
            for i in range(start, end)
        )
        logger.info(
            f"Extraction position for {tmpl_name}: "
            f"token_position={token_position!r} -> [{pos}]={repr(tokens[pos].strip())}  "
            f"context: ...{context}..."
        )
    else:
        logger.warning(
            f"Extraction position {token_position!r} -> index {pos} is out of range "
            f"(prompt has {len(tokens)} tokens)"
        )
