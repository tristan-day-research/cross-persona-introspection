"""Low-level activation patching primitives.

Three functions are the mechanistic core of the Patchscope experiment:

1. **extract_activations_multi_layer** — run a source forward pass, hook
   multiple layers simultaneously, capture one hidden-state vector per layer.

2. **patch_and_decode** — the unified patching function.  Patches a source
   activation into an interpretation prompt at placeholder positions, then
   decodes the model's response.  Two decode modes:
   - ``"logits"``: single forward pass, read next-token probabilities over a
     constrained choice set (e.g. A/B/C/D).
   - ``"generate"``: auto-regressive generation, decode free-form text.

3. **compute_relevancy_scores** — SelfIE-style metric: for each generated
   token, how much did the patched activation change its probability?  Uses
   KV-cache so each token is one incremental step, not a full re-process.

All functions operate on raw PyTorch models / tokenizers and are stateless.
The only coupling to the rest of the codebase is ``_get_transformer_layers``
(architecture adapter in patchscope_helpers).


How hooks work
--------------
PyTorch ``register_forward_hook(fn)`` attaches a callback to a module (here,
a transformer layer).  Every time that module's ``forward()`` runs, PyTorch
calls ``fn(module, input, output)`` immediately after.  The hook can read
*output* (to capture hidden states) or return a modified *output* (to patch
activations).  Hooks are removed by calling ``handle.remove()`` on the handle
returned by ``register_forward_hook``.

Multiple hooks can be registered on different layers at the same time — they
all fire during a single forward pass, which is how
``extract_activations_multi_layer`` captures every layer in one pass.


What "KV-cache on/off" means
-----------------------------
Transformer attention computes Key and Value projections for every token at
every layer.  With ``use_cache=True``, the model saves these K/V tensors after
each forward call.  On the next call you only feed the *new* token(s) and pass
``past_key_values`` back in — the model skips recomputing K/V for all previous
tokens and just appends the new ones.

**Prefill** = the first forward call that processes the entire prompt (many
tokens, no cache yet).  **Decode steps** = subsequent calls that each process
one new token, reusing the cached K/V from all prior tokens.

This matters for patching: the hook modifies hidden states at placeholder
positions during prefill.  Those modifications flow into the K/V cache.  During
decode steps the placeholder positions aren't re-processed — the model just
reads their cached K/V.  So the patch's effect persists through the cache
without the hook needing to fire again.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F

from cross_persona_introspection.experiments.patchscope.patchscope_helpers import (
    _get_transformer_layers,
    resolve_extraction_token_index,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Extract (multi-layer, single forward pass)
# ---------------------------------------------------------------------------

def extract_activations_multi_layer(
    model,
    tokenizer,
    device,
    messages: list[dict],
    layer_indices: list[int],
    token_position: str | int = "last",
    raw_text: Optional[str] = None,
    boundary_marker: Optional[str] = None,
) -> dict[int, torch.Tensor]:
    """Run ONE forward pass and capture hidden states at multiple layers.

    Registers a hook on every requested layer simultaneously, runs one
    forward pass, then removes all hooks.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        device: Target device for input tensors.
        messages: Chat-formatted source prompt.  Ignored when *raw_text* is set.
        layer_indices: Transformer layers to hook (0-indexed, post-embedding).
        token_position: ``"last"``, ``"last_before_assistant"``, or an integer index
            (see ``resolve_extraction_token_index`` in patchscope_helpers).
        raw_text: If given, used verbatim as model input (skips chat template).
        boundary_marker: Used when *token_position* is ``last_before_assistant``;
            substring to search for in the templated string (default in resolver).

    Returns:
        Dict mapping layer index to a 1-D tensor of shape ``(d_model,)``.
    """
    if raw_text is not None:
        input_text = raw_text
    else:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    resolved_pos, _ = resolve_extraction_token_index(
        tokenizer, input_text, token_position, boundary_marker=boundary_marker,
    )
    input_ids = tokenizer.encode(
        input_text, return_tensors="pt", add_special_tokens=False
    ).to(device)

    captured: dict[int, torch.Tensor] = {}
    transformer_layers = _get_transformer_layers(model)

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden[0, resolved_pos, :].detach().clone()
        return hook_fn

    handles = []
    for idx in layer_indices:
        handles.append(transformer_layers[idx].register_forward_hook(make_hook(idx)))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in handles:
            h.remove()

    return captured


def extract_activations_during_decode(
    model,
    tokenizer,
    device,
    messages: list[dict],
    layer_indices: list[int],
    decode_steps: int,
    *,
    max_decode_steps: int = 64,
    temperature: float = 0.0,
    do_sample: bool = False,
    raw_text: Optional[str] = None,
    stop_tokens: Optional[list[str]] = None,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Prefill the source prompt, then run *decode_steps* single-token forwards; capture hidden[-1].

    Use this when ``extraction.readout: autoregressive`` so activations come from the model
    **after** it has begun generating (e.g. after the first answer letter), instead of from a
    fixed prefill index (often ``Answer:`` scaffolding).

    The hook runs on the **last** decode forward; ``hidden[batch, -1, :]`` is the representation
    at the position of the **last** generated token in this rollout (not the prefill prefix).

    Args:
        decode_steps: Must be >= 1. Greedy (``do_sample=False``) or sampled next token each step.
        max_decode_steps: ``decode_steps`` is clamped to this ceiling.
        stop_tokens: Optional list of token strings (e.g. ``["A", "B", "C", "D"]``). When
            provided, decoding stops at the first generated token whose stripped text matches
            one of these strings, and activations are captured at that step. ``decode_steps``
            is treated as ``max_decode_steps`` in this mode.

    Returns:
        (activations, meta) where activations maps layer index to ``(d_model,)`` tensor from the
        last decode forward only, and meta includes ``generated_token_ids`` and decoded strings
        for logging.
    """
    if decode_steps < 1:
        raise ValueError("extract_activations_during_decode requires decode_steps >= 1")

    if raw_text is not None:
        input_text = raw_text
    else:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    input_ids = tokenizer.encode(
        input_text, return_tensors="pt", add_special_tokens=False
    ).to(device)

    steps = min(int(decode_steps), int(max_decode_steps))
    if steps < decode_steps:
        logger.warning(
            "extraction.autoregressive.decode_steps=%s exceeds max_decode_steps=%s; using %s",
            decode_steps,
            max_decode_steps,
            steps,
        )

    transformer_layers = _get_transformer_layers(model)
    captured: dict[int, torch.Tensor] = {}
    generated_ids: list[int] = []

    # When stop_tokens is set, we register hooks on every step (last capture wins)
    # and break as soon as a generated token matches.
    _stop_set: frozenset[str] | None = None
    if stop_tokens:
        _stop_set = frozenset(t.strip() for t in stop_tokens)
    _use_stop = _stop_set is not None
    _stopped_early = False
    capture_step: int = steps  # updated below when stop_tokens fires

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :]

        for step in range(steps):
            if do_sample:
                if temperature and temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                else:
                    probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_id = logits.argmax(dim=-1)

            generated_ids.append(int(next_id.item()))
            next_token = next_id.view(1, 1).to(device=device, dtype=torch.long)

            # Register hooks when this is the designated capture step:
            # - stop_tokens mode: every step (overwrite captured each time)
            # - fixed mode: only the final step
            need_hooks = _use_stop or (step == steps - 1)
            if need_hooks:
                handles = []

                def make_hook(layer_idx):
                    def hook_fn(module, input, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        captured[layer_idx] = hidden[0, -1, :].detach().clone()

                    return hook_fn

                for idx in layer_indices:
                    handles.append(
                        transformer_layers[idx].register_forward_hook(make_hook(idx))
                    )
                try:
                    out = model(
                        next_token,
                        past_key_values=past,
                        use_cache=True,
                    )
                finally:
                    for h in handles:
                        h.remove()
            else:
                out = model(
                    next_token,
                    past_key_values=past,
                    use_cache=True,
                )

            past = out.past_key_values
            logits = out.logits[0, -1, :]

            # Check stop condition
            if _stop_set is not None:
                decoded = tokenizer.decode([generated_ids[-1]], skip_special_tokens=False).strip()
                if decoded in _stop_set:
                    capture_step = step + 1
                    _stopped_early = True
                    break

        if _use_stop and not _stopped_early:
            capture_step = steps
            logger.warning(
                "stop_tokens %s not found in %d decode steps; "
                "capturing at final step.",
                stop_tokens,
                steps,
            )

    decoded_pieces = [
        tokenizer.decode([tid], skip_special_tokens=False) for tid in generated_ids
    ]
    # Token whose forward pass produced the captured hidden[:, -1, :] (1-based capture_at_step).
    act_id: Optional[int] = None
    act_strip: Optional[str] = None
    if generated_ids and capture_step >= 1:
        idx = min(int(capture_step), len(generated_ids)) - 1
        if idx >= 0:
            act_id = generated_ids[idx]
            act_strip = tokenizer.decode([act_id], skip_special_tokens=False).strip()

    meta: dict[str, Any] = {
        "generated_token_ids": generated_ids,
        "generated_decode_concat": "".join(decoded_pieces),
        "generated_decode_pieces": decoded_pieces,
        "generated_token_repr": [repr(p) for p in decoded_pieces],
        "capture_at_step": capture_step,
        "stopped_on_token": _stopped_early,
        "n_generated_tokens": len(generated_ids),
        "activation_token_id": act_id,
        "activation_token_decoded_strip": act_strip,
    }
    return captured, meta


# ---------------------------------------------------------------------------
# 1b. Multi-position extraction variants
# ---------------------------------------------------------------------------

def extract_activations_multi_pos(
    model,
    tokenizer,
    device,
    messages: list[dict],
    layer_indices: list[int],
    token_positions: list[int],
    *,
    raw_text: Optional[str] = None,
) -> dict[int, dict[int, torch.Tensor]]:
    """Single forward pass — capture hidden states at *multiple* token positions.

    Args:
        token_positions: Absolute 0-based indices into the tokenised prompt.

    Returns:
        ``{pos: {layer: tensor}}`` for every requested position and layer.
    """
    if raw_text is not None:
        input_ids = tokenizer.encode(
            raw_text, return_tensors="pt", add_special_tokens=False
        ).to(device)
    else:
        # Use apply_chat_template with return_tensors so special tokens like
        # <|eot_id|>, <|start_header_id|> are encoded as single tokens.
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    seq_len = input_ids.shape[1]

    # Clamp to valid range
    valid_positions = [p for p in token_positions if 0 <= p < seq_len]
    if not valid_positions:
        logger.warning(
            "extract_activations_multi_pos: no valid positions in %s (seq_len=%d)",
            token_positions, seq_len,
        )
        return {}

    captured: dict[tuple[int, int], torch.Tensor] = {}  # (layer, pos) -> tensor
    transformer_layers = _get_transformer_layers(model)

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for pos in valid_positions:
                captured[(layer_idx, pos)] = hidden[0, pos, :].detach().clone()
        return hook_fn

    handles = []
    for idx in layer_indices:
        handles.append(transformer_layers[idx].register_forward_hook(make_hook(idx)))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in handles:
            h.remove()

    # Reshape to {pos: {layer: tensor}}
    result: dict[int, dict[int, torch.Tensor]] = {}
    for (layer_idx, pos), tensor in captured.items():
        result.setdefault(pos, {})[layer_idx] = tensor
    return result


def extract_activations_during_decode_multi_step(
    model,
    tokenizer,
    device,
    messages: list[dict],
    layer_indices: list[int],
    max_capture_steps: int,
    *,
    max_decode_steps: int = 64,
    temperature: float = 0.0,
    do_sample: bool = False,
    raw_text: Optional[str] = None,
    stop_tokens: Optional[list[str]] = None,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[str, Any]]:
    """Decode and capture hidden states at *each* of the first ``max_capture_steps`` generated tokens.

    Like :func:`extract_activations_during_decode` but stores per-step
    activations instead of keeping only the final capture.  If decoding
    produces fewer than *max_capture_steps* tokens (e.g. early stop), only
    the available steps are returned — no error.

    Returns:
        ``({step: {layer: tensor}}, meta)`` where *step* is 0-indexed.
        ``meta["per_step_tokens"]`` is a list of ``(token_id, decoded_text)``
        for each captured step.
    """
    if max_capture_steps < 1:
        raise ValueError("max_capture_steps must be >= 1")

    if raw_text is not None:
        input_ids = tokenizer.encode(
            raw_text, return_tensors="pt", add_special_tokens=False
        ).to(device)
    else:
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

    steps = min(int(max_decode_steps), 256)  # safety cap
    transformer_layers = _get_transformer_layers(model)

    # {step: {layer: tensor}}
    all_captured: dict[int, dict[int, torch.Tensor]] = {}
    generated_ids: list[int] = []
    per_step_tokens: list[tuple[int, str]] = []

    _stop_set: frozenset[str] | None = None
    if stop_tokens:
        _stop_set = frozenset(t.strip() for t in stop_tokens)
    _stopped_early = False

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :]

        for step in range(steps):
            if do_sample:
                if temperature and temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                else:
                    probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_id = logits.argmax(dim=-1)

            tid = int(next_id.item())
            generated_ids.append(tid)
            next_token = next_id.view(1, 1).to(device=device, dtype=torch.long)

            # Capture hooks for the first max_capture_steps steps
            need_hooks = step < max_capture_steps
            if need_hooks:
                handles = []
                step_captured: dict[int, torch.Tensor] = {}

                def make_hook(layer_idx, _captured=step_captured):
                    def hook_fn(module, input, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        _captured[layer_idx] = hidden[0, -1, :].detach().clone()
                    return hook_fn

                for idx in layer_indices:
                    handles.append(
                        transformer_layers[idx].register_forward_hook(make_hook(idx))
                    )
                try:
                    out = model(next_token, past_key_values=past, use_cache=True)
                finally:
                    for h in handles:
                        h.remove()

                all_captured[step] = step_captured
                decoded_text = tokenizer.decode([tid], skip_special_tokens=False).strip()
                per_step_tokens.append((tid, decoded_text))
            else:
                out = model(next_token, past_key_values=past, use_cache=True)

            past = out.past_key_values
            logits = out.logits[0, -1, :]

            # Check stop condition
            if _stop_set is not None:
                decoded = tokenizer.decode([generated_ids[-1]], skip_special_tokens=False).strip()
                if decoded in _stop_set:
                    _stopped_early = True
                    break

    decoded_pieces = [
        tokenizer.decode([tid], skip_special_tokens=False) for tid in generated_ids
    ]
    meta: dict[str, Any] = {
        "generated_token_ids": generated_ids,
        "generated_decode_concat": "".join(decoded_pieces),
        "generated_decode_pieces": decoded_pieces,
        "generated_token_repr": [repr(p) for p in decoded_pieces],
        "n_generated_tokens": len(generated_ids),
        "n_captured_steps": len(all_captured),
        "stopped_on_token": _stopped_early,
        "per_step_tokens": per_step_tokens,
    }
    return all_captured, meta


# ---------------------------------------------------------------------------
# 2. Patch and decode (unified: logits or generate)
# ---------------------------------------------------------------------------

def patch_and_decode(
    model,
    tokenizer,
    device,
    messages: list[dict],
    activation: Optional[torch.Tensor],
    injection_layer: int,
    placeholder_positions: list[int],
    mode: str = "replace",
    alpha: float = 1.0,
    raw_text: Optional[str] = None,
    # ── Decode strategy ───────────────────────────────────────────────
    decode_mode: str = "generate",
    # Generate-mode params (ignored when decode_mode="logits")
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    do_sample: bool = False,
    use_cache: bool = True,
    # Logits-mode params (ignored when decode_mode="generate")
    choice_token_ids: Optional[dict[str, int]] = None,
    save_logprobs: bool = False,
) -> dict:
    """Patch a source activation into an interpretation prompt and decode.

    This is the single entry point for all activation patching.  The operation
    has three stages:

    1. **Tokenize** the interpretation prompt.
    2. **Patch** the activation at placeholder positions via a forward hook
       (skipped when *activation* is None, for text-only baselines).
    3. **Decode** the model's response — either by reading constrained
       next-token logits (``decode_mode="logits"``) or by auto-regressive
       generation (``decode_mode="generate"``).

    Args:
        model / tokenizer / device: HuggingFace model triple.
        messages: Interpretation prompt (chat format).
        activation: Source activation tensor of shape ``(d_model,)``, or None
            for text-only baseline (runs without patching).
        injection_layer: Layer index for the patching hook.
        placeholder_positions: Token indices where placeholders sit in the
            tokenized prompt — these are the positions that get overwritten.
        mode: ``"replace"`` overwrites hidden states; ``"add"`` adds scaled.
        alpha: Scaling factor for ``"add"`` mode.
        raw_text: Verbatim input text (skips chat template when set).
        decode_mode: ``"logits"`` for constrained single-token decode,
            ``"generate"`` for free-form auto-regressive generation.
        max_new_tokens / temperature / do_sample: Generation params
            (only used when ``decode_mode="generate"``).
        use_cache: Whether to enable KV-cache in ``generate`` and forward passes.
            With cache on, decode steps only process the new token, so the hook
            sees sequence length 1 and placeholder indices do not match — the
            patch from prefill persists via cached K/V. With cache off, the model
            may recompute the full prefix each step; the hook runs whenever
            hidden length includes the placeholder indices, and we patch on
            those forwards (no one-shot guard).
        choice_token_ids: ``{label: token_id}`` for constrained decode
            (required when ``decode_mode="logits"``).
        save_logprobs: If True and logits mode, include full-vocab logprobs
            for choice tokens in the result.

    Returns:
        Dict always containing ``"generated_text"`` (str).
        In logits mode, also: ``"probs"``, ``"logits"``, ``"predicted"``,
        and optionally ``"logprobs"`` and ``"total_choice_prob"``.
    """
    # ── 1. Tokenize ───────────────────────────────────────────────────
    if raw_text is not None:
        input_text = raw_text
    else:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    input_ids = tokenizer.encode(
        input_text, return_tensors="pt", add_special_tokens=False
    ).to(device)

    # ── 2. Patch (register hook if activation provided) ───────────────
    transformer_layers = _get_transformer_layers(model)
    handle = None

    if activation is not None:
        if not placeholder_positions:
            raise ValueError(
                "patch_and_decode: activation is set but placeholder_positions is empty — "
                "injection would be a no-op. Ensure interpretation_templates include the "
                "literal {placeholder} token (YAML), and injection.placeholder_token encodes "
                "to token(s) that validate_placeholder_positions can find in the chat-templated "
                "string."
            )
        act = activation.to(device)

        # With use_cache=True during generate(), later decode steps often forward
        # only the new token (hidden length 1), so placeholder token indices
        # are out of range — we skip those forwards; the patched K/V from
        # prefill still carries the edit. With use_cache=False, forwards may
        # include the full growing prefix each time — we patch whenever indices
        # fit; no "prefill only" guard.
        use_one_shot_hook = decode_mode == "generate" and use_cache
        hook_done = [False]

        def injection_hook(module, input, output):
            if use_one_shot_hook and hook_done[0]:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            patched_any = False
            for pos in placeholder_positions:
                if pos < hidden.shape[1]:
                    patched_any = True
                    if mode == "replace":
                        hidden[0, pos, :] = act
                    elif mode == "add":
                        hidden[0, pos, :] = hidden[0, pos, :] + alpha * act
            if use_one_shot_hook and patched_any:
                hook_done[0] = True
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        handle = transformer_layers[injection_layer].register_forward_hook(injection_hook)

    # ── 3. Decode ─────────────────────────────────────────────────────
    try:
        with torch.no_grad():
            if decode_mode == "generate":
                result = _decode_generate(
                    model, tokenizer, input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    use_cache=use_cache,
                )
            elif decode_mode == "logits":
                if choice_token_ids is None:
                    raise ValueError("choice_token_ids is required for decode_mode='logits'")
                result = _decode_logits(
                    model, input_ids, choice_token_ids,
                    use_cache=use_cache,
                    save_logprobs=save_logprobs,
                )
            else:
                raise ValueError(f"Unknown decode_mode: {decode_mode!r}")
    finally:
        if handle is not None:
            handle.remove()

    return result


def _decode_generate(
    model, tokenizer, input_ids,
    max_new_tokens, temperature, do_sample, use_cache,
) -> dict:
    """Auto-regressive generation.  Returns {"generated_text": str}."""
    attention_mask = torch.ones_like(input_ids)
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else None,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=use_cache,
    )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return {"generated_text": tokenizer.decode(new_tokens, skip_special_tokens=True)}


def _decode_logits(
    model, input_ids, choice_token_ids,
    use_cache,
    save_logprobs,
) -> dict:
    """Single forward pass, constrained next-token probs.

    Returns dict with "generated_text", "probs", "logits", "predicted",
    and optionally "logprobs" and "total_choice_prob".
    """
    outputs = model(input_ids, use_cache=use_cache)
    next_logits = outputs.logits[0, -1, :]

    choice_logit_values = {
        choice: next_logits[tid].item()
        for choice, tid in choice_token_ids.items()
    }

    # Softmax restricted to the choice set
    logit_tensor = torch.tensor(list(choice_logit_values.values()))
    probs = F.softmax(logit_tensor, dim=0)
    choice_probs = {
        choice: probs[i].item()
        for i, choice in enumerate(choice_logit_values.keys())
    }

    predicted = max(choice_probs, key=choice_probs.get)

    result = {
        "generated_text": f"[logits] {predicted}",
        "probs": choice_probs,
        "logits": choice_logit_values,
        "predicted": predicted,
    }

    if save_logprobs:
        full_logprobs = F.log_softmax(next_logits, dim=0)
        full_probs = F.softmax(next_logits, dim=0)
        result["logprobs"] = {
            choice: full_logprobs[tid].item()
            for choice, tid in choice_token_ids.items()
        }
        result["total_choice_prob"] = sum(
            full_probs[tid].item() for tid in choice_token_ids.values()
        )

    return result


# ---------------------------------------------------------------------------
# 3. SelfIE relevancy scores (KV-cached)
# ---------------------------------------------------------------------------

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
    raw_text: Optional[str] = None,
) -> list[float]:
    """Per-token relevancy: P(token | WITH patch) - P(token | WITHOUT).

    Uses KV-cache to avoid re-processing the entire sequence for every token.
    Two full forward passes for prefill (with/without patching), then two
    incremental single-token steps per generated token.

    If *raw_text* is set (identity-style plain prompts), it is used as the
    prompt string; otherwise *messages* are rendered with ``apply_chat_template``.

    Returns:
        List of floats, one per token (up to *max_tokens*).
    """
    if raw_text is not None:
        input_text = raw_text
    else:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    base_ids = tokenizer.encode(
        input_text, return_tensors="pt", add_special_tokens=False
    ).to(device)

    transformer_layers = _get_transformer_layers(model)
    act = activation.to(device)

    n_tokens = min(len(generated_token_ids), max_tokens)
    if n_tokens == 0:
        return []

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

    # Prefill WITH patch
    handle = transformer_layers[injection_layer].register_forward_hook(injection_hook)
    try:
        with torch.no_grad():
            out_with = model(base_ids, use_cache=True)
    finally:
        handle.remove()
    past_with = out_with.past_key_values

    # Prefill WITHOUT patch
    with torch.no_grad():
        out_without = model(base_ids, use_cache=True)
    past_without = out_without.past_key_values

    scores = []

    for i in range(n_tokens):
        target_token_id = generated_token_ids[i]

        if i == 0:
            logits_with = out_with.logits[0, -1, :]
            logits_without = out_without.logits[0, -1, :]
        else:
            prev_token = torch.tensor([[generated_token_ids[i - 1]]], device=device)

            with torch.no_grad():
                step_with = model(prev_token, past_key_values=past_with, use_cache=True)
            past_with = step_with.past_key_values
            logits_with = step_with.logits[0, -1, :]

            with torch.no_grad():
                step_without = model(prev_token, past_key_values=past_without, use_cache=True)
            past_without = step_without.past_key_values
            logits_without = step_without.logits[0, -1, :]

        prob_with = F.softmax(logits_with, dim=-1)[target_token_id].item()
        prob_without = F.softmax(logits_without, dim=-1)[target_token_id].item()

        scores.append(prob_with - prob_without)

    return scores
