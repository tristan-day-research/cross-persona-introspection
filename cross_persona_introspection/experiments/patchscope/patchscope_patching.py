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

from typing import Optional

import torch
import torch.nn.functional as F

from cross_persona_introspection.experiments.patchscope.patchscope_helpers import (
    _get_transformer_layers,
    resolve_extraction_token_index,
)


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
) -> list[float]:
    """Per-token relevancy: P(token | WITH patch) - P(token | WITHOUT).

    Uses KV-cache to avoid re-processing the entire sequence for every token.
    Two full forward passes for prefill (with/without patching), then two
    incremental single-token steps per generated token.

    Returns:
        List of floats, one per token (up to *max_tokens*).
    """
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
