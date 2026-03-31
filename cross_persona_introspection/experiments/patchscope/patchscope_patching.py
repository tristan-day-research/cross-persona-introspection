"""Low-level activation patching primitives.

These four functions are the mechanistic core of the Patchscope experiment:

1. extract_activation  — run a source forward pass, hook a layer, capture one
                         hidden-state vector.
2. inject_and_generate — run an interpretation forward pass with a hooked layer
                         that overwrites (or adds to) placeholder positions,
                         then auto-regressively generate text.
3. inject_and_extract_logits — same injection, but instead of generating, read
                               next-token logits over a constrained choice set.
4. compute_relevancy_scores  — SelfIE-style metric: for each generated token,
                               how much did the injected activation change its
                               probability?

All functions operate on raw PyTorch models / tokenizers and are stateless —
they don't depend on the experiment class or config.  The only coupling to
the rest of the codebase is `_get_transformer_layers` (architecture adapter).
"""

from typing import Optional

import torch
import torch.nn.functional as F

from cross_persona_introspection.experiments.patchscope.patchscope_helpers import (
    _get_transformer_layers,
)


# ---------------------------------------------------------------------------
# 1. Extract
# ---------------------------------------------------------------------------

def extract_activation(
    model,
    tokenizer,
    device,
    messages: list[dict],
    layer_idx: int,
    token_position: str | int = "last",
    raw_text: Optional[str] = None,
) -> torch.Tensor:
    """Run a source forward pass and return the hidden state at (layer, position).

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        device: Target device for input tensors.
        messages: Chat-formatted source prompt.  Ignored when *raw_text* is set.
        layer_idx: Transformer layer to hook (0-indexed, post-embedding).
        token_position: ``"last"`` or an integer token index.
        raw_text: If given, used verbatim as model input (skips chat template).

    Returns:
        1-D tensor of shape ``(d_model,)``.
    """
    if raw_text is not None:
        input_text = raw_text
    else:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    input_ids = tokenizer.encode(
        input_text, return_tensors="pt", add_special_tokens=False
    ).to(device)

    captured = {}
    transformer_layers = _get_transformer_layers(model)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        pos = -1 if token_position == "last" else int(token_position)
        captured["activation"] = hidden[0, pos, :].detach().clone()

    handle = transformer_layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    return captured["activation"]


# ---------------------------------------------------------------------------
# 2. Inject + generate
# ---------------------------------------------------------------------------

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
    """Inject an activation into an interpretation prompt, then generate text.

    During the prefill pass the hook fires and overwrites (``mode="replace"``)
    or adds to (``mode="add"``) the hidden states at each placeholder position.
    KV-cache is enabled so the hook does **not** fire during auto-regressive
    decoding — the injected information propagates only through the cached
    key/value representations.

    Args:
        model / tokenizer / device: HuggingFace model triple.
        messages: Interpretation prompt (chat format).
        activation: Source activation, shape ``(d_model,)``.
        injection_layer: Layer index for the hook.
        placeholder_positions: Token indices where placeholders sit.
        mode: ``"replace"`` or ``"add"``.
        alpha: Scaling factor (only used when *mode* is ``"add"``).
        max_new_tokens / temperature / do_sample: Generation parameters.
        raw_text: Verbatim input text (skips chat template when set).

    Returns:
        Decoded string of newly generated tokens.
    """
    if raw_text is not None:
        input_text = raw_text
    else:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    input_ids = tokenizer.encode(
        input_text, return_tensors="pt", add_special_tokens=False
    ).to(device)
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
                use_cache=True,
            )
    finally:
        handle.remove()

    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 3. Inject + read logits (no generation)
# ---------------------------------------------------------------------------

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
    save_logprobs: bool = False,
) -> dict:
    """Single forward pass with optional injection; return constrained next-token probs.

    When *activation* is ``None`` (text-only baseline), the forward pass runs
    without any hook — useful for measuring what the model predicts from the
    prompt text alone.

    Returns:
        Dict with keys ``probs``, ``logits``, ``predicted``, and optionally
        ``logprobs`` and ``total_choice_prob``.
    """
    if raw_text is not None:
        input_text = raw_text
    else:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    input_ids = tokenizer.encode(
        input_text, return_tensors="pt", add_special_tokens=False
    ).to(device)

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

    # Next-token logits at the last position
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
# 4. SelfIE relevancy scores
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
    """Per-token relevancy: P(token | WITH injection) − P(token | WITHOUT).

    Two forward passes per generated token position — one with the injection
    hook active, one without.  Covers the first *max_tokens* generated tokens.

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
    scores = []

    n_tokens = min(len(generated_token_ids), max_tokens)

    for i in range(n_tokens):
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
