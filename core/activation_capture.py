"""Residual-stream activation capture for the self-recognition experiments.

Captures the residual stream (decoder-block output) at a handful of layers and a
handful of NAMED TOKEN POSITIONS per sequence — never whole sequences — so one
shared, compact activation dataset can be reused across analyses. Pairs with
core.activation_store (sharded on-disk storage) and is driven by the generation
and evaluation phases.

Design:
  * Layers default to DEFAULT_LAYERS (0-indexed, for a 32-layer Llama-3-8B);
    resolve_layers() scales them by depth fraction for models of other depths
    (e.g. the 24B Dolphin) and clamps to the model's range.
  * Capture is via forward hooks on the decoder blocks (memory-cheap: only the
    chosen layers, only the named positions are kept), and runs on ONE sequence
    at a time (batch=1, no padding) so token offsets line up exactly with an
    offset-mapping tokenization — keeping span identification simple and correct.
  * Named positions (*_mean / *_last10_mean / *_final / single token) are computed
    from the captured per-token states by extract_positions() given token spans.

The capture forward is separate from the (possibly batched) logit read the eval
uses for prob_A/prob_B, so batching/padding never corrupts the spans.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 0-indexed residual-stream layers to capture, tuned for a 32-layer Llama-3-8B.
DEFAULT_LAYERS = (4, 6, 8, 12, 14, 16, 20, 24, 31)
_BASE_DEPTH = 32  # the depth DEFAULT_LAYERS were chosen for


def resolve_layers(n_layers: int, layers=DEFAULT_LAYERS) -> tuple[int, ...]:
    """The capture layers for a model with `n_layers` decoder blocks.

    If every requested layer is in range, use as-is. Otherwise scale each by the
    depth fraction relative to the 32-layer base (so e.g. the 24B Dolphin's
    deeper/shallower stack is sampled at the same relative depths), clamp to
    [0, n_layers-1], de-duplicate, and sort.
    """
    if max(layers) < n_layers:
        return tuple(sorted(set(layers)))
    scaled = {min(n_layers - 1, max(0, round(l * (n_layers - 1) / (_BASE_DEPTH - 1))))
              for l in layers}
    return tuple(sorted(scaled))


def decoder_layers(model) -> nn.ModuleList:
    """The decoder-block ModuleList of a causal LM, robust to PEFT wrapping.

    Picks the longest ModuleList whose qualified name ends in "layers" — the
    decoder stack for Llama / Mistral / their PEFT-wrapped variants."""
    cands = [(name, mod) for name, mod in model.named_modules()
             if isinstance(mod, nn.ModuleList) and name.endswith("layers") and len(mod) > 0]
    if not cands:
        raise ValueError("could not locate decoder layers (no '*.layers' ModuleList)")
    return max(cands, key=lambda nm: len(nm[1]))[1]


def model_num_layers(model) -> int:
    return len(decoder_layers(model))


class ActivationCapturer:
    """Context manager that hooks the residual stream at the chosen layers.

    Within the context, every forward pass through a hooked block appends its
    output hidden states (fp16, on CPU) to a per-layer buffer. For a single
    forward over one sequence, `last_states()` returns a [num_layers, seq, hidden]
    tensor (layers in the order given). Re-usable across sequences via reset().
    """

    def __init__(self, model, layers):
        self.layers = list(layers)
        self._blocks = decoder_layers(model)
        self._buf: dict[int, list[torch.Tensor]] = {}
        self._handles: list = []

    def __enter__(self) -> "ActivationCapturer":
        for li in self.layers:
            self._handles.append(self._blocks[li].register_forward_hook(self._hook(li)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, li: int):
        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out  # [batch, seq, hidden]
            self._buf.setdefault(li, []).append(hs.detach().to("cpu", torch.float16))
        return hook

    def reset(self) -> None:
        self._buf = {}

    def last_states(self, batch_index: int = 0) -> torch.Tensor:
        """Stack the most recent forward's hidden states across the capture layers
        for one batch row → [num_layers, seq, hidden]."""
        return torch.stack([self._buf[li][-1][batch_index] for li in self.layers])


def capture_sequence(backend, prompt_text: str, layers) -> tuple[torch.Tensor, list]:
    """Run ONE unbatched forward over a pre-rendered prompt string and return
    (states, offsets):

      states  — [num_layers, seq, hidden] fp16 residual stream (CPU).
      offsets — per-token (char_start, char_end) from the fast tokenizer, so a
                char span in `prompt_text` maps to token indices (see
                char_span_to_token_span). None if the tokenizer lacks offsets.

    No padding and add_special_tokens=False (the prompt already carries its own
    template/special tokens), so token positions line up with `prompt_text`.
    """
    enc = backend.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False,
                            return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist() if "offset_mapping" in enc else None
    input_ids = enc["input_ids"].to(backend.input_device)
    attention_mask = torch.ones_like(input_ids)
    with ActivationCapturer(backend.model, layers) as cap:
        with torch.no_grad():
            backend.model(input_ids, attention_mask=attention_mask)
        states = cap.last_states(0)
    return states, offsets


def char_span_to_token_span(offsets: list, char_start: int, char_end: int):
    """Token [start, end) covering a character range, or None if empty. `offsets`
    is the fast tokenizer's per-token (char_start, char_end) list."""
    toks = [i for i, (cs, ce) in enumerate(offsets)
            if ce > char_start and cs < char_end and ce > cs]
    return (toks[0], toks[-1] + 1) if toks else None


def extract_positions(states: torch.Tensor, spans: dict) -> dict:
    """Reduce per-token states to the named positions.

    `states` is [num_layers, seq, hidden]; `spans` maps a capture name to a spec:
      ("mean", s, e)         mean over tokens [s:e]
      ("last10_mean", s, e)  mean over the final ≤10 tokens of [s:e]
      ("final", s, e)        the token at e-1
      ("token", i)           the single token i
    Returns {name: [num_layers, hidden] fp16}; a name maps to None when its span
    is empty/missing (e.g. text2 for single-text cases).
    """
    seq = states.shape[1]
    out: dict[str, torch.Tensor | None] = {}
    for name, spec in spans.items():
        if spec is None:
            out[name] = None
            continue
        kind = spec[0]
        if kind == "token":
            idx = spec[1]
            out[name] = states[:, idx, :].clone() if -seq <= idx < seq else None
            continue
        s, e = spec[1], spec[2]
        s = max(0, min(s, seq)); e = max(0, min(e, seq))
        if e <= s:
            out[name] = None
            continue
        seg = states[:, s:e, :]
        if kind == "mean":
            out[name] = seg.mean(dim=1)
        elif kind == "last10_mean":
            out[name] = seg[:, -10:, :].mean(dim=1)
        elif kind == "final":
            out[name] = seg[:, -1, :].clone()
        else:
            raise ValueError(f"unknown position kind {kind!r} for {name!r}")
    return out
