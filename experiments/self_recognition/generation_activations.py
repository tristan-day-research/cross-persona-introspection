"""Activation capture for the text-generation phase.

Captures the spec's generation-phase named positions once per generated text:
  persona_prompt_mean, persona_prompt_final, generation_prompt_final,
  generated_text_mean, generated_text_last10_mean
via one extra forward over (rendered_prompt + raw_generated_text). Keyed by
text_id = "{group}/{persona}/{task_id}" — the SAME id the eval records as
text1_generation_id / text2_generation_id, so the two phases join cleanly.

Uses the shared core.activation_capture / core.activation_store primitives.
"""

from __future__ import annotations

from pathlib import Path

from core.activation_capture import (
    capture_sequence, char_span_to_token_span, extract_positions, model_num_layers,
    resolve_layers,
)
from core.activation_store import ActivationStore


def generation_capture_spans(full_prompt: str, rendered_prompt: str, *, offsets: list,
                             seq_len: int, system_prompt_text: str | None) -> dict:
    """Token-index position specs for one generated text. `rendered_prompt` is the
    chat-templated prompt (no generation yet); `full_prompt` = rendered_prompt +
    raw generated text; `offsets` are the full prompt's per-token char spans."""
    out: dict = {}
    # Persona (system) prompt span.
    if system_prompt_text:
        needle = system_prompt_text.strip()
        sc = full_prompt.find(needle)
        if sc != -1:
            st = char_span_to_token_span(offsets, sc, sc + len(needle))
            if st:
                out["persona_prompt_mean"] = ("mean", st[0], st[1])
                out["persona_prompt_final"] = ("final", st[0], st[1])
    # Final prompt token (the position whose next token starts the generation).
    cut = len(rendered_prompt)
    prompt_end = char_span_to_token_span(offsets, cut - 1, cut)
    if prompt_end:
        out["generation_prompt_final"] = ("token", prompt_end[1] - 1)
    # Generated-text span: tokens that start at/after the prompt cut.
    gen_start = next((i for i, (cs, _ce) in enumerate(offsets) if cs >= cut), None)
    if gen_start is not None and gen_start < seq_len:
        out["generated_text_mean"] = ("mean", gen_start, seq_len)
        out["generated_text_last10_mean"] = ("last10_mean", gen_start, seq_len)
    return out


class GenerationActivationCollector:
    """Owns the per-run activation store + capture layers for the generation phase."""

    def __init__(self, backend, run_config, out_dir: Path):
        n = model_num_layers(backend.model)
        self.layers = tuple(run_config.activation_layers) if run_config.activation_layers \
            else resolve_layers(n)
        hidden = getattr(backend.model.config, "hidden_size", None)
        self.store = ActivationStore(
            out_dir, "generation", layers=self.layers, hidden_dim=hidden,
            model=run_config.model_name, shard_size=run_config.activation_shard_size,
        )

    def has(self, text_id: str) -> bool:
        return self.store.has(text_id)

    def capture(self, backend, *, text_id: str, rendered_prompt: str, raw_text: str,
                system_prompt_text: str | None, meta: dict) -> None:
        if self.store.has(text_id):
            return
        full_prompt = rendered_prompt + (raw_text or "")
        states, offsets = capture_sequence(backend, full_prompt, self.layers)
        if offsets is None:
            return
        spans = generation_capture_spans(
            full_prompt, rendered_prompt, offsets=offsets, seq_len=states.shape[1],
            system_prompt_text=system_prompt_text,
        )
        self.store.add(text_id, extract_positions(states, spans), meta)

    def close(self) -> None:
        self.store.close()
