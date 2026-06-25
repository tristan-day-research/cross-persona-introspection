"""Activation capture for the binary 12-case evaluation.

Maps the eval's NAMED token positions (from the activation-capture spec) to token
spans in the primed prompt and drives core.activation_capture / core.activation_store.
Capture runs as ONE unbatched forward per trial (so token offsets line up exactly
with the prompt), separate from the batched A/B logit read — see _run_binary_trials.

Named eval positions captured per trial:
  active_system_prompt_final, pre_description_token, other_description_mean,
  pre_text_token, text1_{mean,last10_mean,final}, text2_{mean,last10_mean,final},
  assistant_header_token, final_prompt_token_before_answer
text2_* (and descriptions for case1) are absent → stored as nothing for that trial.
"""

from __future__ import annotations

from pathlib import Path

from core.activation_capture import (
    capture_sequence, char_span_to_token_span, extract_positions, model_num_layers,
    resolve_layers,
)
from core.activation_store import ActivationStore
from core.self_recognition import ANSWER_PRIMER, render_with_primer


def eval_capture_spans(full_prompt: str, user_prompt: str, seg_spans: dict, *,
                       offsets: list, seq_len: int, system_prompt_text: str | None = None,
                       primer: str = ANSWER_PRIMER) -> dict:
    """Token-index position specs (for extract_positions) for one eval trial.

    `seg_spans` are char spans of the text/description segments WITHIN
    `user_prompt`; `offsets` are per-token char spans of the full primed prompt.
    The user content appears verbatim in the templated prompt, so its segment
    offsets shift by where the user turn starts."""
    user_start = full_prompt.find(user_prompt)
    user_start = max(0, user_start)

    def seg_tok(name):
        if name not in seg_spans:
            return None
        cs, ce = seg_spans[name]
        return char_span_to_token_span(offsets, user_start + cs, user_start + ce)

    out: dict = {}
    # Text 1 (or the single text for cases 1–2).
    t1 = seg_tok("text1") or seg_tok("single_text")
    if t1:
        s, e = t1
        out["text1_mean"] = ("mean", s, e)
        out["text1_last10_mean"] = ("last10_mean", s, e)
        out["text1_final"] = ("final", s, e)
        if s >= 1:
            out["pre_text_token"] = ("token", s - 1)
    # Text 2 (two-text cases only).
    t2 = seg_tok("text2")
    if t2:
        s, e = t2
        out["text2_mean"] = ("mean", s, e)
        out["text2_last10_mean"] = ("last10_mean", s, e)
        out["text2_final"] = ("final", s, e)
    # Descriptions (describing cases only). other_description_mean spans all
    # description tokens; the two-described-persona cases (5/8/9) also capture each
    # description individually for switching analyses.
    desc = [(nm, seg_tok(nm)) for nm in
            ("other_description", "other_description_1", "other_description_2")]
    desc = [(nm, ts) for nm, ts in desc if ts]
    if desc:
        ds = min(ts[0] for _, ts in desc)
        de = max(ts[1] for _, ts in desc)
        out["other_description_mean"] = ("mean", ds, de)
        if ds >= 1:
            out["pre_description_token"] = ("token", ds - 1)
        for nm, (s, e) in desc:
            if nm != "other_description":
                out[f"{nm}_mean"] = ("mean", s, e)
    # Active system prompt (when induced): final token of its span.
    if system_prompt_text:
        needle = system_prompt_text.strip()
        sc = full_prompt.find(needle)
        if sc != -1:
            st = char_span_to_token_span(offsets, sc, sc + len(needle))
            if st:
                out["active_system_prompt_final"] = ("final", st[0], st[1])
    # Assistant header = last token before the answer primer.
    pc = full_prompt.rfind(primer)
    if pc > 0:
        ht = char_span_to_token_span(offsets, pc - 1, pc)
        if ht:
            out["assistant_header_token"] = ("token", ht[0])
    # The answer is the next token after the final prompt token.
    out["final_prompt_token_before_answer"] = ("token", seq_len - 1)
    return out


def capture_metadata(trial, *, predicted: str, prob_correct, logprob_A, logprob_B,
                     run_id: str = "", config_name: str = "",
                     generation_run_id: str = "") -> dict:
    """Columnar metadata row joined to activations on trial_id (spec: METADATA)."""
    return {
        "trial_id": trial.trial_id,
        "run_id": run_id,
        "generation_run_id": generation_run_id,
        "config_name": config_name,
        "case": trial.case_id,
        "case_type": trial.case_type,
        "group": trial.group,
        "task_id": trial.task_id,
        "evaluator_persona": trial.evaluator_persona,
        "other_persona": trial.other_persona,
        "other_persona_1": trial.other_persona_1,
        "other_persona_2": trial.other_persona_2,
        "true_author": trial.source_persona,
        "text1_source_persona": trial.text1_source_persona,
        "text2_source_persona": trial.text2_source_persona,
        "text_order": trial.text_order,
        "answer_order": trial.answer_order,
        "system_prompt_present": bool(trial.eval_system_prompt_enabled),
        "other_description_style": trial.other_description_style,
        "correct_answer": trial.correct_answer,
        "predicted_answer": predicted,
        "correct": predicted == trial.correct_answer,
        "answer_confidence": prob_correct,
        "logprob_A": logprob_A,
        "logprob_B": logprob_B,
    }


class BinaryActivationCollector:
    """Owns the per-run activation store + capture layers for the eval phase."""

    def __init__(self, backend, opts, run_config, out_dir: Path):
        n = model_num_layers(backend.model)
        self.layers = tuple(opts.activation_layers) if opts.activation_layers \
            else resolve_layers(n)
        hidden = getattr(backend.model.config, "hidden_size", None)
        self.store = ActivationStore(
            out_dir, "evaluation", layers=self.layers, hidden_dim=hidden,
            model=run_config.model_name, shard_size=opts.activation_shard_size,
        )

    def has(self, trial_id: str) -> bool:
        return self.store.has(trial_id)

    def capture(self, backend, trial, messages, user_prompt, seg_spans,
                system_prompt_text, meta: dict) -> None:
        full_prompt = render_with_primer(backend, messages)
        states, offsets = capture_sequence(backend, full_prompt, self.layers)
        if offsets is None:  # tokenizer without offset mapping — cannot place spans
            return
        spans = eval_capture_spans(
            full_prompt, user_prompt, seg_spans, offsets=offsets,
            seq_len=states.shape[1], system_prompt_text=system_prompt_text,
        )
        self.store.add(trial.trial_id, extract_positions(states, spans), meta)

    def close(self) -> None:
        self.store.close()
