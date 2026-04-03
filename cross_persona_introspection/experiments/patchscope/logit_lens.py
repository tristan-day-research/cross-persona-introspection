"""Logit lens: project intermediate hidden states to vocabulary space.

For each specified layer, captures the hidden state at the extraction token
position, multiplies by the model's unembedding matrix (``lm_head``), and
records the top-N most-likely and bottom-N least-likely tokens.

Results are saved to a separate JSON file using the same
``ps_{model}_{timestamp}_`` prefix with ``_LOGIT_LENS_`` inserted.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from . import patchscope_helpers

logger = logging.getLogger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────


@dataclass
class LogitLensEntry:
    """One token-position × layer snapshot."""

    layer: int
    token_text: str
    token_id: int
    token_position: int
    top_tokens: list[dict]      # [{token, token_id, logit, prob}, ...]
    bottom_tokens: list[dict]   # [{token, token_id, logit, prob}, ...]


@dataclass
class LogitLensRecord:
    """One prompt's full logit-lens sweep across layers."""

    phase: str                  # "source" or "reporter"
    persona: str
    question_id: str
    question_text: str
    layers: list[LogitLensEntry] = field(default_factory=list)
    prompt_text: str = ""
    timestamp: Optional[str] = None


# ── Core: project hidden states to vocab ──────────────────────────────────


def _get_lm_head(model) -> torch.nn.Module:
    """Return the unembedding (lm_head) linear layer."""
    if hasattr(model, "lm_head"):
        return model.lm_head  # Llama, Mistral, Qwen, most HF causal LMs
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte  # GPT-2 (tied weights)
    raise ValueError(
        "Cannot locate lm_head. Supported: model.lm_head, model.transformer.wte"
    )


def run_logit_lens(
    model,
    tokenizer,
    device,
    input_text: str,
    layer_indices: list[int],
    token_position: str | int = "last",
    top_n: int = 20,
    bottom_n: int = 20,
    boundary_marker: str | None = None,
) -> list[LogitLensEntry]:
    """Run a single forward pass and project every requested layer's hidden
    state through the unembedding matrix.

    Returns a list of ``LogitLensEntry``, one per layer.
    """
    resolved_pos, _ = patchscope_helpers.resolve_extraction_token_index(
        tokenizer, input_text, token_position, boundary_marker=boundary_marker,
    )
    input_ids = tokenizer.encode(
        input_text, return_tensors="pt", add_special_tokens=False,
    ).to(device)

    token_id = int(input_ids[0, resolved_pos])
    token_text = tokenizer.decode([token_id]).strip()

    transformer_layers = patchscope_helpers._get_transformer_layers(model)
    lm_head = _get_lm_head(model)

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
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

    # Project each captured hidden state through lm_head
    entries: list[LogitLensEntry] = []
    for layer_idx in sorted(captured):
        hidden = captured[layer_idx].unsqueeze(0)  # (1, d_model)
        with torch.no_grad():
            logits = lm_head(hidden).squeeze(0)     # (vocab_size,)
        probs = F.softmax(logits, dim=-1)

        # Top N
        top_vals, top_ids = torch.topk(logits, min(top_n, logits.shape[0]))
        top_tokens = []
        for i in range(top_ids.shape[0]):
            tid = int(top_ids[i])
            top_tokens.append({
                "token": tokenizer.decode([tid]),
                "token_id": tid,
                "logit": round(float(top_vals[i]), 4),
                "prob": round(float(probs[tid]), 6),
            })

        # Bottom N
        bot_vals, bot_ids = torch.topk(logits, min(bottom_n, logits.shape[0]), largest=False)
        bottom_tokens = []
        for i in range(bot_ids.shape[0]):
            tid = int(bot_ids[i])
            bottom_tokens.append({
                "token": tokenizer.decode([tid]),
                "token_id": tid,
                "logit": round(float(bot_vals[i]), 4),
                "prob": round(float(probs[tid]), 6),
            })

        entries.append(LogitLensEntry(
            layer=layer_idx,
            token_text=token_text,
            token_id=token_id,
            token_position=resolved_pos,
            top_tokens=top_tokens,
            bottom_tokens=bottom_tokens,
        ))

    return entries


# ── Experiment-level driver ───────────────────────────────────────────────


def run_logit_lens_sweep(
    experiment,  # PatchscopeExperiment instance
) -> Path | None:
    """Run logit lens for source and/or reporter passes and save results.

    Reads ``logit_lens`` section from the patchscope config.  Returns the
    output JSON path, or ``None`` if logit lens is disabled.
    """
    ps_config = experiment.ps_config
    ll_cfg = ps_config.get("logit_lens")
    if not ll_cfg or not ll_cfg.get("enabled", False):
        logger.info("Logit lens: disabled or not configured — skipping")
        return None

    model = experiment.backend.model
    tokenizer = experiment.backend.tokenizer
    device = experiment.backend.input_device

    top_n = int(ll_cfg.get("top_n", 20))
    bottom_n = int(ll_cfg.get("bottom_n", 20))
    phases = [p.strip().lower() for p in ll_cfg.get("phases", ["source"])]

    # Resolve layers
    num_layers = model.config.num_hidden_layers
    layer_spec = ll_cfg.get("layers", "all")
    layer_indices = patchscope_helpers._resolve_layers(layer_spec, num_layers)

    extraction_cfg = ps_config["extraction"]
    token_position = extraction_cfg.get("token_position", "last")
    boundary_marker = extraction_cfg.get("assistant_boundary_marker")

    source_persona_names = ps_config["source_personas"]
    reporter_persona_names = ps_config["reporter_personas"]

    records: list[LogitLensRecord] = []
    total_start = time.monotonic()

    # ── Source phase ──────────────────────────────────────────────────
    if "source" in phases:
        logger.info("Logit lens: running source-pass sweep")
        for sp_name in source_persona_names:
            source_persona = experiment.all_personas[sp_name]
            for question_idx, question in enumerate(experiment.questions):
                question_id = question.get("question_id", f"q{question_idx}")
                is_manual = (ps_config.get("source_mode") or "questions") == "manual"

                if is_manual:
                    target_word = question["_manual_target_word"]
                    target_strategy = question.get("_manual_target_strategy", "last")
                    prompt_text = question["question_text"]
                    sys_text = (source_persona.system_prompt or "").strip()
                    input_text = (sys_text + "\n\n" + prompt_text) if sys_text else prompt_text
                    tok_pos = patchscope_helpers.find_token_position(
                        tokenizer, input_text, target_word, strategy=target_strategy,
                    )
                else:
                    source_messages = _build_source_messages(
                        ps_config, source_persona, question, tokenizer,
                    )
                    input_text = tokenizer.apply_chat_template(
                        source_messages, tokenize=False, add_generation_prompt=True,
                    )
                    tok_pos = token_position

                entries = run_logit_lens(
                    model, tokenizer, device,
                    input_text=input_text,
                    layer_indices=layer_indices,
                    token_position=tok_pos,
                    top_n=top_n,
                    bottom_n=bottom_n,
                    boundary_marker=boundary_marker,
                )
                records.append(LogitLensRecord(
                    phase="source",
                    persona=sp_name,
                    question_id=question_id,
                    question_text=question.get("question_text", ""),
                    layers=entries,
                    prompt_text=input_text,
                    timestamp=datetime.now().isoformat(),
                ))

        logger.info(f"  Source logit lens: {len(records)} records")

    # ── Reporter phase ────────────────────────────────────────────────
    if "reporter" in phases:
        logger.info("Logit lens: running reporter-pass sweep")
        injection_cfg = ps_config["injection"]
        style = ps_config["prompt_style"]
        configured_placeholder = injection_cfg["placeholder_token"]
        num_placeholders = int(injection_cfg.get("num_placeholders", 1))
        placeholder_token_id = patchscope_helpers._get_placeholder_token_id(
            tokenizer, configured_placeholder,
        )
        placeholder_token = tokenizer.decode([placeholder_token_id])
        base_prompt = ps_config["interpretation_base_prompt"]
        templates = ps_config["interpretation_templates"]
        enabled = ps_config.get("enabled_templates", [])
        if enabled:
            templates = {k: v for k, v in templates.items() if k in enabled}

        for rp_name in reporter_persona_names:
            reporter_persona = experiment.all_personas[rp_name]
            for question_idx, question in enumerate(experiment.questions):
                question_id = question.get("question_id", f"q{question_idx}")
                for tmpl_name, tmpl_cfg in templates.items():
                    interp_text, interp_messages, _ = (
                        patchscope_helpers.build_interpretation_prompt(
                            tokenizer, tmpl_cfg, style, base_prompt,
                            placeholder_token, num_placeholders,
                            question, reporter_persona.system_prompt,
                            use_chat_template=bool(
                                ps_config.get("use_chat_template", True)
                            ),
                        )
                    )
                    entries = run_logit_lens(
                        model, tokenizer, device,
                        input_text=interp_text,
                        layer_indices=layer_indices,
                        token_position="last",
                        top_n=top_n,
                        bottom_n=bottom_n,
                    )
                    records.append(LogitLensRecord(
                        phase="reporter",
                        persona=rp_name,
                        question_id=question_id,
                        question_text=question.get("question_text", ""),
                        layers=entries,
                        prompt_text=interp_text,
                        timestamp=datetime.now().isoformat(),
                    ))

        logger.info(
            f"  Reporter logit lens: "
            f"{len([r for r in records if r.phase == 'reporter'])} records"
        )

    elapsed = time.monotonic() - total_start
    logger.info(f"Logit lens sweep done: {len(records)} total records in {elapsed:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────
    output_path = _build_output_path(experiment)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "layers": layer_indices,
            "top_n": top_n,
            "bottom_n": bottom_n,
            "phases": phases,
            "model": experiment.config.model_name,
        },
        "records": [asdict(r) for r in records],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Logit lens results saved to {output_path}")
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────


def _build_source_messages(
    ps_config: dict, source_persona, question: dict, tokenizer,
) -> list[dict]:
    """Build the chat messages for a source forward pass."""
    source_template = ps_config["source_pass"]["user_message_template"]
    options = question.get("options", {})
    options_formatted = "\n".join(
        f"{label}. {text}" for label, text in sorted(options.items())
    )
    user_text = source_template.format(
        question_text=question.get("question_text", ""),
        options_formatted=options_formatted,
    )
    messages = []
    if source_persona.system_prompt:
        messages.append({"role": "system", "content": source_persona.system_prompt})
    messages.append({"role": "user", "content": user_text.strip()})
    return messages


def _build_output_path(experiment) -> Path:
    """Build the output path with ``_LOGIT_LENS_`` inserted after the
    ``ps_{model}_{timestamp}_`` prefix."""
    experiment._init_output_paths()
    base = experiment._base_name  # e.g. ps_l8b_20240402_153045_detail
    # Split after the timestamp portion: ps_{model}_{date}_{time}_{rest}
    parts = base.split("_", 4)  # ['ps', model, date, time, rest...]
    if len(parts) >= 5:
        prefix = "_".join(parts[:4])
        rest = parts[4]
        name = f"{prefix}_LOGIT_LENS_{rest}.json"
    else:
        name = f"{base}_LOGIT_LENS.json"
    out_dir = Path(experiment.config.output_dir)
    return out_dir / name
