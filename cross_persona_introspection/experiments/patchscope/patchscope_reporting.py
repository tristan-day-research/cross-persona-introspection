"""Patchscope run log reporting.

Writes the detailed companion .txt log file for a patchscope experiment run.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

from cross_persona_introspection.experiments.patchscope.patchscope_helpers import _resolve_layers


def write_run_log(
    log_path: Path,
    base_name: str,
    records: list,
    ps_config: dict,
    run_config,
    backend,
    all_personas: dict,
    sample_source_prompts: dict,
    sample_prompts: dict,
    errors: list[str],
    elapsed: float,
    n_questions: int,
    evaluate_fn: Callable[[], dict],
    no_persona_chat_per_layer: Optional[dict] = None,
) -> None:
    """Write the detailed companion .txt log file.

    Args:
        log_path: Where to write the log.
        base_name: Run name for the header.
        records: List of PatchscopeRecord.
        ps_config: Patchscope YAML config dict.
        run_config: RunConfig for this experiment.
        backend: HFBackend (or None).
        all_personas: {name: PersonaConfig} dict.
        sample_source_prompts: Sample source prompts captured during Phase 1.
        sample_prompts: Sample reporter prompts captured during Phase 2.
        no_persona_chat_per_layer: Optional {layer_tag: sample dict} for log-only
            plain prompts (synthetic preamble + template; no apply_chat_template).
        errors: List of error messages.
        elapsed: Wall-clock seconds elapsed.
        n_questions: Number of questions loaded.
        evaluate_fn: Callable that returns metrics dict (PatchscopeExperiment.evaluate).
    """
    sep = "=" * 70
    thin = "-" * 70
    lines = []

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
        f"Records    : {len(records)}",
        f"Errors     : {sum(1 for r in records if r.error)}",
        "",
    ]

    # ── Matrix dimensions (aligned with experiment layer / template resolution)
    ps = ps_config
    n_src = len(ps.get("source_personas", []))
    n_reporter = len(ps.get("reporter_personas", []))
    all_templates = ps.get("interpretation_templates", {})
    enabled_only = ps.get("enabled_templates") or []
    if enabled_only:
        template_keys = [k for k in enabled_only if k in all_templates]
        n_templates = len(template_keys)
    else:
        template_keys = list(all_templates.keys())
        n_templates = len(template_keys)
    conditions = ["real"]
    if ps.get("controls", {}).get("text_only_baseline"):
        conditions.append("text_only_baseline")
    if ps.get("controls", {}).get("shuffled_activation"):
        conditions.append("shuffled")
    n_conditions = len(conditions)

    num_model_layers = backend.model.config.num_hidden_layers if backend else None
    if ps.get("layer_sweep", {}).get("enabled"):
        n_src_layers = len(ps["layer_sweep"]["source_layers"])
        n_inj_layers = len(ps["layer_sweep"]["injection_layers"])
        layer_cells = n_src_layers * n_inj_layers
    elif ps.get("layer_pairs"):
        pairs = ps["layer_pairs"]
        n_src_layers = len({int(p[0]) for p in pairs})
        n_inj_layers = len({int(p[1]) for p in pairs})
        layer_cells = len(pairs)
    else:
        n_src_layers = (
            len(_resolve_layers(ps["extraction"]["layers"], num_model_layers))
            if num_model_layers is not None
            else "?"
        )
        n_inj_layers = 1
        layer_cells = n_src_layers * n_inj_layers if isinstance(n_src_layers, int) else "?"

    if isinstance(layer_cells, int):
        total = n_questions * n_src * layer_cells * n_reporter * n_templates * n_conditions
    else:
        total = "?"

    lines += [sep, "MATRIX DIMENSIONS", thin]
    lines += [
        f"  questions           : {n_questions}",
        f"  source_personas     : {n_src}  {ps.get('source_personas', [])}",
        f"  extraction_layers   : {n_src_layers}",
        f"  injection_layers    : {n_inj_layers}",
        f"  layer_cells (src×inj per config): {layer_cells}",
        f"  reporter_personas   : {n_reporter}  {ps.get('reporter_personas', [])}",
        f"  templates           : {n_templates}  {template_keys}",
        f"  conditions          : {n_conditions}  {conditions}",
        f"  total cells (nominal): {total}",
        "",
    ]

    ex_cfg = ps.get("extraction") or {}
    lines += [
        thin,
        "SOURCE ACTIVATION EXTRACTION (Phase 1 — patchscope_patching)",
        thin,
        f"  extraction.readout : {ex_cfg.get('readout', 'prefill')!r}",
        "    prefill = one forward at token_position (prompt-only; NOT assistant-generated text).",
        "    during_generation / autoregressive = prefill then decode_steps single-token forwards,",
        "    capture hidden[batch,-1,:] on the last decode forward (generated-token activations).",
        f"  extraction.token_position : {ex_cfg.get('token_position', 'last')!r}",
        "    → prefill only when readout is prefill. Use token_position: during_generation OR",
        "    readout: during_generation for activations while the model generates its answer.",
        "  Prefill values: 'last', 'last_before_assistant' (end of user turn, often whitespace), int index.",
        f"  assistant_boundary_marker : {ex_cfg.get('assistant_boundary_marker', '(default in code)')!r}",
        "",
    ]

    # ── Run parameters
    cfg = run_config
    lines += [sep, "RUN PARAMETERS (from experiments.yaml)", thin]
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
    lines += [yaml.dump(ps_config, default_flow_style=False)]

    # ── Model info
    lines += [sep, "MODEL INFO", thin]
    if backend:
        model = backend.model
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
        ps_config.get("source_personas", [])
        + ps_config.get("reporter_personas", [])
    )
    for pname in sorted(all_persona_names):
        persona = all_personas.get(pname)
        lines.append(f"\n[{pname}]")
        if persona and persona.system_prompt:
            for sline in persona.system_prompt.strip().splitlines():
                lines.append(f"  {sline}")
        else:
            lines.append("  (no system prompt)")
    lines.append("")

    # ── Source prompt examples
    if sample_source_prompts:
        lines += [sep, "SOURCE PROMPT EXAMPLES (verbatim, including special tokens)", thin]
        for sp_name, sample in sorted(sample_source_prompts.items()):
            lines.append(f"\n--- {sp_name} ---")
            lines.append(f"  question_id   : {sample['question_id']}")
            if sample.get("option_order"):
                lines.append(f"  option_order  : {sample['option_order']}")
            if sample.get("answer") is not None:
                lines.append(
                    f"  source_answer : {sample['answer']} (shuffled: {sample.get('shuffled_answer', '?')})"
                )
            lines.append("")
            site = sample.get("extraction_site") or {}
            if site.get("error"):
                lines += [
                    "  ── ACTIVATION EXTRACTION SITE (source prompt; same index at every hooked layer) ──",
                    f"  ERROR: {site['error']}",
                    "",
                ]
            elif site.get("capture_mode") == "during_generation" or site.get("readout") in (
                "autoregressive",
                "during_generation",
                "while_generating",
            ):
                lines += [
                    "  ── ACTIVATION EXTRACTION (during generation — not prefill token_position) ──",
                ]
                _mode = site.get("autoregressive_mode")
                if _mode:
                    lines.append(f"  autoregressive_mode : {_mode}")
                if site.get("stop_tokens") is not None:
                    lines.append(f"  stop_tokens         : {site.get('stop_tokens')}")
                if site.get("max_decode_budget") is not None:
                    lines.append(
                        f"  max_decode_budget   : {site.get('max_decode_budget')} "
                        f"(safety cap; actual rollout may stop earlier)"
                    )
                if site.get("decode_steps") is not None:
                    lines.append(f"  decode_steps        : {site.get('decode_steps')}")
                lines += [
                    f"  max_decode_steps    : {site.get('max_decode_steps')}",
                    f"  temperature         : {site.get('temperature')}",
                    f"  do_sample           : {site.get('do_sample')}",
                    f"  capture (spec)      : {site.get('token_position_spec')!r}",
                    f"  prefill_tokens      : {site.get('n_tokens')}",
                ]
                if site.get("capture_at_step") is not None:
                    lines.append(
                        f"  capture_at_step     : {site.get('capture_at_step')} "
                        f"(post-prefill generated-token index, 1-based)"
                    )
                if site.get("n_generated_tokens") is not None:
                    lines.append(
                        f"  n_generated_tokens  : {site.get('n_generated_tokens')}"
                    )
                if site.get("stopped_on_token") is True:
                    lines.append(
                        "  stopped_on_token    : True (matched stop_tokens before hitting budget)"
                    )
                elif site.get("stopped_on_token") is False and site.get("stop_tokens"):
                    lines.append(
                        "  stopped_on_token    : False (ran full budget without stop_tokens match)"
                    )
                _act_strip = site.get("activation_token_decoded_strip")
                _act_id = site.get("activation_token_id")
                if _act_id is not None:
                    lines += [
                        "  ── Phase-1 vector = hidden[:, -1, :] after THIS generated token ──",
                        f"  activation_token_id           : {_act_id}",
                        f"  activation_token_decoded_strip : {_act_strip!r}",
                    ]
                lines.append("")
                gids = site.get("generated_token_ids")
                if gids is not None:
                    lines.append(f"  generated_ids    : {gids}")
                gtext = site.get("generated_decode_concat")
                if gtext is not None:
                    lines.append(f"  generated_text   : {gtext!r}")
                pieces = site.get("generated_decode_pieces") or []
                trepr = site.get("generated_token_repr") or []
                if pieces:
                    lines.append("  per-step tokens  :")
                    for i, piece in enumerate(pieces):
                        tid = gids[i] if gids and i < len(gids) else "?"
                        r = trepr[i] if i < len(trepr) else repr(piece)
                        lines.append(f"    step {i + 1}: id={tid} {r} -> {piece!r}")
                lines.append("")
            elif site.get("capture_mode") == "manual_target_word":
                lines += [
                    "  ── ACTIVATION EXTRACTION (manual prompt — target word) ──",
                    f"  target_word        : {site.get('target_word')!r}",
                    f"  target_strategy    : {site.get('target_strategy')!r}",
                    f"  token_index (0-based) : {site.get('token_index')}  (sequence length {site.get('n_tokens')} tokens)",
                    f"  token_id              : {site.get('token_id')}",
                    f"  decoded token (repr)  : {site.get('token_decoded_repr')}",
                    f"  decoded token (strip) : {site.get('token_decoded_strip')!r}",
                    "",
                ]
            elif site:
                lines += [
                    "  ── ACTIVATION EXTRACTION SITE (source prompt; same index at every hooked layer) ──",
                    f"  token_position (config) : {site.get('token_position_spec')!r}",
                ]
                if site.get("boundary_marker") is not None:
                    lines.append(
                        f"  boundary_marker         : {site.get('boundary_marker')!r} "
                        f"@ char {site.get('boundary_char_index')}"
                    )
                lines += [
                    f"  token_index (0-based)   : {site.get('token_index')}  (sequence length {site.get('n_tokens')} tokens, indices 0..{site.get('n_tokens', 1) - 1})",
                    f"  token_id                  : {site.get('token_id')}",
                    f"  decoded token (repr)      : {site.get('token_decoded_repr')}",
                    f"  decoded token (strip)     : {site.get('token_decoded_strip')!r}",
                ]
                _ts = site.get("token_decoded_strip")
                if _ts is not None and str(_ts).strip() == "":
                    lines.append(
                        "  note                      : Whitespace-only (or empty-after-strip) token is "
                        "typical for last_before_assistant — end of user turn before assistant header, "
                        "not the MCQ letter. For activations at generated answer tokens, set "
                        "extraction.readout: during_generation (or token_position: during_generation)."
                    )
                lines.append("")
            lines += [
                "  ── FULL SOURCE PROMPT (exact tokens sent to model) ──",
            ]
            for pline in sample["prompt_text"].splitlines():
                lines.append(f"  {pline}")
            lines.append("")
        lines.append("")

    # ── One no-persona plain-prompt example per source→injection layer pair (.txt only)
    npc = no_persona_chat_per_layer or {}
    if npc:

        def _layer_tag_sort_key(tag: str) -> tuple[int, int]:
            try:
                body = tag[1:] if tag.startswith("L") else tag
                src_s, inj_s = body.split("to", 1)
                return (int(src_s), int(inj_s))
            except (ValueError, IndexError):
                return (0, 0)

        lines += [
            sep,
            "NO-PERSONA PROMPT (one per layer pair; same template as matrix, no reporter persona)",
            thin,
            "  Exact string encoded with add_special_tokens=false. Body = interpretation_templates",
            "  for this template×prompt_style, unless reporting.no_persona_layer_log_body is set.",
            "  Optional preamble: reporting.no_persona_layer_log_system_prompt (usually empty).",
            "  Same activation as the first 'real' cell for that layer pair.",
            "",
        ]
        for layer_tag in sorted(npc.keys(), key=_layer_tag_sort_key):
            sample = npc[layer_tag]
            syn = (sample.get("layer_log_system_prompt") or "").strip()
            syn_note = repr(syn[:200] + ("…" if len(syn) > 200 else "")) if syn else "(empty)"
            bod = (sample.get("layer_log_body_override") or "").strip()
            bod_note = repr(bod[:120] + ("…" if len(bod) > 120 else "")) if bod else "(empty — same body as matrix template)"
            lines += [
                f"\n--- {layer_tag} ---",
                f"  template         : {sample.get('template', '')}",
                f"  question_id      : {sample.get('question_id', '')}",
                f"  source_persona   : {sample.get('source_persona', '')}",
                f"  reporter_persona : {sample.get('reporter_persona', '')} (persona text not prepended here)",
                f"  source_layer     : {sample.get('source_layer', '')}",
                f"  injection_layer  : {sample.get('injection_layer', '')}",
                f"  preamble (reporting.no_persona_layer_log_system_prompt) : {syn_note}",
                f"  body override (reporting.no_persona_layer_log_body)     : {bod_note}",
                "",
                "  ── FULL PROMPT (verbatim; add_special_tokens=false when encoding) ──",
            ]
            for pline in sample.get("interp_prompt_text", "").splitlines():
                lines.append(f"  {pline}")
            lines += [
                "",
                "  ── MODEL RESPONSE ──",
            ]
            for rline in sample.get("reporter_generated_text", "").splitlines():
                lines.append(f"  {rline}")
            lines.append("")
        lines.append("")

    # ── Sample reporter prompts (match + oppose per template×condition×layer pair; + no-system)
    lines += [
        sep,
        "SAMPLE REPORTER PROMPTS AND RESPONSES (verbatim, including special tokens)",
        thin,
        "  One *match* (source==reporter) and one *oppose* sample per layer pair, per template×condition,",
        "  when such cells occur in the matrix. Keys look like: {template}_{condition}_L{src}to{inj}_{match|oppose}.",
        "  Each block may include a log-only decode with no reporter system (see reporting config).",
        "",
    ]

    def _reporter_sample_sort_key(item: tuple[str, dict]) -> tuple:
        _k, s = item
        align = s.get("persona_alignment")
        align_rank = {"match": 0, "oppose": 1}.get(align, 2)
        return (
            s.get("template", ""),
            s.get("condition", ""),
            int(s.get("source_layer", 0)),
            int(s.get("injection_layer", 0)),
            align_rank,
            _k,
        )

    for sample_key, sample in sorted(sample_prompts.items(), key=_reporter_sample_sort_key):
        lines += [
            f"\n--- {sample_key} ---",
            f"  template          : {sample['template']}",
            f"  condition         : {sample['condition']}",
            f"  persona_alignment : {sample.get('persona_alignment', '(n/a)')}",
            f"  source_persona    : {sample['source_persona']}",
            f"  reporter_persona  : {sample['reporter_persona']}",
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
        for rline in sample["reporter_generated_text"].splitlines():
            lines.append(f"  {rline}")
        lines.append("")
        ns = sample.get("no_reporter_system")
        if ns:
            lines += [
                "",
                "  ── LOG-ONLY: no reporter persona from personas.yaml (not in JSONL) ──",
                "      Uses reporting.no_persona_layer_log_system_prompt when set; if unset and",
                "      text_only_baseline with use_chat_template false, chat template is forced for this decode only.",
            ]
            for pline in ns["interp_prompt_text"].splitlines():
                lines.append(f"  {pline}")
            lines += [
                "",
                "  ── MODEL RESPONSE (no reporter system) ──",
            ]
            for rline in ns["reporter_generated_text"].splitlines():
                lines.append(f"  {rline}")
            lines.append("")
    lines.append("")

    # ── Source override results (Phase 3)
    override_records = [r for r in records if r.source_persona == "source_override"]
    if override_records:
        lines += [sep, "SOURCE OVERRIDE RESULTS (raw text, specific word extraction)", thin]
        current_qid = None
        for r in override_records:
            if r.question_id != current_qid:
                current_qid = r.question_id
                lines.append(f"\n  {r.question_id}: \"{r.question_text}\"")
            lines.append(
                f"    L{r.source_layer:>2} → {r.injection_layer:>2}  "
                f"{r.template_name:<20s}  '{r.reporter_generated_text.strip()[:70]}'"
            )
        lines.append("")

    # ── Results summary
    lines += [sep, "RESULTS SUMMARY", thin]
    metrics = evaluate_fn()
    for k, v in metrics.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # ── Errors
    lines += [sep, f"ERRORS AND WARNINGS ({len(errors)} total)", thin]
    if errors:
        for err in errors:
            lines.append(f"  {err}")
    else:
        lines.append("  None")
    lines.append("")

    lines.append(sep)
    log_path.write_text("\n".join(lines), encoding="utf-8")
