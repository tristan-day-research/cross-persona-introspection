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
        "SOURCE ACTIVATION EXTRACTION (Phase 1 — see patchscope_patching.extract_activations_multi_layer)",
        thin,
        f"  extraction.token_position : {ex_cfg.get('token_position', 'last')!r}",
        "    → one sequence index per layer; hidden[batch, pos, :] at each hooked layer after one forward.",
        "  Values: 'last' = final token of the full templated string (often '\\n\\n' after the",
        "    assistant header — that is normal Llama chat scaffolding, not your MCQ text).",
        "  'last_before_assistant' = last token before extraction.assistant_boundary_marker",
        "    (default substring matches Llama 3 user/assistant boundary).",
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
            lines += [
                f"\n--- {sp_name} ---",
                f"  question_id   : {sample['question_id']}",
                f"  option_order  : {sample.get('option_order', 'ABCD')}",
                f"  source_answer : {sample['answer']} (shuffled: {sample.get('shuffled_answer', '?')})",
                "",
            ]
            site = sample.get("extraction_site") or {}
            if site.get("error"):
                lines += [
                    "  ── ACTIVATION EXTRACTION SITE (source prompt; same index at every hooked layer) ──",
                    f"  ERROR: {site['error']}",
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
                    "",
                ]
            lines += [
                "  ── FULL SOURCE PROMPT (exact tokens sent to model) ──",
            ]
            for pline in sample["prompt_text"].splitlines():
                lines.append(f"  {pline}")
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
        for rline in sample["generated_text"].splitlines():
            lines.append(f"  {rline}")
        lines.append("")
        ns = sample.get("no_reporter_system")
        if ns:
            lines += [
                "",
                "  ── LOG-ONLY: same cell with NO reporter system message (not in JSONL) ──",
            ]
            for pline in ns["interp_prompt_text"].splitlines():
                lines.append(f"  {pline}")
            lines += [
                "",
                "  ── MODEL RESPONSE (no reporter system) ──",
            ]
            for rline in ns["generated_text"].splitlines():
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
                f"{r.template_name:<20s}  '{r.generated_text.strip()[:70]}'"
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
