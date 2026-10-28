"""Readable sample prompts (prompts_preview.md) for the probability runs and the diagnostics.

The full requests are always in requests.jsonl / trials.jsonl, but as JSON they are hard to read.
This writes a short Markdown file showing a handful of representative requests with BOTH chat
messages (system and user) exactly as the model sees them, plus the exact model input for the
first example (special tokens shown) when it is available.

Used by InstrumentalProbs.setup() and AgentVsModelDiagnostic.setup(), so every run saves one. The
same function can be run on an existing run folder (it only reads trials.jsonl and adds a file):

    python -m experiments.agent_vs_model.previews <run_dir> [<run_dir> ...]

The main grid (agent_vs_model_experiment.py) writes its own prompts_preview.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _message_lines(messages: list[dict]) -> list[str]:
    out: list[str] = []
    for m in messages:
        out += [f"**{m['role']} message**", "", "```", m["content"], "```", ""]
    return out


def select_probe_examples(rows: list[dict]) -> list[dict]:
    """v2 probability runs: first code pair, variant a — every condition under the normal and the
    'ignore what you were told' request in the even-rule-first wording, plus the odd-rule-first
    wording once."""
    if not rows:
        return []
    fam = rows[0]["family_id"]
    mapping = next(r["mapping"] for r in rows if r["family_id"] == fam)
    pool = [r for r in rows if r["family_id"] == fam and r["variant"] == "a" and r["mapping"] == mapping]
    conds = list(dict.fromkeys(r["condition"] for r in pool))
    picks: list[dict] = []
    for c in conds:
        for b in ("neutral", "released"):
            hit = next((r for r in pool if r["condition"] == c and r["branch"] == b and r["rule_order"] == "even_first"), None)
            if hit:
                picks.append(hit)
    odd = next((r for r in pool if r["condition"] == conds[0] and r["branch"] == "neutral" and r["rule_order"] == "odd_first"), None)
    if odd:
        picks.append(odd)
    return [{"title": f"{r['condition']} · {r['branch']} request · {r['rule_order'].replace('_', '-')} wording "
                      f"(code {r['target_code']}, mapping {r['mapping']})", "row": r} for r in picks]


def select_diag_examples(rows: list[dict]) -> list[dict]:
    """Diagnostics: crossed check -> first code pair, first mapping, all parity x wording cells;
    capability follow-up -> one of each task (and both wordings of the parity-given task)."""
    picks: list[dict] = []
    crossed = [r for r in rows if r["phase"] == "instrumental_crossed"]
    if crossed:
        fam, mapping = crossed[0]["family_id"], crossed[0]["mapping"]
        picks += [r for r in crossed if r["family_id"] == fam and r["mapping"] == mapping]
    for phase, n in (("extract_digit", 1), ("classify_parity", 1), ("select_given_parity", 2)):
        picks += [r for r in rows if r["phase"] == phase][:n]
    return [{"title": r["item_id"], "row": r} for r in picks]


def write_preview(path: Path, title: str, intro: list[str], examples: list[dict], *,
                  render=None, exact_title: str = "", extra: list[str] | None = None) -> None:
    """`render(messages) -> str` builds the exact model input when the rows don't carry it."""
    L = [f"# {title}", ""] + intro + [""]
    for ex in examples:
        L += [f"## {ex['title']}", ""] + _message_lines(ex["row"]["messages"])
    exact_input = None
    if examples:
        row = examples[0]["row"]
        exact_input = row.get("decoded_input") or row.get("rendered_text") or (render(row["messages"]) if render else None)
    if exact_input:
        L += [f"## Exact model input ({exact_title})", "",
              "The first example above, after the chat template (special tokens such as `<|begin_of_text|>` shown), "
              "exactly as the model reads it.", "", "```", exact_input, "```", ""]
    L += extra or []
    path.write_text("\n".join(L) + "\n")


def _where(row: dict) -> str:
    loc = row.get("code_location") or (row.get("model_settings") or {}).get("code_location") or "user"
    return ("the **system message** (the user message holds only what the Assistant was told about itself, the request "
            "and the question)" if loc == "system" else "the **user message**, together with everything else")


def write_probe_preview(run_dir: Path, rows: list[dict], *, version: str, primer: str, render=None) -> None:
    """Sample prompts for a v2 probability run."""
    intro = [
        f"Sample requests from this run (`{version}`). Each is one chat with the Assistant, shown as the model sees it.",
        "",
        f"- **Where the door code is:** {_where(rows[0])}.",
        "- **How the answer is read:** nothing is generated. The Assistant's reply is started for it with "
        f"`{primer}` and we read how likely it is to continue with `L`, `R` or `ASK`.",
        "- **Examples shown:** the first code pair; every condition under the normal request and under the "
        "\"ignore what you were told earlier\" request, plus the other wording of the door-procedure question once.",
        "- All requests: `requests.jsonl`. Every trial with its probabilities: `trials.jsonl`.",
    ]
    write_preview(run_dir / "prompts_preview.md", f"Sample prompts — {version}", intro, select_probe_examples(rows),
                  render=render, exact_title="with the answer prefix added")


def write_diag_preview(run_dir: Path, rows: list[dict], *, version: str, render=None) -> None:
    """Sample prompts for a diagnostic run."""
    intro = [
        f"Sample requests from this diagnostic (`{version}`). Each is a separate chat, shown as the model sees it.",
        "",
        f"- **Where the door code is (where one is used):** {_where(rows[0])}.",
        "- Examples shown: the crossed check's first code pair in every parity x wording cell, or one of each "
        "capability task.",
        "- All items: `items.jsonl`. Every answer: `trials.jsonl`.",
    ]
    write_preview(run_dir / "prompts_preview.md", f"Sample prompts — {version}", intro, select_diag_examples(rows),
                  render=render, exact_title="what the model is given before it answers")


def backfill(run_dir: Path) -> str:
    """Write prompts_preview.md for an existing probability / diagnostic run folder (adds one file)."""
    run_dir = Path(run_dir)
    rows = [json.loads(l) for l in open(run_dir / "trials.jsonl") if l.strip()]
    if not rows:
        return f"{run_dir.name}: no trials"
    first = rows[0]
    if "p_L" in first:
        write_probe_preview(run_dir, rows, version=first.get("version", "v2-probs"),
                            primer=(first.get("model_settings") or {}).get("primer", '{"choice": "'))
    elif "phase" in first:
        write_diag_preview(run_dir, rows, version=first.get("diagnostic_version", "diag"))
    else:
        return f"{run_dir.name}: main-grid run, already has prompts_preview.md (skipped)"
    return f"{run_dir.name}: wrote prompts_preview.md"


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(backfill(Path(d)))
