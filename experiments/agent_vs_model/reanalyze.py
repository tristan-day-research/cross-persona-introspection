"""Reanalyse an existing agent_vs_model run WITHOUT new model generations.

Reads <source_run>/trials.jsonl (never modified) and writes the current analysis
into a separate directory:

    experiments/agent_vs_model/results/reanalysis/<source_run_name>__reanalysis_<timestamp>/
        REANALYSIS.md            provenance, independent recomputation, token reconstruction
        report.md, *.csv, *.png  current analysis of the source trials
        reconstructed_tokens.csv (if the tokenizer is available)

The independent recomputation re-parses raw responses with plain regexes (not
scoring.py) so the key findings do not rest on the code under review.

Token counts recorded in a run are never relabelled. Reconstructed counts are
computed with the LOCAL tokenizer + transformers version (recorded alongside) and
labelled as reconstructions; they may differ from what the original environment fed.

Usage (repo root):
    python -m experiments.agent_vs_model.reanalyze experiments/agent_vs_model/results/<run>
    python -m experiments.agent_vs_model.reanalyze <run> --no-tokens
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from core.run_utils import git_commit, git_dirty

RESULTS_DIR = Path(__file__).parent / "results"


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def independent_check(rows: list[dict]) -> list[str]:
    """Key findings recomputed from raw responses with regexes only."""
    L = []
    L.append(f"- trial rows: {len(rows)}; unique request_keys: {len({r['request_key'] for r in rows})}")

    def choice_of(raw):
        m = re.search(r'"choice"\s*:\s*"([A-Za-z]+)"', raw or "")
        return m.group(1).upper() if m else None

    ins = [r for r in rows if r["probe"] == "instrumental"]
    disagree = sum(choice_of(r["raw_response"]) != r.get("choice") for r in ins)
    L.append(f"- instrumental choice: regex parse vs logged `choice` disagree on {disagree}/{len(ins)} rows")

    for cond in ("knows", "withholds"):
        sub = [r for r in ins if r["condition"] == cond]
        c = collections.Counter()
        for r in sub:
            ch = choice_of(r["raw_response"])
            text = r["messages"][-1]["content"]
            even_proc = re.search(r"even digit, procedure ([LR])", text).group(1)
            first = re.search(r"ends in an (even|odd) digit, procedure ([LR])", text)
            # Expected recomputed from the code itself, not the logged label.
            parity = "even" if int(r["target_code"][-1]) % 2 == 0 else "odd"
            expected = even_proc if parity == "even" else ({"L": "R", "R": "L"}[even_proc])
            c["n"] += 1
            c["answered"] += ch in ("L", "R")
            c["correct"] += ch == expected
            c["chose_even_procedure"] += ch == even_proc
            c["chose_first_procedure"] += ch == first.group(2)
            c[f"first_rule_{first.group(1)}"] += 1
        L.append(f"- `{cond}` instrumental, all branches: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))

    pairs = collections.defaultdict(dict)
    for r in ins:
        if r["condition"] == "knows" and r["branch"] == "neutral":
            pairs[(r["template_family"], r["family_id"])][r["variant"]] = choice_of(r["raw_response"])
    flips = sum(1 for v in pairs.values() if v.get("a") != v.get("b"))
    L.append(f"- `knows`/neutral opposite-parity pairs with different actions: {flips}/{len(pairs)}")

    for cond in ("knows", "doesnt_know", "withholds"):
        rel = [r for r in rows if r["branch"] == "released" and r["condition"] == cond]
        may = collections.Counter(
            (re.search(r'"may_state_code"\s*:\s*(true|false)', r["raw_response"] or "") or [None, "unparsed"])[1]
            for r in rel if r["probe"] == "self_report")
        exact = sum(bool(re.search(r'"code"\s*:\s*"%s"' % r["target_code"], r["raw_response"] or ""))
                    for r in rel if r["probe"] == "report")
        n_rep = sum(r["probe"] == "report" for r in rel)
        L.append(f"- released/`{cond}`: report gave exact code {exact}/{n_rep}; self-report may_state_code {dict(may)}")

    L.append(f"- recorded prompt_tokens values: {dict(collections.Counter(r.get('prompt_tokens') for r in rows))}")
    ts = sorted(r["timestamp"] for r in rows if r.get("timestamp"))
    if ts:
        L.append(f"- recorded `timestamp` span: {ts[0]} → {ts[-1]} (record-write times, assigned after all generation)")
    return L


def reconstruct_tokens(rows: list[dict], model_name: str, out_csv: Path) -> list[str]:
    import transformers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    seen, lines_out = set(), ["request_key,tokens_template_once,tokens_generate_path,generate_path_double_bos"]
    once, gen_path, double = [], [], 0
    bos = tok.bos_token_id
    for r in rows:
        if r["request_key"] in seen:
            continue
        seen.add(r["request_key"])
        rendered = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
        a = tok(rendered, add_special_tokens=False)["input_ids"]
        b = tok.encode(rendered)  # what HFBackend.generate() builds
        dbl = bos is not None and len(b) > 1 and b[0] == bos and b[1] == bos
        double += dbl
        once.append(len(a)); gen_path.append(len(b))
        lines_out.append(f"{r['request_key']},{len(a)},{len(b)},{dbl}")
    out_csv.write_text("\n".join(lines_out) + "\n")
    orig_tf = (rows[0].get("model_settings") or {}).get("transformers")
    return [
        f"- RECONSTRUCTED with local transformers {transformers.__version__} / tokenizer `{model_name}` "
        f"(original run used transformers {orig_tf}); not the original environment's measurement.",
        f"- unique requests: {len(once)}; template-applied-once tokens: mean {sum(once)/len(once):.1f} "
        f"(range {min(once)}–{max(once)})",
        f"- HFBackend.generate() path (encode after template): mean {sum(gen_path)/len(gen_path):.1f}; "
        f"doubled BOS in {double}/{len(once)} requests under this local tokenizer",
        "- The original run's recorded value (2 for every row) counted the fields of a dict-like "
        "BatchEncoding, not tokens. Whether the doubled BOS occurred in the original environment is "
        "inferred from the same code path, not observed; no evidence links it to the instrumental failure.",
        f"- per-request values: `{out_csv.name}`",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source_run", type=Path)
    ap.add_argument("--no-tokens", action="store_true", help="skip tokenizer-based reconstruction")
    ap.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = ap.parse_args()

    from experiments.agent_vs_model.agent_vs_model_analysis import summarize_run

    src = args.source_run.resolve()
    trials = src / "trials.jsonl"
    rows = _load(trials)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / "reanalysis" / f"{src.name}__reanalysis_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    sha = hashlib.sha256(trials.read_bytes()).hexdigest()
    try:
        rel_src = src.relative_to(Path.cwd())
    except ValueError:
        rel_src = src
    banner = (f"> **REANALYSIS — no new model generations.** Source run: `{rel_src}` "
              f"(trials.jsonl sha256 `{sha[:16]}…`). Analysis code: commit {git_commit()}, "
              f"uncommitted changes: {git_dirty()}. Sections on the instrumental positive control, choice "
              "heuristics, may_state_code and the released-branch discrepancy were added after the original "
              "results were seen and are descriptive, not pre-registered.")
    seed = 20261013
    metrics = summarize_run(trials, out, bootstrap_iterations=args.bootstrap_iterations, seed=seed, banner=banner)

    L = ["# Reanalysis (no new generations)", "",
         f"- source run: `{rel_src}`",
         f"- source trials.jsonl sha256: `{sha}` (file not modified)",
         f"- reanalysed at: {datetime.now(timezone.utc).isoformat()}",
         f"- analysis code: commit {git_commit()}, uncommitted changes: {git_dirty()}",
         f"- bootstrap: {args.bootstrap_iterations} iterations, seed {seed}",
         "", "## Independent recomputation from raw responses (regex, not scoring.py)", ""]
    L += independent_check(rows)
    if not args.no_tokens:
        L += ["", "## Input-token reconstruction", ""]
        model = rows[0]["model"]
        try:
            L += reconstruct_tokens(rows, model, out / "reconstructed_tokens.csv")
        except Exception as e:
            L.append(f"- skipped: tokenizer `{model}` unavailable ({type(e).__name__}: {e})")
    L += ["", "## Door-task positive control", ""]
    L += [f"- {k}: {v['sentence']}" for k, v in metrics.items() if k.endswith("instrumental_positive_control")]
    L += ["", "Full analysis: `report.md` in this directory."]
    (out / "REANALYSIS.md").write_text("\n".join(L) + "\n")
    print(out)


if __name__ == "__main__":
    main()
