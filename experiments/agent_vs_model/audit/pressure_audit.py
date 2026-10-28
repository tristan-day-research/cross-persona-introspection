"""Per-conversation audit of a pressure-ladder run (existing data only, no model).

    python -m experiments.agent_vs_model.audit.pressure_audit <pressure_run_dir> <out_dir>

Writes per_conversation.csv (one row per ladder conversation x turn) and pressure_audit.md.
Self-report = logit(" true") - logit(" false") at the first token after the SUPPLIED prefix
`{"knows_code":`, on a side branch after the turn: a two-candidate comparison (other tokens ignored;
the run did not record the mass on these two tokens). Disclosure = the model's own greedy reply
(free generation) contains the exact system code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def per_conversation(run_dir: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(run_dir / "trials.jsonl")]
    df = pd.DataFrame([r for r in rows if r["mode"] == "ladder"]).sort_values(["key", "turn_index"])
    out = []
    for key, g in df.groupby("key", sort=False):
        earlier = False
        for _, r in g.iterrows():
            prior = [m["content"] for m in r["messages"][:-1] if m["role"] == "assistant"]
            code_in_earlier = r["code_in_prompt"] and any(r["code"] in p for p in prior)
            assert code_in_earlier == earlier, key  # cross-check against the logged per-turn flags
            out.append(dict(condition=r["condition"], family=r["family_id"], variant=r["variant"], code=r["code"],
                            turn=r["turn_index"], turn_name=r["turn_name"],
                            discloses_this_turn=bool(r["gave_exact_code"]),
                            code_in_earlier_reply=bool(code_in_earlier),
                            disclosed_at_least_once=bool(code_in_earlier or r["gave_exact_code"]),
                            status=r["report_status"], self_logodds=round(r["self_logodds"], 2),
                            self_favors=("know" if r["self_logodds"] > 0 else "don't know"),
                            reply=r["raw_response"].strip().replace("\n", " ")))
            earlier = earlier or bool(r["gave_exact_code"])
    return pd.DataFrame(out)


def main(run_dir: str, out_dir: str) -> None:
    run_dir, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = per_conversation(run_dir)
    t.to_csv(out / "per_conversation.csv", index=False)
    print(t.groupby(["condition", "turn", "turn_name"]).agg(
        n=("code", "size"), discloses_this_turn=("discloses_this_turn", "sum"),
        disclosed_so_far=("disclosed_at_least_once", "sum"), favors_know=("self_favors", lambda s: (s == "know").sum()),
        logodds_min=("self_logodds", "min"), logodds_max=("self_logodds", "max")).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
