"""Aggregation + figures for the Panickssery et al. (2024) replication.

Reproduces the paper's headline out-of-the-box numbers for a single evaluator:
  - pairwise self-recognition / self-preference per alternative source (Table 7,
    Fig 2-left / Fig 4-left): the order-averaged confidence in the evaluator's
    OWN summary, already computed per-trial in `score`.
  - individual self-recognition: mean renormalized P("Yes") per source — the
    "raw self-recognition scores (individual)" of Fig 3 / Tables 11,13.
  - individual self-preference: mean prob-weighted Likert rating per source.

`summarize_run` writes pairwise_scores.csv, individual_scores.csv, summary.md,
and bar figures, and returns a metrics dict. summary.md prints the paper's
published Llama-2-7b values alongside the observed ones for a direct check.
"""

from pathlib import Path

import pandas as pd

from core.results_logger import load_results

# Published Llama-2-7b "No Fine-Tuning" reference values, for side-by-side check.
#   Pairwise: Table 7.  Individual recognition (raw P(Yes)): Fig 3 (xsum+cnn agg).
PAPER_REF = {
    "pairwise": {
        "xsum": {"self_recognition": 0.514, "self_preference": 0.511},
        "cnn": {"self_recognition": 0.505, "self_preference": 0.505},
    },
    # Fig 3 raw individual-recognition confidence, Llama-2 evaluator row.
    "individual_recognition_raw_fig3": {
        "llama": 0.725, "gpt35": 0.730, "gpt4": 0.715, "human": 0.726,
    },
}


def _mean_by(df: pd.DataFrame, phase: str, group_col: str) -> dict[str, float]:
    sub = df[(df["phase"] == phase) & df["score"].notna()]
    if sub.empty or group_col not in sub.columns:
        return {}
    return {str(k): float(v) for k, v in sub.groupby(group_col)["score"].mean().items()}


def summarize_run(jsonl_path: str | Path, run_dir: str | Path) -> dict:
    jsonl_path, run_dir = Path(jsonl_path), Path(run_dir)
    df = load_results(jsonl_path)
    if df.empty:
        return {"error": "no trials"}

    dataset = str(df["dataset"].iloc[0])
    evaluator = str(df["evaluator"].iloc[0])
    model = str(df["model"].iloc[0])

    # Pairwise: score is the evaluator's confidence in its OWN summary.
    self_recognition = _mean_by(df, "detection", "other_source")   # vs each alternative
    self_preference = _mean_by(df, "comparison", "other_source")
    # Individual: mean per source shown.
    indiv_recognition = _mean_by(df, "recognition", "target_source")  # P(Yes)
    indiv_scoring = _mean_by(df, "scoring", "target_source")           # Likert mean

    metrics = {
        "model": model,
        "dataset": dataset,
        "evaluator": evaluator,
        "pairwise_self_recognition_by_alternative": self_recognition,
        "pairwise_self_recognition_overall": _overall(self_recognition),
        "pairwise_self_preference_by_alternative": self_preference,
        "pairwise_self_preference_overall": _overall(self_preference),
        "individual_recognition_pYes_by_source": indiv_recognition,
        "individual_scoring_likert_by_source": indiv_scoring,
    }

    _write_csvs(run_dir, self_recognition, self_preference, indiv_recognition, indiv_scoring)
    _write_figures(run_dir, dataset, model, self_recognition, self_preference,
                   indiv_recognition, indiv_scoring)
    _write_markdown(run_dir / "summary.md", metrics)
    return metrics


def _overall(d: dict[str, float]) -> float | None:
    return float(sum(d.values()) / len(d)) if d else None


def _write_csvs(run_dir, self_rec, self_pref, indiv_rec, indiv_score) -> None:
    pw = pd.DataFrame({"self_recognition": pd.Series(self_rec),
                       "self_preference": pd.Series(self_pref)})
    pw.index.name = "alternative_source"
    pw.to_csv(run_dir / "pairwise_scores.csv")

    iv = pd.DataFrame({"recognition_pYes": pd.Series(indiv_rec),
                       "scoring_likert": pd.Series(indiv_score)})
    iv.index.name = "source"
    iv.to_csv(run_dir / "individual_scores.csv")


def _write_figures(run_dir, dataset, model, self_rec, self_pref, indiv_rec, indiv_score) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    panels = [
        ("Pairwise self-recognition\n(conf. in own summary, vs alternative)", self_rec, 0.5),
        ("Pairwise self-preference\n(pref. for own summary, vs alternative)", self_pref, 0.5),
        ("Individual recognition\nmean P(Yes) by source", indiv_rec, 0.5),
        ("Individual scoring\nmean Likert by source", indiv_score, None),
    ]
    panels = [p for p in panels if p[1]]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
    for ax, (title, data, ref) in zip(axes[0], panels):
        labels = list(data.keys())
        ax.bar(labels, [data[k] for k in labels], color="#4C72B0")
        if ref is not None:
            ax.axhline(ref, ls="--", c="gray", lw=1)
            ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle(f"{model}  —  {dataset}", fontsize=10)
    fig.tight_layout()
    fig.savefig(run_dir / "scores.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(path: Path, m: dict) -> None:
    ds = m["dataset"]
    ref_pw = PAPER_REF["pairwise"].get(ds, {})
    ref_fig3 = PAPER_REF["individual_recognition_raw_fig3"]
    lines = [
        f"# Paper replication — {m['model']}",
        "",
        f"- dataset: **{ds}**",
        f"- evaluator self-source: **{m['evaluator']}**",
        "",
        "Faithful out-of-the-box replication of Panickssery et al. (2024), §2.",
        "Pairwise scores are ordering-bias-corrected (each pair run in both orders).",
        "",
        "## Pairwise (Table 7 / Figs 2,4)",
        "",
        f"- self-recognition (overall): **{_fmt(m['pairwise_self_recognition_overall'])}**"
        + (f"  · paper Llama-2-7b: {ref_pw.get('self_recognition')}" if ref_pw else ""),
        f"- self-preference (overall): **{_fmt(m['pairwise_self_preference_overall'])}**"
        + (f"  · paper Llama-2-7b: {ref_pw.get('self_preference')}" if ref_pw else ""),
        "",
        "By alternative source (recognition / preference):",
    ]
    rec = m["pairwise_self_recognition_by_alternative"]
    pref = m["pairwise_self_preference_by_alternative"]
    for src in sorted(set(rec) | set(pref)):
        lines.append(f"  - {src}: {_fmt(rec.get(src))} / {_fmt(pref.get(src))}")
    lines += [
        "",
        "## Individual (Fig 3 / Tables 11-14)",
        "",
        "mean P(Yes) by source  ·  paper Fig-3 Llama-2 raw value:",
    ]
    for src, v in sorted(m["individual_recognition_pYes_by_source"].items()):
        ref = f"  · paper: {ref_fig3[src]}" if src in ref_fig3 else ""
        lines.append(f"  - {src}: {_fmt(v)}{ref}")
    lines += ["", "mean Likert score by source:"]
    for src, v in sorted(m["individual_scoring_likert_by_source"].items()):
        lines.append(f"  - {src}: {_fmt(v)}")
    lines += [
        "",
        "## Expected qualitative result",
        "",
        "Per the paper, Llama-2-7b out-of-the-box has weak self-recognition: it is",
        "near chance (~0.5) at distinguishing itself from the other LLMs, and only",
        "Human is reliably distinguishable. A sign that the *machinery* works is the",
        "Human column standing out from the model columns, not exact value matches.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"
