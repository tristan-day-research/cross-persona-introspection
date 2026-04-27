"""Summary analysis for the persona self-recognition experiment.

Reads a trials JSONL produced by `PersonaSelfRecognition` and writes:
  - individual_matrix.csv  : source × evaluator individual-recognition accuracy
  - paired_matrix.csv      : (source = evaluator) × paired-partner accuracy
  - summary.md             : human-readable run summary
  - heatmap_individual.png : individual matrix heatmap (if matplotlib is available)
  - heatmap_paired.png     : paired matrix heatmap (if matplotlib is available)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from core.results_logger import load_results

logger = logging.getLogger(__name__)


def summarize_run(jsonl_path: str | Path, run_dir: str | Path) -> dict:
    """Compute matrices, write CSVs / heatmap / markdown. Return a metric dict."""
    jsonl_path = Path(jsonl_path)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(jsonl_path)
    if df.empty:
        return {"n_trials": 0}

    metrics: dict = {"n_trials": len(df)}

    # ── Individual matrix ────────────────────────────────────────────────
    ind = df[(df["phase"] == "individual") & (df["error"].isna())]
    if not ind.empty:
        ind_matrix = (
            ind.pivot_table(
                index="source_persona",
                columns="evaluator_persona",
                values="is_correct",
                aggfunc="mean",
            )
        )
        ind_matrix.to_csv(run_dir / "individual_matrix.csv")
        diag, off = _diag_offdiag_means(ind_matrix)
        metrics["individual"] = {
            "n_trials": int(len(ind)),
            "overall_accuracy": float(ind["is_correct"].mean()),
            "diagonal_mean": diag,
            "off_diagonal_mean": off,
            "diagonal_advantage": (diag - off) if (diag is not None and off is not None) else None,
        }
        _save_heatmap(ind_matrix, run_dir / "heatmap_individual.png", "Individual recognition accuracy")

    # ── Paired matrix ────────────────────────────────────────────────────
    paired = df[(df["phase"] == "paired") & (df["error"].isna()) & (df["has_ground_truth"] == True)]
    if not paired.empty:
        # source_persona is set to the evaluator (the persona whose authorship is being recovered).
        # Index by that, columns = the *other* persona in the pair.
        paired = paired.copy()
        paired["partner"] = paired.apply(
            lambda r: r["candidate_b_source"] if r["candidate_a_source"] == r["source_persona"] else r["candidate_a_source"],
            axis=1,
        )
        paired_matrix = paired.pivot_table(
            index="source_persona",
            columns="partner",
            values="is_correct",
            aggfunc="mean",
        )
        paired_matrix.to_csv(run_dir / "paired_matrix.csv")
        metrics["paired"] = {
            "n_trials": int(len(paired)),
            "overall_accuracy": float(paired["is_correct"].mean()),
        }
        _save_heatmap(paired_matrix, run_dir / "heatmap_paired.png", "Paired recognition accuracy (own vs partner)")

    # ── Generation summary ───────────────────────────────────────────────
    gen = df[(df["phase"] == "generation") & (df["error"].isna())]
    if not gen.empty:
        metrics["generation"] = {
            "n_trials": int(len(gen)),
            "mean_token_length": float(gen["token_length"].mean()),
            "by_persona_mean_length": gen.groupby("source_persona")["token_length"].mean().to_dict(),
        }

    _write_markdown(run_dir / "summary.md", metrics, jsonl_path)
    return metrics


def _diag_offdiag_means(matrix: pd.DataFrame) -> tuple[float | None, float | None]:
    common = [p for p in matrix.index if p in matrix.columns]
    if not common:
        return None, None
    diag_vals, off_vals = [], []
    for s in common:
        for e in common:
            v = matrix.loc[s, e]
            if pd.isna(v):
                continue
            (diag_vals if s == e else off_vals).append(float(v))
    diag = sum(diag_vals) / len(diag_vals) if diag_vals else None
    off = sum(off_vals) / len(off_vals) if off_vals else None
    return diag, off


def _save_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping heatmap")
        return

    import numpy as np
    fig, ax = plt.subplots(figsize=(1.4 * len(matrix.columns) + 2, 1.0 * len(matrix.index) + 1.5))
    values = matrix.astype(float).values
    im = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("evaluator persona")
    ax.set_ylabel("source persona")
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_markdown(path: Path, metrics: dict, jsonl_path: Path) -> None:
    lines = [
        "# Persona self-recognition — run summary",
        "",
        f"Trials file: `{jsonl_path}`",
        f"Total rows: {metrics.get('n_trials', 0)}",
        "",
    ]
    if "generation" in metrics:
        g = metrics["generation"]
        lines += [
            "## Generation",
            f"- n: {g['n_trials']}",
            f"- mean token length: {g['mean_token_length']:.1f}",
            "- mean length by persona:",
        ]
        for p, v in g["by_persona_mean_length"].items():
            lines.append(f"  - {p}: {v:.1f}")
        lines.append("")
    if "individual" in metrics:
        ind = metrics["individual"]
        lines += [
            "## Individual recognition",
            f"- n: {ind['n_trials']}",
            f"- overall accuracy: {ind['overall_accuracy']:.3f}",
            f"- diagonal mean (source == evaluator): "
            + (f"{ind['diagonal_mean']:.3f}" if ind['diagonal_mean'] is not None else "n/a"),
            f"- off-diagonal mean: "
            + (f"{ind['off_diagonal_mean']:.3f}" if ind['off_diagonal_mean'] is not None else "n/a"),
            f"- diagonal advantage: "
            + (f"{ind['diagonal_advantage']:+.3f}" if ind['diagonal_advantage'] is not None else "n/a"),
            "",
            "Matrix CSV: `individual_matrix.csv`",
            "Heatmap: `heatmap_individual.png`",
            "",
        ]
    if "paired" in metrics:
        p = metrics["paired"]
        lines += [
            "## Paired recognition",
            f"- n (with ground truth, evaluator in pair): {p['n_trials']}",
            f"- overall accuracy: {p['overall_accuracy']:.3f}",
            "",
            "Matrix CSV: `paired_matrix.csv`",
            "Heatmap: `heatmap_paired.png`",
            "",
        ]
    lines += [
        "## Caveats",
        "- Behavioral skeleton only; no claim of privileged access.",
        "- A diagonal advantage is consistent with — but does not establish — self-recognition beyond stylistic matching.",
        "- See module TODOs for paraphrase / mechanistic follow-ups.",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    """CLI: python -m experiments.persona_self_recognition.analysis <trials.jsonl>"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path")
    parser.add_argument("--run-dir", default=None, help="Output dir (defaults to trials.jsonl's parent)")
    args = parser.parse_args()
    jsonl_path = Path(args.jsonl_path)
    run_dir = Path(args.run_dir) if args.run_dir else jsonl_path.parent
    metrics = summarize_run(jsonl_path, run_dir)
    print(metrics)


if __name__ == "__main__":
    main()
