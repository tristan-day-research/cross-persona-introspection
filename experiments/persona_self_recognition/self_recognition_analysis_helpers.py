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

    # Human-readable model label, recovered from the run-dir name (which encodes
    # the short model slug). Stamped onto each heatmap title so the saved PNGs
    # are self-identifying. Import lazily to avoid a heavy import at module load.
    from experiments.persona_self_recognition.self_recognition_experiment import (
        label_from_run_dir,
    )
    model_label = label_from_run_dir(run_dir.name)

    df = load_results(jsonl_path)
    if df.empty:
        return {"n_trials": 0}

    metrics: dict = {"n_trials": len(df)}

    # Detect run-level POV: present in any row's `source_pov` field. Older runs
    # without the field are treated as 1st_person.
    run_pov = None
    if "source_pov" in df.columns:
        povs = df["source_pov"].dropna().unique().tolist()
        run_pov = "3rd_person" if "3rd_person" in povs else ("1st_person" if "1st_person" in povs else None)
    metrics["source_pov"] = run_pov

    # Helper: keep only diagonal (1st-person) rows when target_persona is present.
    # Used so the source × evaluator matrices remain interpretable in 3rd_person
    # runs — off-diagonal rows add a target dimension that doesn't fit a 2D matrix.
    def _diagonal_only(frame):
        if "target_persona" not in frame.columns:
            return frame
        tgt = frame["target_persona"]
        return frame[tgt.isna() | (tgt == frame["source_persona"])]

    # ── Individual matrix ────────────────────────────────────────────────
    ind = df[(df["phase"] == "individual") & (df["error"].isna())]
    ind = _diagonal_only(ind)
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
        _save_heatmap(ind_matrix, run_dir / "heatmap_individual.png", "Individual recognition accuracy", model_label)

    # ── Paired matrix ────────────────────────────────────────────────────
    paired = df[(df["phase"] == "paired") & (df["error"].isna()) & (df["has_ground_truth"] == True)]
    # In 3rd_person runs, restrict to 1st-person candidates on both sides so the
    # source × partner matrix stays interpretable (drops 3rd-person rows).
    if "candidate_a_target" in paired.columns and "candidate_b_target" in paired.columns:
        ca_diag = paired["candidate_a_target"].isna() | (paired["candidate_a_target"] == paired["candidate_a_source"])
        cb_diag = paired["candidate_b_target"].isna() | (paired["candidate_b_target"] == paired["candidate_b_source"])
        paired = paired[ca_diag & cb_diag]
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
        _save_heatmap(paired_matrix, run_dir / "heatmap_paired.png", "Paired recognition accuracy (own vs partner)", model_label)

    # ── Generation summary ───────────────────────────────────────────────
    gen = df[(df["phase"] == "generation") & (df["error"].isna())]
    if not gen.empty:
        gen_metrics: dict = {
            "n_trials": int(len(gen)),
            "mean_token_length": float(gen["token_length"].mean()),
            "by_persona_mean_length": gen.groupby("source_persona")["token_length"].mean().to_dict(),
        }
        if "source_pov" in gen.columns and gen["source_pov"].notna().any():
            gen_metrics["n_by_pov"] = gen["source_pov"].value_counts().to_dict()
        metrics["generation"] = gen_metrics

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


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    *,
    xlabel: str = "evaluator persona",
    ylabel: str = "source persona",
    vmin: float = 0.0,
    vmax: float = 1.0,
    ax=None,
):
    """Render a labelled 0–1 heatmap. Returns the axes for further customization."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(
            figsize=(1.4 * len(matrix.columns) + 2, 1.0 * len(matrix.index) + 1.5)
        )
    values = matrix.astype(float).values
    im = ax.imshow(values, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isnan(v):
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < (vmin + vmax) / 2 else "black", fontsize=9,
                )
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.figure.tight_layout()
    return ax


def plot_choice_bars(
    matrix: pd.DataFrame,
    title: str,
    *,
    ylabel: str = "count",
    ax=None,
    ylim: tuple[float, float] | None = None,
    hline: float | None = None,
):
    """Grouped bar chart from a wide DataFrame (rows = group, cols = choices).

    `hline` draws a horizontal reference line (e.g. 0.5 for chance).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.2 * len(matrix) + 2), 4))
    matrix.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if hline is not None:
        ax.axhline(hline, color="red", linestyle="--", linewidth=1, alpha=0.6)
    legend_title = matrix.columns.name if matrix.columns.name else None
    ax.legend(title=legend_title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.figure.tight_layout()
    return ax


def _save_heatmap(matrix: pd.DataFrame, path: Path, title: str, model_label: str | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping heatmap")
        return
    full_title = f"{title}\n{model_label}" if model_label else title
    ax = plot_heatmap(matrix, full_title)
    ax.figure.savefig(path, dpi=140)
    plt.close(ax.figure)


def _write_markdown(path: Path, metrics: dict, jsonl_path: Path) -> None:
    lines = [
        "# Persona self-recognition — run summary",
        "",
        f"Trials file: `{jsonl_path}`",
        f"Total rows: {metrics.get('n_trials', 0)}",
        f"source_pov: {metrics.get('source_pov') or '1st_person (legacy / unrecorded)'}",
        "",
    ]
    if metrics.get("source_pov") == "3rd_person":
        lines += [
            "_Matrices below are restricted to 1st-person trials "
            "(target_persona == source_persona). Off-diagonal 3rd-person "
            "trials are in `trials.jsonl` for downstream analysis._",
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
    """CLI: python -m experiments.persona_self_recognition.self_recognition_analysis_helpers <trials.jsonl>"""
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
