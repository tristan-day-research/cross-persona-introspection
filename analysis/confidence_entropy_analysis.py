"""Analysis and visualization for the confidence-vs-entropy experiment.

Core plots:
1. Entropy vs reported confidence (scatter, colored by persona)
2. Entropy distribution by domain (box plot, hued by persona)
3. Accuracy vs confidence calibration curve
4. Persona entropy shift relative to default_assistant

Usage:
    python analysis/confidence_entropy_analysis.py results/raw/confidence_entropy_*.jsonl

Scientific cautions:
- Entropy here is computed over answer options only (A/B/C/D), not full vocabulary.
- Reported confidence can be distorted by persona style; that is part of what we test.
- Domain personas (Chemist, Historian, Artist) and style personas (Cautious, Bold)
  should be compared separately.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ce_results(path: str | Path) -> pd.DataFrame:
    """Load confidence-entropy JSONL results into a DataFrame."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)

    # Filter to valid (non-error) trials
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()

    return df


def plot_entropy_vs_confidence_by_persona(
    df: pd.DataFrame, output_path: str | None = None
) -> None:
    """Scatter plot: x=reported confidence, y=entropy, colored by persona.

    Key interpretation:
    - If confidence accurately tracks entropy (high conf → low entropy), persona has
      genuine self-knowledge.
    - If persona reports low confidence but entropy is also low, the persona is feigning
      uncertainty (surface-level style effect).
    - If persona reports high confidence but entropy is high, it's overconfident.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    personas = df["persona_name"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(personas)))

    for persona, color in zip(personas, colors):
        mask = df["persona_name"] == persona
        subset = df[mask]
        valid = subset["stated_confidence_midpoint"].notna()
        ax.scatter(
            subset.loc[valid, "stated_confidence_midpoint"],
            subset.loc[valid, "answer_option_entropy"],
            label=persona,
            color=color,
            alpha=0.6,
            s=30,
        )

    ax.set_xlabel("Stated Confidence (midpoint probability)")
    ax.set_ylabel("Answer Option Entropy (from logprobs)")
    ax.set_title("Stated Confidence vs Actual Entropy by Persona")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    else:
        plt.show()


def plot_entropy_distribution_by_domain(
    df: pd.DataFrame, output_path: str | None = None
) -> None:
    """Box plot: x=domain, y=entropy, grouped by persona.

    Shows whether domain expertise (e.g., Chemist on chemistry) actually
    reduces model uncertainty or just changes the style of the response.
    """
    domains = sorted(df["domain"].unique())
    personas = sorted(df["persona_name"].unique())

    fig, axes = plt.subplots(1, len(domains), figsize=(4 * len(domains), 6), sharey=True)
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        domain_df = df[df["domain"] == domain]
        data = [
            domain_df[domain_df["persona_name"] == p]["answer_option_entropy"].dropna()
            for p in personas
        ]
        bp = ax.boxplot(data, labels=personas, patch_artist=True)

        colors = plt.cm.tab10(np.linspace(0, 1, len(personas)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(f"Domain: {domain}")
        ax.set_ylabel("Answer Option Entropy" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Entropy Distribution by Domain and Persona", fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    else:
        plt.show()


def plot_accuracy_vs_confidence_calibration(
    df: pd.DataFrame, output_path: str | None = None
) -> None:
    """Calibration plot: bin by reported confidence, plot actual accuracy per bin.

    A well-calibrated model should follow the diagonal.
    Persona-specific deviations show where style overrides substance.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    personas = sorted(df["persona_name"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(personas)))
    bins = np.linspace(0, 1, 11)  # 10 bins

    for persona, color in zip(personas, colors):
        subset = df[
            (df["persona_name"] == persona)
            & df["stated_confidence_midpoint"].notna()
            & df["is_correct"].notna()
        ]
        if subset.empty:
            continue

        bin_indices = np.digitize(subset["stated_confidence_midpoint"], bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

        bin_centers = []
        bin_accuracies = []
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if mask.sum() >= 2:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_accuracies.append(subset.loc[mask.values, "is_correct"].mean())

        if bin_centers:
            ax.plot(bin_centers, bin_accuracies, "o-", color=color, label=persona, alpha=0.8)

    ax.set_xlabel("Stated Confidence (binned)")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title("Confidence Calibration by Persona")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    else:
        plt.show()


def compute_persona_entropy_shift(df: pd.DataFrame) -> pd.DataFrame:
    """For each question, compute entropy delta relative to default_assistant.

    Returns a DataFrame with columns: question_id, persona_name, domain,
    entropy_default, entropy_persona, entropy_delta.

    Positive delta = persona has MORE uncertainty than default.
    Negative delta = persona has LESS uncertainty (more certain).
    """
    default_df = df[df["persona_name"] == "default_assistant"][
        ["question_id", "answer_option_entropy"]
    ].rename(columns={"answer_option_entropy": "entropy_default"})

    other_df = df[df["persona_name"] != "default_assistant"][
        ["question_id", "persona_name", "domain", "answer_option_entropy"]
    ].rename(columns={"answer_option_entropy": "entropy_persona"})

    merged = other_df.merge(default_df, on="question_id", how="inner")
    merged["entropy_delta"] = merged["entropy_persona"] - merged["entropy_default"]

    return merged


def plot_entropy_shift(
    df: pd.DataFrame, output_path: str | None = None
) -> None:
    """Bar plot of mean entropy delta by persona, optionally split by domain."""
    shift_df = compute_persona_entropy_shift(df)

    if shift_df.empty:
        logger.warning("No entropy shift data (need default_assistant trials)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Overall mean shift by persona
    persona_means = shift_df.groupby("persona_name")["entropy_delta"].mean().sort_values()
    colors = ["tab:red" if v > 0 else "tab:blue" for v in persona_means.values]
    ax1.barh(persona_means.index, persona_means.values, color=colors, alpha=0.7)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Mean Entropy Delta vs Default Assistant")
    ax1.set_title("Overall Entropy Shift by Persona")
    ax1.grid(True, alpha=0.3, axis="x")

    # Per-domain shift for domain personas
    domain_personas = ["chemist", "historian", "artist"]
    domain_data = shift_df[shift_df["persona_name"].isin(domain_personas)]
    if not domain_data.empty:
        pivot = domain_data.groupby(["persona_name", "domain"])["entropy_delta"].mean().unstack()
        pivot.plot(kind="bar", ax=ax2, alpha=0.7)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Persona")
        ax2.set_ylabel("Mean Entropy Delta")
        ax2.set_title("Domain Expert Entropy Shift by Domain")
        ax2.legend(title="Domain")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "No domain persona data", ha="center", va="center")
        ax2.set_title("Domain Expert Entropy Shift by Domain")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {output_path}")
    else:
        plt.show()


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a concise summary table to stdout."""
    print("\n" + "=" * 80)
    print("CONFIDENCE vs ENTROPY — SUMMARY")
    print("=" * 80)

    for persona in sorted(df["persona_name"].unique()):
        subset = df[df["persona_name"] == persona]
        n = len(subset)
        acc = subset["is_correct"].mean() if subset["is_correct"].notna().any() else float("nan")
        conf = subset["stated_confidence_midpoint"].mean() if subset["stated_confidence_midpoint"].notna().any() else float("nan")
        ent = subset["answer_option_entropy"].mean()
        margin = subset["margin_between_top_two"].mean() if subset["margin_between_top_two"].notna().any() else float("nan")

        print(f"\n  {persona} (n={n})")
        print(f"    Accuracy:              {acc:.3f}")
        print(f"    Stated Confidence:     {conf:.3f}")
        print(f"    Mean Entropy:          {ent:.3f}")
        print(f"    Mean Top-2 Margin:     {margin:.3f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze confidence-vs-entropy results")
    parser.add_argument("results_path", help="Path to JSONL results file")
    parser.add_argument("--output-dir", default="results/processed", help="Directory for plots")
    parser.add_argument("--no-show", action="store_true", help="Save plots without displaying")
    args = parser.parse_args()

    df = load_ce_results(args.results_path)
    if df.empty:
        print("No valid results found.")
        sys.exit(1)

    print(f"Loaded {len(df)} valid trials")
    print_summary_table(df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save = args.no_show

    plot_entropy_vs_confidence_by_persona(
        df, output_path=str(output_dir / "entropy_vs_confidence.png") if save else None
    )
    plot_entropy_distribution_by_domain(
        df, output_path=str(output_dir / "entropy_by_domain.png") if save else None
    )
    plot_accuracy_vs_confidence_calibration(
        df, output_path=str(output_dir / "calibration.png") if save else None
    )
    plot_entropy_shift(
        df, output_path=str(output_dir / "entropy_shift.png") if save else None
    )


if __name__ == "__main__":
    main()
