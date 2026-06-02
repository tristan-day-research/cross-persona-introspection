"""Response-bias analysis for the persona self-recognition experiment.

The matrices in `self_recognition_analysis_helpers` answer *whether* recognition
accuracy is above chance. This module is about a different question: how much of
any apparent recognition is genuine self-knowledge versus a **response-format
bias** (a global YES/NO lean, an A/B position preference, mis-calibrated
self-probabilities). It provides the summaries and plots for the three modes:

  individual (YES/NO)            — yes-rate-by-evaluator diagnostic; optional
                                   logprob_yes - logprob_no margin (self vs nonself).
  individual_probability (0-100) — self-vs-nonself score distributions, AUROC
                                   (overall and per evaluator), calibration bins,
                                   parse-failure rate.
  paired (A/B, counterbalanced)  — P(chose_self), P(choose A/B) position bias,
                                   accuracy split by position_of_self, optional
                                   confidence (correct vs incorrect), parse-fail rate.

Backward compatibility: every function tolerates result files that pre-date the
new columns. `ensure_bias_columns` adds any missing column as NaN/None, and the
self/nonself label falls back to (source_persona == evaluator_persona) when the
explicit `is_self` field is absent. Functions return plain DataFrames / dicts
(testable without a display); the `plot_*` helpers draw onto a Matplotlib axis.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns introduced by the response-bias work. Absent in older result files.
BIAS_COLUMNS = (
    "model_answer", "parsed_probability", "parsed_confidence", "is_self",
    "correct_answer", "position_of_self", "chose_self",
    "logprob_yes", "logprob_no", "generated_text_id",
    "prob_yes", "prob_no", "prob_a", "prob_b",
)


# ── Backward-compat helpers ────────────────────────────────────────────────

def ensure_bias_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with every BIAS_COLUMN present (NaN where missing).

    Lets analysis written against the new schema run unchanged on old result
    files instead of raising KeyError on a missing column.
    """
    df = df.copy()
    for col in BIAS_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _coalesce(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    """primary's non-null values, filled by fallback where primary is null.

    Done via explicit object-dtype masked assignment (not .where / combine_first)
    to avoid pandas dtype-inference / empty-concat FutureWarnings when one input
    is an all-NaN column and the other is bool — common with legacy result files.
    """
    out = primary.astype(object).copy()
    mask = out.isna()
    if mask.any():
        out.loc[mask] = fallback.reindex(out.index).loc[mask]
    return out


def effective_is_self(df: pd.DataFrame) -> pd.Series:
    """Self/nonself ground-truth label per row.

    Uses the stored `is_self` where present; otherwise falls back to
    (source_persona == evaluator_persona), which is the correct definition for
    individual / individual_probability rows (self == the inducing persona).
    Returns a boolean Series (object dtype with NaN where undefined).
    """
    if "is_self" in df.columns:
        stored = df["is_self"]
    else:
        stored = pd.Series(np.nan, index=df.index)
    if {"source_persona", "evaluator_persona"} <= set(df.columns):
        derived = df["source_persona"] == df["evaluator_persona"]
    else:
        derived = pd.Series(np.nan, index=df.index)
    # Prefer stored where it is a real bool; else derived.
    return _coalesce(stored, derived)


def _answer_col(df: pd.DataFrame) -> pd.Series:
    """Parsed discrete answer, coalescing model_answer over legacy parsed_choice."""
    ma = df["model_answer"] if "model_answer" in df.columns else pd.Series(np.nan, index=df.index)
    pc = df["parsed_choice"] if "parsed_choice" in df.columns else pd.Series(np.nan, index=df.index)
    return _coalesce(ma, pc)


def _phase_frame(df: pd.DataFrame, phase: str, drop_errors: bool = True) -> pd.DataFrame:
    """Subset to one phase, optionally dropping error rows, with bias columns ensured."""
    df = ensure_bias_columns(df)
    if "phase" in df.columns:
        df = df[df["phase"] == phase]
    if drop_errors and "error" in df.columns:
        df = df[df["error"].isna()]
    return df.copy()


# ── AUROC (manual, tie-aware — no sklearn dependency) ──────────────────────

def auroc(scores: Sequence[float], labels: Sequence) -> float:
    """Area under the ROC curve for separating positive (1) from negative (0).

    Rank-based (Mann-Whitney U) so it is tie-aware and needs no threshold sweep.
    NaN scores/labels are dropped. Returns NaN when either class is empty.
    0.5 = no separation; >0.5 = higher scores track the positive class.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    mask = ~(np.isnan(s) | np.isnan(y))
    s, y = s[mask], y[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0  # 1-based average rank over ties
        i = j
    rank_pos_sum = ranks[y == 1].sum()
    return float((rank_pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ── individual_probability ─────────────────────────────────────────────────

def individual_probability_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Successful individual_probability rows, with `is_self` resolved (bool)."""
    sub = _phase_frame(df, "individual_probability")
    sub["is_self"] = effective_is_self(sub)
    return sub


def probability_parse_failure_rate(df: pd.DataFrame) -> dict:
    """Parse-failure stats for the numeric self-probability mode."""
    sub = individual_probability_frame(df)
    n = len(sub)
    failures = int(sub["parsed_probability"].isna().sum())
    return {
        "n_trials": n,
        "n_parse_failures": failures,
        "parse_failure_rate": (failures / n) if n else float("nan"),
    }


def mean_probability_by_evaluator(df: pd.DataFrame) -> pd.DataFrame:
    """Mean parsed_probability per evaluator × self/nonself, with trial counts.

    Columns: self_mean, nonself_mean, self_minus_nonself, n_self, n_nonself.
    A positive self_minus_nonself is the probability-mode self-recognition signal.
    """
    sub = individual_probability_frame(df).dropna(subset=["parsed_probability"])
    if sub.empty:
        return pd.DataFrame(
            columns=["self_mean", "nonself_mean", "self_minus_nonself", "n_self", "n_nonself"]
        )
    g = sub.groupby(["evaluator_persona", sub["is_self"].astype(bool)])["parsed_probability"]
    means = g.mean().unstack(fill_value=np.nan)
    counts = g.size().unstack(fill_value=0)
    out = pd.DataFrame(index=means.index)
    out["self_mean"] = means.get(True)
    out["nonself_mean"] = means.get(False)
    out["self_minus_nonself"] = out["self_mean"] - out["nonself_mean"]
    out["n_self"] = counts.get(True, 0)
    out["n_nonself"] = counts.get(False, 0)
    return out


def auroc_self_vs_nonself(df: pd.DataFrame) -> float:
    """Overall AUROC for separating self from nonself using parsed_probability."""
    sub = individual_probability_frame(df)
    return auroc(sub["parsed_probability"], sub["is_self"].astype(float))


def auroc_by_evaluator(df: pd.DataFrame, min_per_class: int = 5) -> pd.DataFrame:
    """Per-evaluator AUROC (self vs nonself by parsed_probability).

    AUROC is NaN for an evaluator without at least `min_per_class` scored trials
    in each class — too few to be meaningful. Columns: auroc, n_self, n_nonself.
    """
    sub = individual_probability_frame(df).dropna(subset=["parsed_probability"])
    rows = []
    for evaluator, grp in sub.groupby("evaluator_persona"):
        labels = grp["is_self"].astype(float)
        n_self = int((labels == 1).sum())
        n_nonself = int((labels == 0).sum())
        score = (
            auroc(grp["parsed_probability"], labels)
            if (n_self >= min_per_class and n_nonself >= min_per_class)
            else float("nan")
        )
        rows.append({"evaluator_persona": evaluator, "auroc": score,
                     "n_self": n_self, "n_nonself": n_nonself})
    return pd.DataFrame(rows).set_index("evaluator_persona") if rows else pd.DataFrame(
        columns=["auroc", "n_self", "n_nonself"]
    )


def calibration_table(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Binned predicted self-probability vs empirical self rate.

    Predicted probability = parsed_probability / 100, bucketed into `bins` equal
    bins over [0, 1]. Per bin: mean predicted prob, empirical self rate, count.
    A well-calibrated model has empirical_self_rate ≈ mean_predicted.
    """
    sub = individual_probability_frame(df).dropna(subset=["parsed_probability"]).copy()
    cols = ["bin", "mean_predicted", "empirical_self_rate", "n"]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    sub["pred"] = sub["parsed_probability"].astype(float) / 100.0
    sub["is_self"] = sub["is_self"].astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # include_lowest so pred==0 lands in the first bin; right-closed bins.
    sub["bin"] = pd.cut(sub["pred"], bins=edges, include_lowest=True)
    rows = []
    for interval, grp in sub.groupby("bin", observed=True):
        rows.append({
            "bin": str(interval),
            "mean_predicted": float(grp["pred"].mean()),
            "empirical_self_rate": float(grp["is_self"].mean()),
            "n": int(len(grp)),
        })
    return pd.DataFrame(rows, columns=cols)


def plot_probability_distributions(df: pd.DataFrame, *, bins: int = 20, ax=None):
    """Overlaid histograms of parsed_probability for self vs nonself trials."""
    import matplotlib.pyplot as plt

    sub = individual_probability_frame(df).dropna(subset=["parsed_probability"])
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    if sub.empty:
        ax.set_title("Self-probability distribution (no data)")
        return ax
    is_self = sub["is_self"].astype(bool)
    edges = np.linspace(0, 100, bins + 1)
    ax.hist(sub.loc[is_self, "parsed_probability"], bins=edges, alpha=0.6,
            label=f"self (n={int(is_self.sum())})", density=True)
    ax.hist(sub.loc[~is_self, "parsed_probability"], bins=edges, alpha=0.6,
            label=f"nonself (n={int((~is_self).sum())})", density=True)
    ax.set_xlabel("parsed self-probability (0-100)")
    ax.set_ylabel("density")
    ax.set_title("Self-probability: self vs nonself")
    ax.legend()
    ax.figure.tight_layout()
    return ax


def plot_mean_probability_by_evaluator(df: pd.DataFrame, *, ax=None):
    """Grouped bars of mean parsed_probability per evaluator, self vs nonself."""
    import matplotlib.pyplot as plt

    tbl = mean_probability_by_evaluator(df)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.2 * len(tbl) + 2), 4))
    if tbl.empty:
        ax.set_title("Mean self-probability by evaluator (no data)")
        return ax
    tbl[["self_mean", "nonself_mean"]].plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("mean parsed self-probability")
    ax.set_xlabel("evaluator persona")
    ax.set_title("Mean self-probability by evaluator (self vs nonself)")
    ax.legend(["self", "nonself"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.figure.tight_layout()
    return ax


def plot_calibration(df: pd.DataFrame, *, bins: int = 10, ax=None):
    """Reliability curve: mean predicted self-probability vs empirical self rate."""
    import matplotlib.pyplot as plt

    tbl = calibration_table(df, bins=bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="0.6", label="perfect calibration")
    if not tbl.empty:
        ax.plot(tbl["mean_predicted"], tbl["empirical_self_rate"], "o-", label="observed")
        for _, r in tbl.iterrows():
            ax.annotate(f"n={r['n']}", (r["mean_predicted"], r["empirical_self_rate"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted self-probability")
    ax.set_ylabel("empirical self rate")
    ax.set_title("Self-probability calibration")
    ax.legend()
    ax.figure.tight_layout()
    return ax


# ── paired (A/B) counterbalanced bias ──────────────────────────────────────

def paired_frame(df: pd.DataFrame, *, ground_truth_only: bool = True) -> pd.DataFrame:
    """Successful paired rows. When ground_truth_only, keep only rows where the
    evaluator authored exactly one candidate (has_ground_truth True)."""
    sub = _phase_frame(df, "paired")
    if ground_truth_only and "has_ground_truth" in sub.columns:
        sub = sub[sub["has_ground_truth"] == True]  # noqa: E712
    return sub.copy()


def _chose_self_series(df: pd.DataFrame) -> pd.Series:
    """chose_self per row, falling back to is_correct for legacy paired rows."""
    cs = df["chose_self"] if "chose_self" in df.columns else pd.Series(np.nan, index=df.index)
    if "is_correct" in df.columns:
        cs = _coalesce(cs, df["is_correct"])
    return cs


def _position_of_self_series(df: pd.DataFrame) -> pd.Series:
    """position_of_self per row, derived from candidate sources for legacy rows."""
    pos = df["position_of_self"] if "position_of_self" in df.columns else pd.Series(np.nan, index=df.index)
    if {"candidate_a_source", "source_persona"} <= set(df.columns):
        derived = np.where(df["candidate_a_source"] == df["source_persona"], "A", "B")
        derived = pd.Series(derived, index=df.index).where(df["source_persona"].notna(), np.nan)
        pos = _coalesce(pos, derived)
    return pos


def paired_summary(df: pd.DataFrame) -> dict:
    """Headline paired numbers with the position-bias breakdown.

    accuracy is P(chose_self) over ground-truth rows; p_choose_a / p_choose_b are
    over *all* successful paired rows (counterbalanced, so ~0.5 if unbiased);
    accuracy_self_at_A / _B isolate whether bias is doing the work; parse_*
    reports how often the discrete answer was missing.
    """
    gt = paired_frame(df, ground_truth_only=True)
    allp = paired_frame(df, ground_truth_only=False)

    chose_self = _chose_self_series(gt).astype(float)
    pos = _position_of_self_series(gt)
    ans_all = _answer_col(allp)

    n_all = len(allp)
    p_a = float((ans_all == "A").mean()) if n_all else float("nan")
    p_b = float((ans_all == "B").mean()) if n_all else float("nan")
    acc_at = {
        letter: float(chose_self[pos == letter].mean()) if (pos == letter).any() else float("nan")
        for letter in ("A", "B")
    }
    n_failures = int(ans_all.isna().sum())
    return {
        "n_ground_truth": int(len(gt)),
        "accuracy_chose_self": float(chose_self.mean()) if len(gt) else float("nan"),
        "p_choose_a": p_a,
        "p_choose_b": p_b,
        "position_bias_a_minus_half": (p_a - 0.5) if n_all else float("nan"),
        "accuracy_self_at_A": acc_at["A"],
        "accuracy_self_at_B": acc_at["B"],
        "accuracy_gap_A_minus_B": (
            acc_at["A"] - acc_at["B"]
            if not (np.isnan(acc_at["A"]) or np.isnan(acc_at["B"])) else float("nan")
        ),
        "n_all": n_all,
        "n_parse_failures": n_failures,
        "parse_failure_rate": (n_failures / n_all) if n_all else float("nan"),
    }


def paired_accuracy_by_evaluator(df: pd.DataFrame, *, by_model: bool = True) -> pd.DataFrame:
    """P(chose_self) per evaluator (and model when by_model and a model column exists)."""
    gt = paired_frame(df, ground_truth_only=True)
    if gt.empty:
        return pd.DataFrame(columns=["accuracy_chose_self", "n"])
    gt = gt.copy()
    gt["_chose_self"] = _chose_self_series(gt).astype(float)
    keys = (["model"] if (by_model and "model" in gt.columns) else []) + ["evaluator_persona"]
    out = gt.groupby(keys)["_chose_self"].agg(["mean", "count"])
    return out.rename(columns={"mean": "accuracy_chose_self", "count": "n"})


def position_bias_by_evaluator(df: pd.DataFrame, *, by_model: bool = True) -> pd.DataFrame:
    """P(choose A) per evaluator (and model) over all paired rows. 0.5 = unbiased."""
    allp = paired_frame(df, ground_truth_only=False)
    if allp.empty:
        return pd.DataFrame(columns=["p_choose_a", "n"])
    allp = allp.copy()
    allp["_chose_a"] = (_answer_col(allp) == "A").astype(float)
    keys = (["model"] if (by_model and "model" in allp.columns) else []) + ["evaluator_persona"]
    out = allp.groupby(keys)["_chose_a"].agg(["mean", "count"])
    return out.rename(columns={"mean": "p_choose_a", "count": "n"})


def paired_confidence_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth paired rows that carry a parsed_confidence value."""
    gt = paired_frame(df, ground_truth_only=True)
    return gt.dropna(subset=["parsed_confidence"]).copy()


def confidence_correct_vs_incorrect(df: pd.DataFrame) -> pd.DataFrame:
    """Mean parsed_confidence split by whether the evaluator chose its own text."""
    sub = paired_confidence_frame(df)
    cols = ["mean_confidence", "n"]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    sub = sub.copy()
    sub["_chose_self"] = _chose_self_series(sub).astype(float)
    sub["outcome"] = np.where(sub["_chose_self"] == 1, "chose_self", "chose_other")
    out = sub.groupby("outcome")["parsed_confidence"].agg(["mean", "count"])
    return out.rename(columns={"mean": "mean_confidence", "count": "n"})


def mean_confidence_by_evaluator(df: pd.DataFrame) -> pd.DataFrame:
    """Mean parsed_confidence per evaluator persona."""
    sub = paired_confidence_frame(df)
    if sub.empty:
        return pd.DataFrame(columns=["mean_confidence", "n"])
    out = sub.groupby("evaluator_persona")["parsed_confidence"].agg(["mean", "count"])
    return out.rename(columns={"mean": "mean_confidence", "count": "n"})


def paired_confidence_parse_failure_rate(df: pd.DataFrame) -> dict:
    """How often confidence failed to parse among ground-truth paired rows."""
    gt = paired_frame(df, ground_truth_only=True)
    n = len(gt)
    failures = int(gt["parsed_confidence"].isna().sum())
    return {
        "n_trials": n,
        "n_parse_failures": failures,
        "parse_failure_rate": (failures / n) if n else float("nan"),
    }


def plot_paired_accuracy_by_evaluator(df: pd.DataFrame, *, ax=None):
    """Bars of P(chose_self) per evaluator; red dashed line = 0.5 chance."""
    import matplotlib.pyplot as plt

    tbl = paired_accuracy_by_evaluator(df, by_model=False)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.2 * len(tbl) + 2), 4))
    if tbl.empty:
        ax.set_title("Paired accuracy by evaluator (no data)")
        return ax
    tbl["accuracy_chose_self"].plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(chose self)")
    ax.set_xlabel("evaluator persona")
    ax.set_title("Paired recognition accuracy by evaluator")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.figure.tight_layout()
    return ax


def plot_position_bias_by_evaluator(df: pd.DataFrame, *, ax=None):
    """Bars of P(choose A) per evaluator; red dashed line = 0.5 (unbiased)."""
    import matplotlib.pyplot as plt

    tbl = position_bias_by_evaluator(df, by_model=False)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.2 * len(tbl) + 2), 4))
    if tbl.empty:
        ax.set_title("Position bias by evaluator (no data)")
        return ax
    tbl["p_choose_a"].plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5, color="tab:purple")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(choose A)")
    ax.set_xlabel("evaluator persona")
    ax.set_title("Positional bias by evaluator (0.5 = unbiased)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.figure.tight_layout()
    return ax


def plot_confidence_correct_vs_incorrect(df: pd.DataFrame, *, ax=None):
    """Box/strip of parsed_confidence split by chose_self vs chose_other."""
    import matplotlib.pyplot as plt

    sub = paired_confidence_frame(df)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    if sub.empty:
        ax.set_title("Confidence by outcome (no data)")
        return ax
    sub = sub.copy()
    sub["_chose_self"] = _chose_self_series(sub).astype(float)
    groups, labels = [], []
    for val, name in ((1.0, "chose self"), (0.0, "chose other")):
        vals = sub.loc[sub["_chose_self"] == val, "parsed_confidence"].dropna().values
        if len(vals):
            groups.append(vals)
            labels.append(f"{name}\n(n={len(vals)})")
    if groups:
        ax.boxplot(groups, labels=labels, showmeans=True)
    ax.set_ylabel("stated confidence (0-100)")
    ax.set_title("Paired confidence by outcome")
    ax.figure.tight_layout()
    return ax


# ── individual (YES/NO) response-bias diagnostics ──────────────────────────

def yes_rate_by_evaluator(df: pd.DataFrame, *, by_model: bool = True) -> pd.DataFrame:
    """YES rate per evaluator (and model) — the core YES/NO response-bias check.

    A row near 1.0 or 0.0 means the evaluator barely uses both tokens, so its
    individual-recognition accuracy is a base-rate artifact, not recognition.
    """
    sub = _phase_frame(df, "individual")
    if sub.empty:
        return pd.DataFrame(columns=["yes_rate", "n"])
    sub = sub.copy()
    sub["_yes"] = (_answer_col(sub) == "YES").astype(float)
    keys = (["model"] if (by_model and "model" in sub.columns) else []) + ["evaluator_persona"]
    out = sub.groupby(keys)["_yes"].agg(["mean", "count"])
    return out.rename(columns={"mean": "yes_rate", "count": "n"})


def yes_logprob_margin_by_self(df: pd.DataFrame) -> pd.DataFrame:
    """Mean (logprob_yes - logprob_no) split by self/nonself.

    Empty when the run did not record logprobs. A positive self-minus-nonself
    margin means the model assigns more YES log-probability to its own text —
    a calibration-sensitive self-recognition signal independent of argmax.
    """
    sub = _phase_frame(df, "individual")
    cols = ["mean_logprob_margin", "n"]
    if sub.empty or sub["logprob_yes"].isna().all() or sub["logprob_no"].isna().all():
        return pd.DataFrame(columns=cols)
    sub = sub.copy()
    sub["is_self"] = effective_is_self(sub)
    sub["_margin"] = sub["logprob_yes"] - sub["logprob_no"]
    sub = sub.dropna(subset=["_margin"])
    if sub.empty:
        return pd.DataFrame(columns=cols)
    out = sub.groupby(sub["is_self"].astype(bool))["_margin"].agg(["mean", "count"])
    out.index = out.index.map({True: "self", False: "nonself"})
    return out.rename(columns={"mean": "mean_logprob_margin", "count": "n"})


def plot_yes_rate_by_evaluator(df: pd.DataFrame, *, ax=None):
    """Bars of YES rate per evaluator; red dashed line = 0.5."""
    import matplotlib.pyplot as plt

    tbl = yes_rate_by_evaluator(df, by_model=False)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.2 * len(tbl) + 2), 4))
    if tbl.empty:
        ax.set_title("YES rate by evaluator (no data)")
        return ax
    tbl["yes_rate"].plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5, color="tab:green")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(YES)")
    ax.set_xlabel("evaluator persona")
    ax.set_title("YES rate by evaluator (response-bias diagnostic)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.figure.tight_layout()
    return ax


def plot_yes_logprob_margin(df: pd.DataFrame, *, ax=None):
    """Bars of mean (logprob_yes - logprob_no) for self vs nonself."""
    import matplotlib.pyplot as plt

    tbl = yes_logprob_margin_by_self(df)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    if tbl.empty:
        ax.set_title("YES/NO logprob margin (no logprobs recorded)")
        return ax
    tbl["mean_logprob_margin"].plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="0.4", linewidth=1)
    ax.set_ylabel("mean (logprob_yes - logprob_no)")
    ax.set_title("YES/NO logprob margin: self vs nonself")
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.figure.tight_layout()
    return ax


# ── One-call text summary (used by the notebook + smoke test) ──────────────

def bias_summary(df: pd.DataFrame) -> dict:
    """Aggregate the headline response-bias numbers across all available modes.

    Always returns a dict; sections for modes absent from `df` are simply
    omitted. Safe to call on any (including legacy) self-recognition result set.
    """
    df = ensure_bias_columns(df)
    out: dict = {}

    ind = _phase_frame(df, "individual")
    if not ind.empty:
        out["individual_yes_no"] = {
            "n": int(len(ind)),
            "overall_yes_rate": float((_answer_col(ind) == "YES").mean()),
            "logprob_margin_recorded": bool(not ind["logprob_yes"].isna().all()),
        }

    prob = individual_probability_frame(df)
    if not prob.empty:
        pf = probability_parse_failure_rate(df)
        out["individual_probability"] = {
            "n": int(len(prob)),
            "auroc_self_vs_nonself": auroc_self_vs_nonself(df),
            "parse_failure_rate": pf["parse_failure_rate"],
        }

    paired = paired_frame(df, ground_truth_only=False)
    if not paired.empty:
        out["paired"] = paired_summary(df)
        conf = paired_confidence_frame(df)
        if not conf.empty:
            out["paired"]["mean_confidence"] = float(conf["parsed_confidence"].mean())

    return out
