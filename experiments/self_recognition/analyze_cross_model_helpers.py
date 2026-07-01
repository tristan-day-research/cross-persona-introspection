"""Loading + aggregation for the CROSS-MODEL self-recognition eval
(evaluate_cross_model.py). Kept separate from analyze_behavior_helpers.py (the
12-case single-model analysis), but REUSES its stats + plotters so the figures
match: `agg()` returns the same columns as `accuracy_table` (accuracy, Wilson
ci_lo/ci_hi, mean_prob_correct + normal CI), so you can feed the frames straight
to B.plot_accuracy / B.plot_contrast_by_category.

The rows are PersonaCrossModelEvalRecord (see core/schemas.py). Chance = 0.5.
The key analysis axes:
  evaluator_model   which model is judging ("is this mine?")
  condition         active (persona induced) vs neutral (baseline)
  foil_type         diff_model_same_persona | same_model_diff_persona | diff_model_diff_persona
  evaluator_persona the persona whose text is "self"
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.run_utils import model_slug
from core.results_logger import load_results
import experiments.self_recognition.analyze_behavior_helpers as B

logger = logging.getLogger(__name__)

TEXT_EVALUATIONS_DIR = Path(__file__).parent / "results" / "text_evaluations"

# Persona → coarse category, reused from the 12-case helpers so both notebooks
# bucket personas identically.
_coarse = B.coarse_category
_load_cats = B.load_persona_categories

FOIL_LABELS = {
    "diff_model_same_persona": "other model,\nsame persona",
    "same_model_diff_persona": "same model,\nother persona",
    "diff_model_diff_persona": "other model,\nother persona",
    "persona_vs_model": "neither self:\npersona vs model",
}


def _model_dir(task: str, model: str, eval_dir: str) -> Path:
    return TEXT_EVALUATIONS_DIR / task / model_slug(model) / eval_dir


def load_cross_model(eval_dir: str, models, *, task: str = "persona_category",
                     personas_yaml: Path | None = None) -> pd.DataFrame:
    """Concatenate the `cross_model.jsonl` from each model's slugged eval dir into
    one tidy frame, deduped on trial_id. `models` is the list of evaluator model
    names (same strings as the config's eval_models). Adds:
      evaluator_coarse  persona category of the evaluator persona (self)
      foil_coarse       persona category of the foil persona
      foil_label        pretty foil_type label for plots
    """
    frames = []
    for m in models:
        d = _model_dir(task, m, eval_dir)
        files = sorted(glob.glob(str(d / "cross_model.jsonl")))
        if not files:
            logger.warning(f"no cross_model.jsonl under {d} (evaluator {m}) — skipping")
            continue
        for f in files:
            frames.append(load_results(f))
    if not frames:
        raise FileNotFoundError(
            f"no cross_model.jsonl found for {list(models)} under "
            f"{TEXT_EVALUATIONS_DIR / task}/<model_slug>/{eval_dir}")
    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)
    if "trial_id" in df.columns:
        df = df.drop_duplicates(subset="trial_id", keep="last").reset_index(drop=True)
    if n_raw - len(df):
        logger.info(f"{eval_dir}: dropped {n_raw - len(df)} duplicate trial rows")

    cats = _load_cats(personas_yaml) if personas_yaml else _load_cats()
    df["evaluator_coarse"] = df["evaluator_persona"].map(cats).map(_coarse)
    df["foil_coarse"] = df["foil_persona"].map(cats).map(_coarse)
    df["foil_label"] = df["foil_type"].map(FOIL_LABELS).fillna(df["foil_type"])
    # A short evaluator-model label (the slug) for axis labels / grouping.
    df["evaluator_slug"] = df["evaluator_model"].map(model_slug)
    # Rows written before base_case existed (the original case7 / persona_vs_model
    # run) have no base_case — they are all case7 wording, so treat them as case7.
    if "base_case" not in df.columns:
        df["base_case"] = "case7"
    else:
        df["base_case"] = df["base_case"].fillna("case7")
    return df


# Native condition per 12-case question (active = persona induced; neutral = off).
# Used to pick the right rows when a case (like case7) is present in both conditions.
NATIVE_CONDITION = {"case7": "active", "case3": "active", "case11": "active",
                    "case6": "neutral", "case8": "neutral", "case9": "neutral",
                    "case10": "neutral"}


def _native(df: pd.DataFrame, base_case: str) -> pd.DataFrame:
    """Rows for `base_case` in its native condition (so case7 → active only, even
    when a neutral floor is also present in the same file)."""
    cond = NATIVE_CONDITION.get(base_case)
    sub = df[df["base_case"] == base_case]
    return sub[sub["condition"] == cond] if cond else sub


def agg(df: pd.DataFrame, by) -> pd.DataFrame:
    """Accuracy + Wilson CI + graded mean_prob_correct per group — identical
    columns to B.accuracy_table, so the frame feeds B.plot_accuracy directly.
    `by` is a column name or list of them."""
    return B.accuracy_table(df, by=by)


def model_recognition(df: pd.DataFrame, *, by=("evaluator_slug", "condition")) -> pd.DataFrame:
    """The MODEL-identity probe: restrict to diff_model_same_persona (foil is the
    other model, same persona) and aggregate. This is 'can the model tell its own
    output from the other model's, holding persona fixed?'."""
    sub = df[df["foil_type"] == "diff_model_same_persona"]
    out = agg(sub, by=list(by))
    return out


CASE_LABELS = {
    "case7": "7 introspection\n(active, no desc)",
    "case6": "6 content inference\n(neutral, target described)",
    "case8": "8 content ceiling\n(neutral, both described)",
    "case3": "3 active + foil described",
    "case11": "11 active + foil redacted",
}


def case_ladder(df: pd.DataFrame, *, foil_type: str = "same_model_diff_persona",
                by=("evaluator_slug",), cases=("case7", "case6", "case8")) -> pd.DataFrame:
    """Introspection-vs-content-inference ladder for one foil_type: accuracy per
    base_case (each keeps its native condition), so you can compare
    case7 (active, no description → introspection) against case6 / case8 (neutral,
    described → content inference). `by` may add "evaluator_coarse" for the category
    cut. Adds a `label` column (from CASE_LABELS) for plotting. Requires a `base_case`
    column (present in cross_model_cases runs)."""
    if "base_case" not in df.columns:
        return df.iloc[0:0]
    # Each case in its NATIVE condition (case7 → active, case6/8 → neutral).
    parts = [_native(df[df["foil_type"] == foil_type], c) for c in cases]
    sub = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(
        not p.empty for p in parts) else df.iloc[0:0]
    if sub.empty:
        return sub
    out = agg(sub, by=["base_case", *by])
    out["label"] = out["base_case"].map(CASE_LABELS).fillna(out["base_case"])
    # order the ladder 7 → 6 → 8 (introspection → content inference → ceiling)
    order = {c: i for i, c in enumerate(["case7", "case6", "case8", "case3", "case11"])}
    return out.sort_values(["base_case", *by], key=lambda s: s.map(order) if s.name == "base_case" else s)


def introspection_surplus(df: pd.DataFrame, *, foil_type: str = "same_model_diff_persona",
                          by=("evaluator_slug",), conf: float = 0.95) -> pd.DataFrame:
    """case7 (active, introspection) − case6 (neutral, content inference) per group,
    with a two-proportion 95% CI on the difference. A CI clearing 0 = the active
    state adds recognition beyond what the description alone (content inference)
    supports. Same column shape as active_vs_neutral (acc_surplus + CI)."""
    from experiments.self_recognition.analyze_behavior_helpers import _N
    if "base_case" not in df.columns:
        return df.iloc[0:0]
    z = _N.inv_cdf(1 - (1 - conf) / 2)
    by = list(by)
    ft = df[(df["foil_type"] == foil_type) & df["is_correct"].notna()]
    # case7 in its native (active) condition; case6 is neutral by construction.
    sub = pd.concat([_native(ft, "case7"), _native(ft, "case6")], ignore_index=True)
    if sub.empty:
        return df.iloc[0:0]
    rows = []
    for keys, g in sub.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        a = g[g.base_case == "case7"]; c = g[g.base_case == "case6"]
        if len(a) == 0 or len(c) == 0:
            continue
        pa, pc = a.is_correct.mean(), c.is_correct.mean()
        se = ((pa * (1 - pa) / len(a)) + (pc * (1 - pc) / len(c))) ** 0.5
        d = pa - pc
        rec = dict(zip(by, keys))
        rec.update({"n_introspect": len(a), "n_content": len(c),
                    "acc_introspection": pa, "acc_content": pc, "acc_surplus": d,
                    "acc_surplus_lo": d - z * se, "acc_surplus_hi": d + z * se})
        rows.append(rec)
    return pd.DataFrame(rows)


def persona_anchor(df: pd.DataFrame, *, condition: str = "active",
                   by=("evaluator_slug",)) -> pd.DataFrame:
    """The persona_vs_model probe: NEITHER text is the evaluator's own — one is the
    same persona in the OTHER model, one is a different persona in the SAME model.
    "correct" was scored as the same-persona text, so `accuracy` = the PERSONA-ANCHOR
    rate: >0.5 the active persona claims the same-persona (other-model) text (anchors
    on PERSONA), <0.5 it claims the same-model (other-persona) text (anchors on
    MODEL), 0.5 = no lean. Chance line = 0.5. Returns the standard agg columns."""
    sub = df[(df["foil_type"] == "persona_vs_model") & (df["condition"] == condition)]
    if sub.empty:
        return sub
    out = agg(sub, by=list(by))
    return out


def foil_type_ladder(df: pd.DataFrame, *, condition: str = "active",
                     by_model: bool = True) -> pd.DataFrame:
    """Accuracy per foil_type (optionally split by evaluator model) for one
    condition — the headline 'what makes a foil easy to reject' view. Adds a
    `label` column (foil label, or 'slug | foil') for plotting."""
    sub = df[df["condition"] == condition]
    keys = (["evaluator_slug", "foil_type"] if by_model else ["foil_type"])
    out = agg(sub, by=keys)
    if out.empty:
        return out
    if by_model:
        out["label"] = out["evaluator_slug"] + "\n" + out["foil_type"].map(FOIL_LABELS).fillna(out["foil_type"])
    else:
        out["label"] = out["foil_type"].map(FOIL_LABELS).fillna(out["foil_type"])
    return out


def active_vs_neutral(df: pd.DataFrame, *, foil_type: str = "diff_model_same_persona",
                      by=("evaluator_slug",), conf: float = 0.95,
                      base_case: str = "case7") -> pd.DataFrame:
    """Active − neutral surplus for one foil_type WITHIN a single base_case (default
    case7 — the self-recognition question), per group, WITH a 95% CI. Shows how much
    inducing the persona adds over its no-persona floor (case7 active vs case7
    neutral). Scoping to base_case matters once a file also holds case6/case8, whose
    neutral rows would otherwise contaminate the neutral bucket.

    The surplus is a difference of two independent estimates, so its CI is the
    two-sample one: SE = sqrt(SE_active² + SE_neutral²), half-width = z·SE.
    Columns: <by>, n_active, n_neutral, acc_active, acc_neutral, acc_surplus,
    acc_surplus_lo/hi, prob_active, prob_neutral, prob_surplus, prob_surplus_lo/hi.
    """
    from experiments.self_recognition.analyze_behavior_helpers import _N
    z = _N.inv_cdf(1 - (1 - conf) / 2)
    by = list(by)
    sub = df[df["foil_type"] == foil_type]
    if "base_case" in sub.columns:
        sub = sub[sub["base_case"] == base_case]
    sub = sub[sub["is_correct"].notna()]
    rows = []
    for keys, g in sub.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        a = g[g.condition == "active"]
        n = g[g.condition == "neutral"]
        na, nn = len(a), len(n)
        if na == 0 or nn == 0:
            continue
        pa, pn = a.is_correct.mean(), n.is_correct.mean()
        se_acc = ((pa * (1 - pa) / na) + (pn * (1 - pn) / nn)) ** 0.5
        d_acc = pa - pn
        ma, mn = a.prob_correct.mean(), n.prob_correct.mean()
        # ddof=1 sample std; guard n=1.
        va = a.prob_correct.var(ddof=1) if na > 1 else 0.0
        vn = n.prob_correct.var(ddof=1) if nn > 1 else 0.0
        se_pr = ((va / na) + (vn / nn)) ** 0.5
        d_pr = ma - mn
        rec = dict(zip(by, keys))
        rec.update({
            "n_active": na, "n_neutral": nn,
            "acc_active": pa, "acc_neutral": pn, "acc_surplus": d_acc,
            "acc_surplus_lo": d_acc - z * se_acc, "acc_surplus_hi": d_acc + z * se_acc,
            "prob_active": ma, "prob_neutral": mn, "prob_surplus": d_pr,
            "prob_surplus_lo": d_pr - z * se_pr, "prob_surplus_hi": d_pr + z * se_pr,
        })
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(by).reset_index(drop=True) if not out.empty else out


def plot_surplus(table: pd.DataFrame, *, x="label", value="acc_surplus",
                 lo_col="acc_surplus_lo", hi_col="acc_surplus_hi",
                 title="", ylabel="active − neutral surplus", ax=None, color="#C44E52"):
    """Diverging bar chart for a SURPLUS (difference) with 95% CI error bars and a
    zero reference line. Unlike B.plot_accuracy this is NOT clamped to [0, 1], so
    negative surpluses render correctly. `table` is any frame from active_vs_neutral
    (one bar per row of `x`)."""
    import numpy as np
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 1.1 * len(table) + 1.5), 4))
    if table.empty:
        ax.set_title((title + " (no data)").strip()); return ax
    labels = table[x].astype(str).tolist()
    val = table[value].to_numpy()
    lo = np.clip(np.nan_to_num(val - table[lo_col].to_numpy(), nan=0.0), 0, None)
    hi = np.clip(np.nan_to_num(table[hi_col].to_numpy() - val, nan=0.0), 0, None)
    ax.bar(labels, val, yerr=[lo, hi], capsize=3, color=color,
           edgecolor="black", linewidth=0.5)
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.figure.tight_layout()
    return ax
