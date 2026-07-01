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
    return df


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
                      by=("evaluator_slug",)) -> pd.DataFrame:
    """Active − neutral surplus for one foil_type, per group. Shows how much
    inducing the persona adds over the no-persona baseline (the active-state gain,
    cross-model analogue of case7 − case10)."""
    sub = df[df["foil_type"] == foil_type]
    a = agg(sub[sub.condition == "active"], by=list(by)).rename(
        columns={"accuracy": "acc_active", "mean_prob_correct": "prob_active"})
    nrec = agg(sub[sub.condition == "neutral"], by=list(by)).rename(
        columns={"accuracy": "acc_neutral", "mean_prob_correct": "prob_neutral"})
    keep_a = list(by) + ["acc_active", "prob_active", "n"]
    keep_n = list(by) + ["acc_neutral", "prob_neutral"]
    out = a[keep_a].merge(nrec[keep_n], on=list(by), how="outer")
    out["acc_surplus"] = out["acc_active"] - out["acc_neutral"]
    out["prob_surplus"] = out["prob_active"] - out["prob_neutral"]
    return out
