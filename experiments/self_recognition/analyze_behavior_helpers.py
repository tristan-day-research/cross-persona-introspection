"""Behavioral analysis for the binary 12-case self-recognition eval.

This is the behavioral companion to `analyze_activations_helpers.py`: that module
reads the residual-stream activations and asks *how* recognition is represented;
this one reads the per-trial A/B decisions in the eval `*.jsonl` files and asks
*whether* — and under which conditions — a persona recognizes its own text, and
whether any recognition reflects privileged access rather than style-based content inference.

The 12 cases are NOT meant to be read in order; they form a designed argument
(see the case table). The analysis is organized around that argument:

  baselines    case8 (content-inference ceiling: neutral model handed both secrets),
               case9 (style-only floor: neutral, secrets redacted),
               case10 (minimal: neutral, no persona info).
  headline     case7 (active self, NO secret) vs case8 vs case9. The claim
               "case7 > case8" rules out content inference; "case7 - case9" is the
               active-state surplus over pure style.
  gradient     case7 (no desc) -> case11 (surface desc) -> case3 (full secret):
               how much the hidden secret adds for the ACTIVE persona.
  deactivation case6 (neutral, label wording) vs case7/case3: is recognition
               lost when the persona prompt is off?
  framing      case3 vs case4 (self-as-object attribution).
  control      case5 active vs neutral (does an active persona contaminate
               third-party classification?).
  single-text  case1, case2 — criterion-biased; reported with d' / criterion.
  calibration  case12 (child / archaic obvious-style personas): proves the
               pipeline detects recognition when style IS present, so the nulls
               elsewhere are interpretable.

Crucially, every accuracy is also cut by persona `coarse_category` (suppression /
near_twin / calibration / confound), because the access claim lives there:
style-based content inference cannot recover a SUPPRESSED latent (it is absent from the surface
text) or separate NEAR-TWINS (same surface style), so above-floor recognition on
those rows is the strongest evidence of privileged access.

Conventions mirror paper_replication_analysis.py (load_results + grouped means +
matplotlib) and reuse AnalysisConfig / category maps from
analyze_activations_helpers.py so both notebooks share one vocabulary. No scipy
needed — Wilson intervals and the normal-approx test use numpy + statistics.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from core.results_logger import load_results
from core.run_utils import model_slug
from experiments.self_recognition.analyze_activations_helpers import (
    TEXT_EVALUATIONS_DIR, coarse_category, load_persona_categories, load_task_categories,
)
from experiments.self_recognition.evaluation_cases import CASE_TYPES, resolve_spec

logger = logging.getLogger(__name__)
_N = NormalDist()

# Heavy free-text columns dropped by default — behavioral analysis needs none of
# them, and the result files are large because every row stores the texts shown.
_TEXT_COLS = ("single_text", "text1", "text2", "prompt_text", "system_prompt_text",
              "raw_response")

# One-line "what it tests" per case, for axis labels / tables.
CASE_LABELS = {
    "case1": "1 · single-text self-rec (no alt)",
    "case2": "2 · single-text self-vs-other (secret)",
    "case3": "3 · paired, O secret shown",
    "case4": "4 · paired attribution (self-as-object)",
    "case5": "5 · third-party O1-vs-O2 classify",
    "case6": "6 · deactivated self (neutral, label)",
    "case7": "7 · paired, NO secret (cleanest)",
    "case8": "8 · content-inference ceiling (neutral, full)",
    "case9": "9 · style floor (neutral, redacted)",
    "case10": "10 · minimal baseline (labels only)",
    "case11": "11 · active, redacted O (surface)",
    "case12": "12 · calibration (obvious-style)",
}


# ═══════════════════════════════════════════════════════════════════════════
# Load + clean the eval trial tables
# ═══════════════════════════════════════════════════════════════════════════

def eval_dir_for(run_name: str, task: str = "persona_category",
                 model: str | None = None) -> Path:
    """The collection directory evaluate_self_recognition.py wrote for this run.

    Evals are written under text_evaluations/<task>/<model_slug>/<run_name>/,
    mirroring the generation layout. Pass `model` (the eval model_name) to build
    that slugged path. Falls back to the legacy non-slugged
    text_evaluations/<task>/<run_name>/ when only the old layout exists on disk,
    so pre-existing runs still load; `model=None` also uses the legacy path.
    """
    legacy = TEXT_EVALUATIONS_DIR / task / run_name
    if model:
        slugged = TEXT_EVALUATIONS_DIR / task / model_slug(model) / run_name
        if slugged.is_dir() or not legacy.is_dir():
            return slugged
    return legacy


def load_eval_trials(run_name: str, *, task: str = "persona_category",
                     model: str | None = None,
                     drop_text: bool = True, personas_yaml: Path | None = None) -> pd.DataFrame:
    """Load every eval `*.jsonl` slice for `run_name` into one tidy frame.

    Concatenates all slice files and DEDUPLICATES on `trial_id` — essential
    because overlapping case-sets across runs (e.g. case8 appearing in both a
    `case7+case8` and a `case8+case9+...` slice) otherwise double-count. Adds:
      condition          "active" / "neutral" (from eval_system_prompt_enabled)
      evaluator_category fine persona category (from the roster YAML)
      evaluator_coarse   suppression / near_twin / calibration / confound / other
      task_category      task-family label (from the task set metadata)
      case_label         human-readable "what it tests" string

    `drop_text` removes the bulky text columns (keeps memory small); set False if
    you want to inspect prompts.
    """
    d = eval_dir_for(run_name, task, model)
    files = sorted(glob.glob(str(d / "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no eval *.jsonl under {d}")
    frames = [load_results(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    n_raw = len(df)
    if "trial_id" in df.columns:
        df = df.drop_duplicates(subset="trial_id", keep="last").reset_index(drop=True)
    n_dropped = n_raw - len(df)
    if n_dropped:
        logger.info(f"{run_name}: dropped {n_dropped} duplicate trial rows "
                    f"(overlapping case-sets across {len(files)} slice files)")

    if drop_text:
        df = df.drop(columns=[c for c in _TEXT_COLS if c in df.columns])

    cats = load_persona_categories(personas_yaml) if personas_yaml else load_persona_categories()
    try:
        task_cats = load_task_categories()
    except Exception:
        task_cats = {}
    df["condition"] = np.where(df["eval_system_prompt_enabled"], "active", "neutral")
    df["evaluator_category"] = df["evaluator_persona"].map(cats).fillna("")
    df["evaluator_coarse"] = df["evaluator_category"].map(coarse_category)
    df["task_category"] = df.get("task_id", pd.Series(index=df.index)).map(task_cats).fillna("")
    df["case_label"] = df["case_id"].map(CASE_LABELS).fillna(df["case_id"])
    df["framing"] = df["case_id"].map({c: case_framing(c) for c in df["case_id"].unique()})
    return df


def case_framing(case_id: str) -> str:
    """The question framing of a case — a large accuracy driver, so only compare
    WITHIN a framing:
      "single_text"    one text, yes/no                       (cases 1, 2)
      "pick_one"       two texts, pick which ONE is the target (cases 3, 6, 7, 10, 11)
      "attribute_both" two texts, assign BOTH ("which wrote which")  (cases 4, 5, 8, 9)
    """
    spec = resolve_spec(case_id)
    if spec.n_texts == 1:
        return "single_text"
    return "pick_one" if spec.answer_form == "single" else "attribute_both"


def dedup_report(run_name: str, *, task: str = "persona_category",
                 model: str | None = None) -> pd.DataFrame:
    """Per-slice-file row counts + per-case overlap across files, for a hygiene
    cell. Shows which cases are duplicated across slices (and thus deduped)."""
    d = eval_dir_for(run_name, task, model)
    rows = []
    for f in sorted(glob.glob(str(d / "*.jsonl"))):
        sub = load_results(f)
        for cid, n in sub["case_id"].value_counts().items():
            rows.append({"file": Path(f).name, "case_id": cid, "rows": int(n)})
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl
    return tbl.pivot_table(index="case_id", columns="file", values="rows",
                           aggfunc="sum", fill_value=0)


# ═══════════════════════════════════════════════════════════════════════════
# Statistics (Wilson interval + normal-approx test; no scipy)
# ═══════════════════════════════════════════════════════════════════════════

def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (robust at extremes,
    unlike the normal approximation). Returns (lo, hi); (nan, nan) if n == 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = _N.inv_cdf(1 - (1 - conf) / 2)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def binom_p(k: int, n: int, p0: float = 0.5) -> float:
    """Two-sided p-value that k/n differs from p0 (normal approximation with a
    continuity correction). Adequate at our n (hundreds–thousands); returns nan
    if n == 0."""
    if n == 0:
        return float("nan")
    se = np.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return float("nan")
    z = (abs(k / n - p0) - 0.5 / n) / se  # continuity-corrected
    z = max(z, 0.0)
    return float(2 * (1 - _N.cdf(z)))


def _z_rate(rate: float, n: int) -> float:
    """Inverse-normal of a rate, clamped to (0.5/n, 1-0.5/n) so d' stays finite."""
    lo, hi = 0.5 / n, 1 - 0.5 / n
    return _N.inv_cdf(min(max(rate, lo), hi))


def mean_ci(vals, conf: float = 0.95) -> tuple[float, float, float]:
    """Normal-approx CI for the mean of a continuous score — used for the GRADED
    logprob signal `mean_prob_correct` (the softmax mass the model put on the correct
    option, recorded per trial as `prob_correct`). The argmax `accuracy` throws this
    magnitude away; the mean of `prob_correct` keeps it, giving a lower-variance read
    of the same effect. Returns (mean, lo, hi); (mean, nan, nan) if n < 2."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    m = float(vals.mean())
    if n < 2:
        return (m, float("nan"), float("nan"))
    se = float(vals.std(ddof=1) / np.sqrt(n))
    z = _N.inv_cdf(1 - (1 - conf) / 2)
    return (m, m - z * se, m + z * se)


def mean_p_vs(vals, p0: float = 0.5) -> float:
    """Two-sided p that the mean of a continuous score differs from p0 (one-sample
    z on the mean, normal approx). For 'does the graded logprob mass beat chance=0.5?'
    — strictly more powerful than the argmax binomial test, since it uses how far
    each trial leaned, not just which side of 0.5 it fell on."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n < 2:
        return float("nan")
    sd = vals.std(ddof=1)
    if sd == 0:
        return 0.0 if vals.mean() != p0 else 1.0
    z = (vals.mean() - p0) / (sd / np.sqrt(n))
    return float(2 * (1 - _N.cdf(abs(z))))


# ═══════════════════════════════════════════════════════════════════════════
# Accuracy summaries
# ═══════════════════════════════════════════════════════════════════════════

def accuracy_table(df: pd.DataFrame, by=("case_id", "condition"),
                   conf: float = 0.95) -> pd.DataFrame:
    """Recognition accuracy per group, with a Wilson CI and a vs-chance p-value
    (argmax), AND the graded logprob signal: `mean_prob_correct` (mean softmax mass
    on the correct option) with its own normal CI (`prob_ci_lo`/`prob_ci_hi`) and a
    graded vs-chance p (`p_prob_vs_chance`). The graded columns use the per-trial
    magnitude the argmax discards, so they are lower-variance and more powerful for
    the near-threshold cells. Every binary case has chance = 0.5.

    `by` is any column or list of columns present on the frame (e.g.
    ("case_id", "condition"), or add "evaluator_coarse" for the category cut)."""
    by = [by] if isinstance(by, str) else list(by)
    df = df[df["is_correct"].notna()]
    rows = []
    for keys, g in df.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(g)
        k = int(g["is_correct"].sum())
        lo, hi = wilson_ci(k, n, conf)
        if "prob_correct" in g:
            pvals = g["prob_correct"].to_numpy()
            pm, plo, phi = mean_ci(pvals, conf)
            pp = mean_p_vs(pvals, 0.5)
        else:
            pm = plo = phi = pp = float("nan")
        rec = dict(zip(by, keys))
        rec.update({
            "n": n, "accuracy": k / n if n else float("nan"),
            "ci_lo": lo, "ci_hi": hi, "p_vs_chance": binom_p(k, n, 0.5),
            "mean_prob_correct": pm, "prob_ci_lo": plo, "prob_ci_hi": phi,
            "p_prob_vs_chance": pp,
        })
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(by).reset_index(drop=True) if not out.empty else out


def case_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Headline per-(case, condition) accuracy table, case-ordered with labels.
    The first thing to look at: which conditions sit above the 0.5 chance line."""
    out = accuracy_table(df, by=("case_id", "condition"))
    if out.empty:
        return out
    out["case_label"] = out["case_id"].map(CASE_LABELS).fillna(out["case_id"])
    out["case_type"] = out["case_id"].map(CASE_TYPES).fillna("")
    order = {c: i for i, c in enumerate(CASE_LABELS)}
    out["_o"] = out["case_id"].map(order)
    cols = ["case_id", "case_label", "case_type", "condition", "n",
            "accuracy", "ci_lo", "ci_hi", "p_vs_chance", "mean_prob_correct"]
    return out.sort_values(["_o", "condition"]).reset_index(drop=True)[cols]


# Persona categories ordered for the access claim: style CAN'T help on the left
# (suppression / near_twin), obvious style on the right (confound / calibration).
_CAT_ORDER = ["suppression", "near_twin", "confound", "calibration", "neutral"]


def case_category_matrix(df: pd.DataFrame, value: str = "accuracy") -> pd.DataFrame:
    """The master table: every (case, condition) as a row × every persona category as
    a column, plus an `all` overall column. `value` picks the cell metric —
    "accuracy" (argmax) or "mean_prob_correct" (graded logprob mass). chance = 0.5.

    This is the at-a-glance map the access argument reads off: scan the
    suppression / near_twin columns (where style can't help) for any case sitting
    above 0.5. Pair with `case_category_matrix(df, "mean_prob_correct")` to see the
    graded version of the same grid."""
    cells = accuracy_table(df, by=("case_id", "condition", "evaluator_coarse"))
    if cells.empty:
        return cells
    overall = accuracy_table(df, by=("case_id", "condition")).assign(evaluator_coarse="all")
    t = pd.concat([cells, overall], ignore_index=True)
    t["case"] = t["case_id"].map(CASE_LABELS).fillna(t["case_id"])
    order = {c: i for i, c in enumerate(CASE_LABELS)}
    t["_o"] = t["case_id"].map(order).fillna(len(order)).astype(int)
    t["row"] = t["case"] + "  [" + t["condition"] + "]"
    row_order = t.sort_values(["_o", "condition"])["row"].drop_duplicates().tolist()
    present = set(t["evaluator_coarse"])
    col_order = [c for c in _CAT_ORDER if c in present] \
        + [c for c in present if c not in _CAT_ORDER and c != "all"] + ["all"]
    piv = (t.pivot_table(index="row", columns="evaluator_coarse", values=value, observed=True)
             .reindex(index=row_order, columns=col_order))
    piv.columns.name = f"{value} (chance=0.5)"
    return piv.round(3)


# ═══════════════════════════════════════════════════════════════════════════
# Single-text signal-detection (cases 1, 2): separate sensitivity from bias
# ═══════════════════════════════════════════════════════════════════════════

def _chose_current(df: pd.DataFrame) -> pd.Series:
    """Per row: did the model pick the 'this is my current persona' option?
    Uses the recorded answer_mapping ({letter: semantic_key}) so it is robust to
    the A/B letter counterbalancing."""
    return df.apply(
        lambda r: (r.get("answer_mapping") or {}).get(r.get("predicted_answer")) == "current",
        axis=1,
    )


def dprime_table(df: pd.DataFrame, case_id: str, by="evaluator_persona") -> pd.DataFrame:
    """Signal-detection breakdown for a single-text case (1 or 2).

    Signal = "the shown text is the evaluator's own". Reports the self-hit rate
    H = P(say 'current' | self), the false-self rate F = P(say 'current' | other),
    sensitivity d' = z(H) - z(F), and criterion c = -0.5(z(H)+z(F)) (the yes/no
    lean the case table warns about). d' > 0 is genuine discrimination regardless
    of bias; large |c| means the accuracy is criterion-driven."""
    by = [by] if isinstance(by, str) else list(by)
    sub = df[df["case_id"] == case_id].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["_cur"] = _chose_current(sub)
    sub["_self"] = sub["source_persona"] == sub["evaluator_persona"]
    rows = []
    for keys, g in sub.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        self_g, other_g = g[g["_self"]], g[~g["_self"]]
        n_s, n_o = len(self_g), len(other_g)
        if n_s == 0 or n_o == 0:
            continue
        H, F = self_g["_cur"].mean(), other_g["_cur"].mean()
        zH, zF = _z_rate(H, n_s), _z_rate(F, n_o)
        rec = dict(zip(by, keys))
        rec.update({"n_self": n_s, "n_other": n_o, "hit_rate": float(H),
                    "false_self_rate": float(F), "dprime": float(zH - zF),
                    "criterion": float(-0.5 * (zH + zF)),
                    "accuracy": float(g["is_correct"].mean())})
        rows.append(rec)
    return pd.DataFrame(rows)


def dprime_overall(df: pd.DataFrame, case_id: str) -> dict:
    """Pooled d'/criterion for a single-text case (all evaluators together)."""
    sub = df[df["case_id"] == case_id].copy()
    if sub.empty:
        return {}
    sub["_one"] = "all"
    t = dprime_table(sub, case_id, by="_one")
    return t.iloc[0].to_dict() if not t.empty else {}


# ═══════════════════════════════════════════════════════════════════════════
# Designed contrasts — each returns a tidy accuracy frame over named conditions
# ═══════════════════════════════════════════════════════════════════════════

# A "condition" key → (case_id, condition_filter) selecting the rows for one bar.
# condition_filter is "active" / "neutral" / None (either).
def _select(df: pd.DataFrame, case_id: str, condition: str | None) -> pd.DataFrame:
    sub = df[df["case_id"] == case_id]
    if condition is not None:
        sub = sub[sub["condition"] == condition]
    return sub


def contrast(df: pd.DataFrame, conditions: dict[str, tuple[str, str | None]],
             by=None, conf: float = 0.95) -> pd.DataFrame:
    """Accuracy for each named condition (and optionally split by `by`, e.g.
    'evaluator_coarse'). `conditions` maps a display label → (case_id, condition).

    Returns one row per (label[, by-group]) with n / accuracy / Wilson CI /
    p-vs-chance / mean_prob_correct — the building block for every comparison
    figure. The dict order is preserved as a categorical so bars stay in the
    intended ladder order."""
    by = [] if by is None else ([by] if isinstance(by, str) else list(by))
    parts = []
    for label, (case_id, cond) in conditions.items():
        sub = _select(df, case_id, cond)
        if sub.empty:
            continue
        if by:
            t = accuracy_table(sub, by=by, conf=conf)
        else:
            t = accuracy_table(sub.assign(_all="all"), by="_all", conf=conf).drop(columns="_all")
        t.insert(0, "label", label)
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["label"] = pd.Categorical(out["label"], categories=list(conditions), ordered=True)
    sort_cols = ["label"] + by
    return out.sort_values(sort_cols).reset_index(drop=True)


def access_ladder(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """The headline comparison: active-self with no secret vs the no-access
    baselines. case7 > case8 rules out content inference; case7 - case9 is the active
    surplus over style. case10 anchors the floor."""
    return contrast(df, {
        "active self\n(7: no secret)": ("case7", "active"),
        "content-inference ceiling\n(8: neutral, full)": ("case8", "neutral"),
        "style floor\n(9: neutral, redacted)": ("case9", "neutral"),
        "minimal\n(10: labels only)": ("case10", "neutral"),
    }, by=by)


def secret_gradient(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """How much the hidden secret adds for the ACTIVE persona, as the comparison
    class goes from absent -> named-but-redacted -> full secret."""
    return contrast(df, {
        "no desc\n(7)": ("case7", "active"),
        "surface desc\n(11: redacted)": ("case11", "active"),
        "full secret\n(3)": ("case3", "active"),
    }, by=by)


def deactivation(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """Active-state contribution: paired self-recognition with the persona prompt
    ON (case7) vs the deactivated, label-worded version (case6). Interpret the gap
    against calibration (case12), whose cue is in the text and should survive
    deactivation."""
    return contrast(df, {
        "active (7)": ("case7", "active"),
        "deactivated (6)": ("case6", "neutral"),
        "calibration active (12)": ("case12", "active"),
        "calibration neutral (12)": ("case12", "neutral"),
    }, by=by)


def framing(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """Does forcing third-person attribution suppress the self-anchor?
    case3 ('which is mine?') vs case4 (assign both texts)."""
    return contrast(df, {
        "first-person (3)": ("case3", "active"),
        "attribution (4)": ("case4", "active"),
    }, by=by)


def classification_contamination(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """Does an active persona contaminate third-party judgment? case5 (classify
    two OTHER personas, E authored neither) with E active vs neutral."""
    return contrast(df, {
        "E active (5)": ("case5", "active"),
        "E neutral (5)": ("case5", "neutral"),
    }, by=by)


# ═══════════════════════════════════════════════════════════════════════════
# Framing-matched comparisons (control for the question-framing confound)
# ═══════════════════════════════════════════════════════════════════════════
# Question framing is a LARGE accuracy driver: "pick_one" cases (which ONE text is
# the target — 3,6,7,10,11) run ~0.57-0.60, while "attribute_both" cases (which
# wrote which — 4,5,8,9) run ~0.50-0.53. So case8's low score is mostly the harder
# framing, NOT content inference failing — comparing case7 (pick_one) to case8
# (attribute_both) is not apples-to-apples. The clean active-state test holds
# framing constant: case7 (active, no desc) vs case10 (neutral, no desc).

def framing_matched_ladder(df: pd.DataFrame, by=None) -> pd.DataFrame:
    """The clean headline, all within the `pick_one` framing: active self with no
    description (7) vs a neutral model handed E's full secret (6 — pure content inference)
    vs a neutral model with no info at all (10 — the matched floor). If 7 ≈ 6,
    being the active persona is no better than being handed the secret (content inference
    is NOT ruled out); 7 ≫ 10 is the active-state contribution over no-info."""
    return contrast(df, {
        "active self\n(7: no desc)": ("case7", "active"),
        "neutral + secret\n(6: content inference)": ("case6", "neutral"),
        "neutral, no info\n(10: floor)": ("case10", "neutral"),
    }, by=by)


def active_state_effect(df: pd.DataFrame, *, active_case: str = "case7",
                        neutral_case: str = "case10", by="evaluator_coarse") -> pd.DataFrame:
    """The clean active-state effect: accuracy(active_case, active) −
    accuracy(neutral_case, neutral) per `by` group, with a two-proportion z-test.

    The two cases must share the SAME framing and description so the only thing
    that differs is whether the persona is active. Default case7 vs case10 are both
    `pick_one` with NO description — so a positive effect can't be style (the floor
    sits at chance) or content inference (nothing is described). The category cut is the
    point: a genuine introspective signal should appear in suppression / near_twin
    (where style can't help), not only in calibration / confound."""
    by = [by] if isinstance(by, str) else list(by)
    a = accuracy_table(_select(df, active_case, "active"), by=by)[by + ["accuracy", "n"]]
    a = a.rename(columns={"accuracy": "acc_active", "n": "n_active"})
    nf = accuracy_table(_select(df, neutral_case, "neutral"), by=by)[by + ["accuracy", "n"]]
    nf = nf.rename(columns={"accuracy": "acc_neutral", "n": "n_neutral"})
    m = a.merge(nf, on=by)
    m["effect"] = m["acc_active"] - m["acc_neutral"]
    m["se"] = np.sqrt(m["acc_active"] * (1 - m["acc_active"]) / m["n_active"]
                      + m["acc_neutral"] * (1 - m["acc_neutral"]) / m["n_neutral"])
    m["z"] = m["effect"] / m["se"].replace(0, np.nan)
    m["p_value"] = m["z"].map(lambda z: float(2 * (1 - _N.cdf(abs(z)))) if pd.notna(z) else float("nan"))
    return m


def graded_active_state_effect(df: pd.DataFrame, *, active_case: str = "case7",
                               neutral_case: str = "case10", by="evaluator_coarse") -> pd.DataFrame:
    """The GRADED twin of `active_state_effect`: difference in mean `prob_correct`
    (the logprob mass on the correct option) between the active case and the neutral
    floor, per `by` group, with a Welch two-sample z on the means. Same contrast as
    the argmax version, but using the per-trial magnitude the argmax discards — so a
    near-threshold cell like near_twin gets a more powerful test. A positive graded
    effect concentrated in suppression / near_twin would be the introspection signal;
    one concentrated in calibration / confound is access to expressed style."""
    by = [by] if isinstance(by, str) else list(by)

    def _stats(sub):
        rows = []
        for keys, g in sub.groupby(by, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            v = g["prob_correct"].to_numpy()
            v = v[~np.isnan(v)]
            rec = dict(zip(by, keys))
            rec.update({"mean": float(v.mean()) if len(v) else float("nan"),
                        "var": float(v.var(ddof=1)) if len(v) > 1 else float("nan"),
                        "n": len(v)})
            rows.append(rec)
        return pd.DataFrame(rows)

    a = _stats(_select(df, active_case, "active")).rename(
        columns={"mean": "prob_active", "var": "var_a", "n": "n_active"})
    nf = _stats(_select(df, neutral_case, "neutral")).rename(
        columns={"mean": "prob_neutral", "var": "var_n", "n": "n_neutral"})
    m = a.merge(nf, on=by)
    m["effect"] = m["prob_active"] - m["prob_neutral"]
    m["se"] = np.sqrt(m["var_a"] / m["n_active"] + m["var_n"] / m["n_neutral"])
    m["z"] = m["effect"] / m["se"].replace(0, np.nan)
    m["p_value"] = m["z"].map(lambda z: float(2 * (1 - _N.cdf(abs(z)))) if pd.notna(z) else float("nan"))
    return m[by + ["prob_active", "n_active", "prob_neutral", "n_neutral",
                   "effect", "se", "z", "p_value"]]


# ═══════════════════════════════════════════════════════════════════════════
# Plots (matplotlib; Agg-safe — each takes/returns an Axes like the paper module)
# ═══════════════════════════════════════════════════════════════════════════

def plot_accuracy(table: pd.DataFrame, *, x="label", title="", ax=None,
                  chance=0.5, annotate_n=True, value="accuracy", lo_col="ci_lo",
                  hi_col="ci_hi", ylabel="recognition accuracy", color="#4C72B0"):
    """Bar chart with error bars and a chance line. `table` is any frame from
    accuracy_table/contrast (one bar per row of column `x`). Defaults plot argmax
    `accuracy`; pass value="mean_prob_correct", lo_col="prob_ci_lo",
    hi_col="prob_ci_hi", ylabel="mean P(correct)" for the GRADED logprob view."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 1.1 * len(table) + 1.5), 4))
    if table.empty:
        ax.set_title((title + " (no data)").strip())
        return ax
    labels = table[x].astype(str).tolist()
    acc = table[value].to_numpy()
    # Error-bar offsets, clipped at 0: a CI bound that crosses the point estimate
    # (small-n / degenerate cells, or a NaN in one column — common in partial runs)
    # would otherwise be negative, which matplotlib rejects. Render those as a
    # zero-length whisker rather than crashing.
    lo = np.nan_to_num(acc - table[lo_col].to_numpy(), nan=0.0)
    hi = np.nan_to_num(table[hi_col].to_numpy() - acc, nan=0.0)
    lo = np.clip(lo, 0, None)
    hi = np.clip(hi, 0, None)
    bars = ax.bar(labels, acc, yerr=[lo, hi], capsize=3, color=color,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(chance, ls="--", c="gray", lw=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    if annotate_n and "n" in table:
        for b, n in zip(bars, table["n"]):
            ax.annotate(f"n={int(n)}", (b.get_x() + b.get_width() / 2, 0.02),
                        ha="center", va="bottom", fontsize=7, color="white")
    ax.figure.tight_layout()
    return ax


def plot_contrast_by_category(table: pd.DataFrame, *, group="evaluator_coarse",
                              title="", ax=None, chance=0.5, value="accuracy",
                              lo_col="ci_lo", hi_col="ci_hi",
                              ylabel="recognition accuracy"):
    """Grouped bars (with error bars) : each named condition (label) split by
    persona category. This is the figure that carries the access argument — look
    for active-self bars above the style-floor bars specifically within
    suppression / near_twin. Pass value="mean_prob_correct",
    lo_col="prob_ci_lo", hi_col="prob_ci_hi", ylabel="mean P(correct)" for the
    GRADED view (the CI columns must match the value)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))
    if table.empty:
        ax.set_title((title + " (no data)").strip())
        return ax
    pivot = table.pivot_table(index="label", columns=group, values=value,
                              observed=True)
    pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.4)

    # Overlay per-bar CI whiskers. Pivot the CI bounds on the SAME index/columns
    # so each bar's error offsets line up; read back the drawn patch geometry to
    # place the whisker at each bar's true center. Offsets are clipped at 0 and
    # NaN-safe (degenerate/small-n cells → zero-length whisker, never negative).
    have_ci = lo_col in table.columns and hi_col in table.columns
    if have_ci:
        lo_piv = table.pivot_table(index="label", columns=group, values=lo_col,
                                   observed=True).reindex(index=pivot.index,
                                                          columns=pivot.columns)
        hi_piv = table.pivot_table(index="label", columns=group, values=hi_col,
                                   observed=True).reindex(index=pivot.index,
                                                          columns=pivot.columns)
        n_series = pivot.shape[1]                 # bars per category group
        for col_i in range(n_series):
            patches = ax.patches[col_i * pivot.shape[0]:(col_i + 1) * pivot.shape[0]]
            cat = pivot.columns[col_i]
            xs, ys, los, his = [], [], [], []
            for row_i, patch in enumerate(patches):
                v = pivot.iloc[row_i, col_i]
                if v != v:                        # NaN bar (missing cell) — skip
                    continue
                xs.append(patch.get_x() + patch.get_width() / 2)
                ys.append(v)
                los.append(v - lo_piv.iloc[row_i, col_i])
                his.append(hi_piv.iloc[row_i, col_i] - v)
            if xs:
                los = np.clip(np.nan_to_num(los, nan=0.0), 0, None)
                his = np.clip(np.nan_to_num(his, nan=0.0), 0, None)
                ax.errorbar(xs, ys, yerr=[los, his], fmt="none",
                            ecolor="black", elinewidth=0.7, capsize=2)

    ax.axhline(chance, ls="--", c="gray", lw=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(title=group, fontsize=8, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)
    ax.figure.tight_layout()
    return ax
