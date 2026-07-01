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


# ═══════════════════════════════════════════════════════════════════════════
# Privileged-access verdict (numeric): case7(active) vs case6(neutral)
# ═══════════════════════════════════════════════════════════════════════════
# Implements the AI-1 §7e request: per (evaluator × persona category), a ladder,
# an introspective-surplus test (two-proportion z + Holm across categories), a
# precondition check (is content inference at floor?), a graded (Welch) companion,
# a clustered bootstrap CI (AI-2 #4), and a printed verdict. case8 is reported as a
# ceiling only and never used in the verdict.

_CAT_ORDER = ["suppression", "near_twin", "confound", "calibration", "neutral"]


def _p_two_sided(z: float) -> float:
    from experiments.self_recognition.analyze_behavior_helpers import _N
    return 2.0 * (1.0 - _N.cdf(abs(z))) if z == z else float("nan")


def _two_prop(k1, n1, k2, n2):
    """Difference of proportions p1−p2: (delta, z [pooled test], p, wald_lo, wald_hi)."""
    import numpy as np
    if n1 == 0 or n2 == 0:
        return (float("nan"),) * 5
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    pp = (k1 + k2) / (n1 + n2)
    se_pool = (pp * (1 - pp) * (1 / n1 + 1 / n2)) ** 0.5
    z = d / se_pool if se_pool > 0 else float("nan")
    se_un = (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) ** 0.5
    from experiments.self_recognition.analyze_behavior_helpers import _N
    zc = _N.inv_cdf(0.975)
    return d, z, _p_two_sided(z), d - zc * se_un, d + zc * se_un


def _one_prop_vs(k, n, p0=0.5):
    """One-sample proportion vs p0: (phat, z, p, wilson_lo, wilson_hi, includes_p0)."""
    from experiments.self_recognition.analyze_behavior_helpers import _N, wilson_ci
    if n == 0:
        return (float("nan"),) * 5 + (False,)
    phat = k / n
    se = (p0 * (1 - p0) / n) ** 0.5
    z = (phat - p0) / se if se > 0 else float("nan")
    lo, hi = wilson_ci(k, n)
    return phat, z, _p_two_sided(z), lo, hi, (lo <= p0 <= hi)


def _welch(a, b):
    """Welch two-sample on arrays a,b (graded): (delta, t, p) via normal approx."""
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    d = a.mean() - b.mean()
    se = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 0.5
    t = d / se if se > 0 else float("nan")
    return d, t, _p_two_sided(t)


def _holm(pvals):
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    import numpy as np
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    order = np.argsort(np.where(ok, p, np.inf))
    m = ok.sum()
    adj = np.full_like(p, np.nan)
    run = 0.0
    for rank, idx in enumerate(order[:m]):
        val = min(1.0, (m - rank) * p[idx])
        run = max(run, val)
        adj[idx] = run
    return adj


def _clustered_boot_diff(sub, cluster_cols, n_boot=1000, seed=0):
    """Clustered bootstrap 95% CI for acc(case7) − acc(case6). Resamples CLUSTERS
    (default persona×foil×task) with replacement to respect within-cluster
    correlation (trials share tasks/personas/texts). Vectorized over per-cluster
    (sum,n) counts. Returns (lo, hi, method_str)."""
    import numpy as np
    g = sub.assign(_is7=sub.base_case.eq("case7"), _is6=sub.base_case.eq("case6"))
    per = g.groupby(cluster_cols, observed=True).apply(
        lambda x: pd.Series({
            "s7": x.loc[x._is7, "is_correct"].sum(), "n7": int(x._is7.sum()),
            "s6": x.loc[x._is6, "is_correct"].sum(), "n6": int(x._is6.sum())}),
        include_groups=False) if hasattr(pd.DataFrame, "groupby") else None
    arr = per[["s7", "n7", "s6", "n6"]].to_numpy(float)
    C = len(arr)
    if C < 2:
        return float("nan"), float("nan"), "clustered_bootstrap(insufficient clusters)"
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        sel = arr[rng.integers(0, C, C)].sum(0)
        s7, n7, s6, n6 = sel
        if n7 > 0 and n6 > 0:
            deltas.append(s7 / n7 - s6 / n6)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi), f"clustered_bootstrap(clusters={C}, key={'×'.join(cluster_cols)})"


def access_verdict(df: pd.DataFrame, foil_type: str = "same_model_diff_persona",
                   chance: float = 0.5, n_boot: int = 1000,
                   cluster_cols=("evaluator_persona", "foil_persona", "task_id")):
    """Numeric privileged-access verdict for one foil_type. Returns a dict of
    DataFrames (ladder / surplus / precondition / verdict), one row per
    (evaluator_slug, evaluator_coarse), suppression & near_twin first.

    - ladder: case7/6/8 accuracy + graded mean + n.
    - surplus: case7−case6 with two-proportion z, two-sided p, Holm-adjusted p
      (across categories, per evaluator), Wald 95% CI, clustered-bootstrap 95% CI,
      and a graded Welch companion.
    - precondition: case6 vs chance (one-sample z, Wilson CI, `clean` = CI covers
      chance ⇒ content inference is at floor here).
    - verdict: PRIVILEGED ACCESS / no access (7≈6) / content-inference / ambiguous.
    case8 (attribute-both) is a ceiling; never used for the verdict.

    Requires case6 and case7 to be pick-one (2 options); raises if not matched.
    """
    if "base_case" not in df.columns:
        return {}
    ft = df[df.foil_type == foil_type]
    c7, c6, c8 = _native(ft, "case7"), _native(ft, "case6"), _native(ft, "case8")
    if c7.empty or c6.empty:
        return {}
    # Framing guard: both must be pick-one (exactly 2 answer options A/B). Our binary
    # eval is always A/B, but assert the answer_mapping has 2 keys for both.
    def _n_opts(g):
        am = g["answer_mapping"].iloc[0]
        if isinstance(am, str):
            import json as _j; am = _j.loads(am)
        return len(am)
    if _n_opts(c7) != 2 or _n_opts(c6) != 2:
        raise AssertionError("case6/case7 are not both 2-option pick-one — framing not matched")

    def cat_rows(g):
        return g.groupby(["evaluator_slug", "evaluator_coarse"], observed=True)

    lad, sur, pre, ver = [], [], [], []
    # Precompute per (evaluator, category) counts.
    keys = sorted(set(c7[["evaluator_slug", "evaluator_coarse"]].apply(tuple, axis=1))
                  | set(c6[["evaluator_slug", "evaluator_coarse"]].apply(tuple, axis=1)),
                  key=lambda t: (t[0], _CAT_ORDER.index(t[1]) if t[1] in _CAT_ORDER else 99))
    for ev, cat in keys:
        g7 = c7[(c7.evaluator_slug == ev) & (c7.evaluator_coarse == cat)]
        g6 = c6[(c6.evaluator_slug == ev) & (c6.evaluator_coarse == cat)]
        g8 = c8[(c8.evaluator_slug == ev) & (c8.evaluator_coarse == cat)]
        n7, k7 = len(g7), int(g7.is_correct.sum())
        n6, k6 = len(g6), int(g6.is_correct.sum())
        if n7 == 0 or n6 == 0:
            continue
        acc7, acc6 = k7 / n7, k6 / n6
        lad.append({"evaluator_slug": ev, "evaluator_coarse": cat,
                    "acc7": acc7, "acc6": acc6,
                    "acc8": (g8.is_correct.mean() if len(g8) else float("nan")),
                    "graded7": g7.prob_correct.mean(), "graded6": g6.prob_correct.mean(),
                    "graded8": (g8.prob_correct.mean() if len(g8) else float("nan")),
                    "n7": n7, "n6": n6, "n8": len(g8)})
        d, z, p, lo, hi = _two_prop(k7, n7, k6, n6)
        blo, bhi, method = _clustered_boot_diff(
            pd.concat([g7, g6]), list(cluster_cols), n_boot=n_boot)
        gd, gt, gp = _welch(g7.prob_correct.to_numpy(), g6.prob_correct.to_numpy())
        sur.append({"evaluator_slug": ev, "evaluator_coarse": cat, "surplus": d,
                    "z": z, "p": p, "ci_lo": lo, "ci_hi": hi,
                    "boot_lo": blo, "boot_hi": bhi, "boot_method": method,
                    "graded_surplus": gd, "graded_t": gt, "graded_p": gp})
        ph, zc, pc, wlo, whi, clean = _one_prop_vs(k6, n6, chance)
        pre.append({"evaluator_slug": ev, "evaluator_coarse": cat, "acc6": ph,
                    "ci_lo": wlo, "ci_hi": whi, "z_vs_chance": zc, "p_vs_chance": pc,
                    "clean": clean})
    lad = pd.DataFrame(lad); sur = pd.DataFrame(sur); pre = pd.DataFrame(pre)
    if sur.empty:
        return {"ladder": lad, "surplus": sur, "precondition": pre, "verdict": pd.DataFrame()}
    # Holm across categories WITHIN each evaluator.
    sur["p_holm"] = sur.groupby("evaluator_slug")["p"].transform(lambda s: _holm(s.to_numpy()))
    # Verdict.
    pre_idx = pre.set_index(["evaluator_slug", "evaluator_coarse"])["clean"].to_dict()
    prep_idx = pre.set_index(["evaluator_slug", "evaluator_coarse"])["p_vs_chance"].to_dict()
    rows = []
    for _, r in sur.iterrows():
        key = (r.evaluator_slug, r.evaluator_coarse)
        c6_clean = pre_idx.get(key, False)
        c6_above = (prep_idx.get(key, 1.0) < 0.05) and not c6_clean
        sig_pos = (r.ci_lo > 0) and (r.p_holm < 0.05)
        ci_incl_0 = (r.ci_lo <= 0 <= r.ci_hi)
        if sig_pos and c6_clean:
            v = "PRIVILEGED ACCESS"
        elif ci_incl_0:
            v = "no access (7≈6)"
        elif c6_above and not sig_pos:
            v = "content-inference"
        else:
            v = "ambiguous"
        rows.append({**{k: r[k] for k in ["evaluator_slug", "evaluator_coarse", "surplus",
                        "ci_lo", "ci_hi", "boot_lo", "boot_hi", "p", "p_holm"]},
                     "case6_clean": c6_clean, "verdict": v})
    ver = pd.DataFrame(rows)
    return {"ladder": lad, "surplus": sur, "precondition": pre, "verdict": ver}


def model_vs_persona(df: pd.DataFrame, condition: str = "active") -> pd.DataFrame:
    """AI-2 #5 diagnostic: per evaluator, model-recognition (diff_model_same_persona),
    persona-recognition (same_model_diff_persona), and persona-anchor
    (persona_vs_model) accuracy, with an interpretation flag."""
    def acc(ft):
        s = df[(df.foil_type == ft) & (df.condition == condition) & (df.base_case == "case7")]
        return (s.is_correct.mean(), len(s)) if len(s) else (float("nan"), 0)
    rows = []
    for ev in sorted(df.evaluator_slug.unique()):
        d = df[df.evaluator_slug == ev]
        mr, nmr = (lambda s: (s.is_correct.mean(), len(s)))(
            d[(d.foil_type == "diff_model_same_persona") & (d.condition == "active") & (d.base_case == "case7")])
        pr, npr = (lambda s: (s.is_correct.mean(), len(s)))(
            d[(d.foil_type == "same_model_diff_persona") & (d.condition == "active") & (d.base_case == "case7")])
        pa_s = d[(d.foil_type == "persona_vs_model") & (d.condition == "active")]
        pa, npa = (pa_s.is_correct.mean(), len(pa_s)) if len(pa_s) else (float("nan"), 0)
        interp = "persona-dominant" if (pr == pr and mr == mr and pr - 0.5 > 2 * (mr - 0.5)) else "mixed/model-informative"
        rows.append({"evaluator_slug": ev, "model_recognition": mr, "persona_recognition": pr,
                     "persona_anchor": pa, "n_model": nmr, "n_persona": npr, "n_anchor": npa,
                     "interpretation": interp})
    return pd.DataFrame(rows)


def validation_report(df: pd.DataFrame) -> dict:
    """AI-2 #6: counts by base_case×foil_type×evaluator, A/B + text-order balance,
    unique tasks/personas/pairs, and any missing (base_case, foil_type) cells."""
    import pandas as pd
    counts = df.groupby(["base_case", "foil_type", "evaluator_slug"], observed=True).size().rename("n")
    bal = (df.assign(cA=df.correct_answer.eq("A"),
                     t1=df.text_order.astype(str).str.contains("self_first|pm_first"))
             .groupby(["base_case", "foil_type"], observed=True)
             .agg(frac_correct_A=("cA", "mean"), frac_targetlike_first=("t1", "mean"), n=("cA", "size")))
    uniq = pd.Series({
        "unique_tasks": df.task_id.nunique(),
        "unique_evaluator_personas": df.evaluator_persona.nunique(),
        "unique_persona_pairs": df[["evaluator_persona", "foil_persona"]].drop_duplicates().shape[0],
        "n_rows": len(df)})
    have = set(map(tuple, df[["base_case", "foil_type"]].drop_duplicates().to_numpy()))
    all_cases = sorted(df.base_case.unique()); all_foils = sorted(df.foil_type.unique())
    missing = [(bc, ft) for bc in all_cases for ft in all_foils if (bc, ft) not in have]
    return {"counts": counts.to_frame(), "balance": bal.round(3), "uniques": uniq.to_frame("value"),
            "missing_cells": missing}
