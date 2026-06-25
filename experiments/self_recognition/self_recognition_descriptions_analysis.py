"""Analysis for the multi-class persona-source classification eval
(`all_persona_descriptions`, produced by evaluate_self_recognition.py).

Reads the eval's trials.jsonl (PersonaDescriptionEvalRecord rows) and answers the
experiment's headline question: does an *active* evaluator persona classify its
own generations better than other evaluator personas classify that same source
persona's generations, over and above a neutral description-only baseline?

Metrics (all restricted to the `all_persona_descriptions` phase):
  - top-1 accuracy and mean probability assigned to the correct source;
  - per-source and per-evaluator accuracy / correct-probability tables;
  - confusion matrices (true source × predicted source) per condition;
  - self-advantage, the headline contrast:
        self_advantage_prob = mean P(correct | evaluator_persona == source_persona)
                            - mean P(correct | evaluator_persona != source_persona)
        self_advantage_acc  = acc(evaluator_persona == source_persona)
                            - acc(evaluator_persona != source_persona)
    over the active_persona rows, plus a per-source breakdown;
  - active-persona self cells vs the neutral description-only baseline, which
    isolates any *active-self* gain from ordinary prompt-based classification.

The metric functions are pure (operate on a passed DataFrame, testable headless),
mirroring self_recognition_bias_analysis; `summarize_run` loads a run's
trials.jsonl, writes CSVs + summary.md, and returns a metrics dict, mirroring
paper_replication_analysis.summarize_run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DESCRIPTIONS_PHASE = "all_persona_descriptions"
ACTIVE = "active_persona"
NEUTRAL = "description_only"
EXCL_SELF = "active_persona_excl_self"


# ── Preparation / derived columns ───────────────────────────────────────────

def _predicted_source(row) -> str | None:
    """The predicted persona NAME for a row, via the hidden per-trial mapping."""
    mapping, label = row.get("candidate_mapping"), row.get("predicted_label")
    if isinstance(mapping, dict) and label in mapping:
        return mapping[label]
    return None


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to the descriptions phase and add derived columns.

    Adds `predicted_source` (the predicted persona name, resolved through the
    per-trial {label: persona} mapping), `n_candidates`, and `self_label` (the
    effective self/nonself ground truth: the stored `is_self`, falling back to
    evaluator_persona == source_persona for active rows; NaN for the neutral
    condition where there is no active persona). Failed-parse rows are kept so
    parse rates remain computable; the metric helpers drop them as needed.
    """
    if "phase" in df.columns:
        df = df[df["phase"] == DESCRIPTIONS_PHASE]
    df = df.copy()
    if df.empty:
        return df
    df["predicted_source"] = df.apply(_predicted_source, axis=1)
    df["n_candidates"] = df["candidate_mapping"].apply(
        lambda m: len(m) if isinstance(m, dict) else np.nan)
    df["self_label"] = _self_label(df)
    return df


def _self_label(df: pd.DataFrame) -> pd.Series:
    """Effective self/nonself ground truth (object Series of True/False/NaN).

    Prefers the stored `is_self`; where absent, derives evaluator == source for
    the active condition only (the neutral baseline has no self and stays NaN).
    """
    stored = df["is_self"] if "is_self" in df.columns else pd.Series(np.nan, index=df.index)
    out = stored.astype(object).copy()
    need = out.isna()
    if need.any() and {"condition", "evaluator_persona", "source_persona"} <= set(df.columns):
        active = need & (df["condition"] == ACTIVE) & df["evaluator_persona"].notna()
        out.loc[active] = (df.loc[active, "evaluator_persona"]
                           == df.loc[active, "source_persona"])
    return out


def _scored(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a usable parse: is_correct present (probabilities recovered)."""
    return df[df["is_correct"].notna()]


def _chance(df: pd.DataFrame) -> float | None:
    """Chance accuracy = 1 / (number of candidates). NaN-safe over the run."""
    n = df["n_candidates"].dropna()
    return float(1.0 / n.mean()) if not n.empty else None


# ── Overall, per-condition ──────────────────────────────────────────────────

def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition overview: trials, parse-failure rate, top-1 accuracy, mean
    correct-source probability, and chance (1 / n_candidates)."""
    d = prepare(df)
    if d.empty:
        return pd.DataFrame()
    rows = []
    for cond, g in d.groupby("condition"):
        s = _scored(g)
        rows.append({
            "condition": cond,
            "n_trials": int(len(g)),
            "n_scored": int(len(s)),
            "parse_failure_rate": float((g["parse_status"] == "failed").mean())
            if "parse_status" in g else float(g["is_correct"].isna().mean()),
            "top1_accuracy": float(s["is_correct"].mean()) if len(s) else np.nan,
            "mean_correct_probability": float(s["correct_probability"].mean())
            if len(s) else np.nan,
            "chance": _chance(g),
        })
    return pd.DataFrame(rows).set_index("condition").sort_index()


# ── Group tables (accuracy + correct-probability + n) ───────────────────────

def accuracy_by(df: pd.DataFrame, group_cols, condition: str | None = ACTIVE) -> pd.DataFrame:
    """Accuracy, mean correct-probability, and n grouped by `group_cols`.

    `group_cols` is a column name or list of names (e.g. "evaluator_persona",
    "source_persona", "key", "model", or a combination). `condition` filters to
    one condition (default the active-persona rows); pass None to pool both.
    """
    d = prepare(df)
    if condition is not None and not d.empty:
        d = d[d["condition"] == condition]
    s = _scored(d)
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if s.empty:
        return pd.DataFrame(columns=group_cols + ["accuracy", "mean_correct_probability", "n"])
    g = s.groupby(group_cols)
    out = pd.DataFrame({
        "accuracy": g["is_correct"].mean().astype(float),
        "mean_correct_probability": g["correct_probability"].mean().astype(float),
        "n": g.size().astype(int),
    })
    return out.sort_index()


def _group_col(df: pd.DataFrame) -> str:
    """The column naming the generation group (task set). Prefers the clearer
    `task_set`; falls back to `dataset` for older result files."""
    return "task_set" if "task_set" in df.columns else "dataset"


def accuracy_by_group(df: pd.DataFrame, condition: str | None = ACTIVE) -> pd.DataFrame:
    """Accuracy / correct-prob / n per generation group (task set)."""
    return accuracy_by(df, _group_col(df), condition=condition)


def self_advantage_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Self-advantage computed separately within each generation group (task set),
    so neutral vs misaligned (etc.) task sets can be compared post-hoc.

    Returns one row per group with self/nonself accuracy + correct-prob means and
    the two deltas (same definitions as `self_advantage`)."""
    d = prepare(df)
    if d.empty:
        return pd.DataFrame()
    gcol = _group_col(d)
    d = _scored(d[d["condition"] == ACTIVE])
    if d.empty:
        return pd.DataFrame()
    rows = []
    for grp, g in d.groupby(gcol):
        sf = g[g["self_label"] == True]     # noqa: E712
        ot = g[g["self_label"] == False]    # noqa: E712

        def _m(s, col):
            return float(s[col].mean()) if len(s) else np.nan

        rows.append({
            gcol: grp,
            "self_accuracy": _m(sf, "is_correct"),
            "nonself_accuracy": _m(ot, "is_correct"),
            "self_advantage_acc": _m(sf, "is_correct") - _m(ot, "is_correct"),
            "self_correct_probability": _m(sf, "correct_probability"),
            "nonself_correct_probability": _m(ot, "correct_probability"),
            "self_advantage_prob": _m(sf, "correct_probability") - _m(ot, "correct_probability"),
            "n_self": int(len(sf)),
            "n_nonself": int(len(ot)),
        })
    return pd.DataFrame(rows).set_index(gcol).sort_index()


def confusion_matrix(df: pd.DataFrame, condition: str | None = ACTIVE,
                     normalize: bool = True) -> pd.DataFrame:
    """True source (rows) × predicted source (cols) confusion matrix.

    `normalize=True` row-normalizes (each true source's predictions sum to 1, so
    the diagonal is per-source recall). `condition` selects active/neutral
    (default active); pass None to pool both.
    """
    d = prepare(df)
    if condition is not None and not d.empty:
        d = d[d["condition"] == condition]
    s = _scored(d)
    s = s[s["predicted_source"].notna()]
    if s.empty:
        return pd.DataFrame()
    cm = pd.crosstab(s["source_persona"], s["predicted_source"],
                     normalize="index" if normalize else False)
    # Square it up so the diagonal is well-defined even if a class was never predicted.
    labels = sorted(set(cm.index) | set(cm.columns))
    cm = cm.reindex(index=labels, columns=labels, fill_value=0.0)
    cm.index.name, cm.columns.name = "true_source", "predicted_source"
    return cm


# ── Self-advantage (the headline contrast) ──────────────────────────────────

def self_advantage(df: pd.DataFrame) -> dict:
    """Self-advantage over the active-persona rows (see module docstring).

    Returns self/nonself means + counts and the two deltas:
      self_advantage_prob = mean correct-prob(self) - mean correct-prob(nonself)
      self_advantage_acc  = accuracy(self)          - accuracy(nonself)
    where self == (evaluator_persona == source_persona). Means are NaN (and the
    delta NaN) when a side has no scored rows.
    """
    d = prepare(df)
    d = _scored(d[d["condition"] == ACTIVE]) if not d.empty else d
    if d.empty:
        return {"n_self": 0, "n_nonself": 0}
    is_self = d["self_label"] == True   # noqa: E712 — object Series, want exact True
    self_rows, nonself_rows = d[is_self], d[d["self_label"] == False]  # noqa: E712

    def _mean(s):
        return float(s.mean()) if len(s) else np.nan

    self_acc, nonself_acc = _mean(self_rows["is_correct"]), _mean(nonself_rows["is_correct"])
    self_p = _mean(self_rows["correct_probability"])
    nonself_p = _mean(nonself_rows["correct_probability"])
    return {
        "n_self": int(len(self_rows)),
        "n_nonself": int(len(nonself_rows)),
        "self_accuracy": self_acc,
        "nonself_accuracy": nonself_acc,
        "self_advantage_acc": self_acc - nonself_acc,
        "self_correct_probability": self_p,
        "nonself_correct_probability": nonself_p,
        "self_advantage_prob": self_p - nonself_p,
    }


def self_advantage_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Per-source self-advantage: for each source persona S, the active evaluator
    that IS S (self) vs the other active evaluators (≠S) classifying S's texts.

    Columns: self_accuracy / other_accuracy / self_advantage_acc and the
    correct-probability analogues, with n_self / n_other counts.
    """
    d = prepare(df)
    if d.empty:
        return pd.DataFrame()
    d = _scored(d[d["condition"] == ACTIVE])
    if d.empty:
        return pd.DataFrame()
    rows = []
    for src, g in d.groupby("source_persona"):
        sf = g[g["self_label"] == True]      # noqa: E712
        ot = g[g["self_label"] == False]     # noqa: E712

        def _m(s, col):
            return float(s[col].mean()) if len(s) else np.nan

        rows.append({
            "source_persona": src,
            "self_accuracy": _m(sf, "is_correct"),
            "other_accuracy": _m(ot, "is_correct"),
            "self_advantage_acc": _m(sf, "is_correct") - _m(ot, "is_correct"),
            "self_correct_probability": _m(sf, "correct_probability"),
            "other_correct_probability": _m(ot, "correct_probability"),
            "self_advantage_prob": _m(sf, "correct_probability") - _m(ot, "correct_probability"),
            "n_self": int(len(sf)),
            "n_other": int(len(ot)),
        })
    return pd.DataFrame(rows).set_index("source_persona").sort_index()


def guess_counts(df: pd.DataFrame, condition: str | None = None) -> pd.Series:
    """How often each persona was the model's top-1 guess (`predicted_source`).

    Counts scored trials only (parse succeeded). `condition` filters to
    active_persona / description_only; pass None to pool both.
    """
    d = prepare(df)
    if condition is not None and not d.empty:
        d = d[d["condition"] == condition]
    s = _scored(d)
    s = s[s["predicted_source"].notna()]
    if s.empty:
        return pd.Series(dtype=int, name="guess_count")
    out = s["predicted_source"].value_counts().sort_index()
    out.name = "guess_count"
    return out


def guess_counts_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Top-1 guess counts per persona, split by condition (rows sum to n_scored)."""
    d = prepare(df)
    s = _scored(d)
    s = s[s["predicted_source"].notna()]
    if s.empty:
        return pd.DataFrame()
    return pd.crosstab(s["condition"], s["predicted_source"]).sort_index(axis=1)


def mean_probability_by_persona(df: pd.DataFrame,
                                condition: str | None = None) -> pd.DataFrame:
    """Mean probability mass assigned to each persona across trials.

    For every scored trial, reads the normalized `probabilities` at the label
    that maps to each candidate persona. A persona can receive high mean mass
    even when it is rarely the argmax — useful for detecting systematic bias.
    Includes `chance` (= 1 / n_candidates) for reference.
    """
    d = prepare(df)
    if condition is not None and not d.empty:
        d = d[d["condition"] == condition]
    d = d[d["probabilities"].notna()]
    if d.empty:
        return pd.DataFrame(columns=["mean_probability", "std_probability", "n_trials", "chance"])

    by_persona: dict[str, list[float]] = {}
    for _, row in d.iterrows():
        mapping, probs = row.get("candidate_mapping"), row.get("probabilities")
        if not isinstance(mapping, dict) or not isinstance(probs, dict):
            continue
        for label, persona in mapping.items():
            if label in probs:
                by_persona.setdefault(persona, []).append(float(probs[label]))

    chance = _chance(d)
    rows = [{
        "persona": persona,
        "mean_probability": float(np.mean(vals)),
        "std_probability": float(np.std(vals)) if len(vals) > 1 else 0.0,
        "n_trials": len(vals),
        "chance": chance,
    } for persona, vals in sorted(by_persona.items())]
    return pd.DataFrame(rows).set_index("persona")


def trial_counts_evaluator_source(df: pd.DataFrame,
                                  condition: str = ACTIVE) -> pd.DataFrame:
    """How many scored trials exist for each (evaluator, true source) pair.

    The eval loops every active evaluator over every source persona's texts, so
    counts are nearly uniform (± parse failures / missing generations).
    """
    d = prepare(df)
    if condition is not None and not d.empty:
        d = d[d["condition"] == condition]
    d = _scored(d)
    if d.empty:
        return pd.DataFrame()
    ct = pd.crosstab(d["evaluator_persona"], d["source_persona"])
    ct.index.name, ct.columns.name = "evaluator_persona", "source_persona"
    return ct.sort_index().sort_index(axis=1)


def prediction_given_source(df: pd.DataFrame, evaluator: str,
                            condition: str = ACTIVE) -> pd.DataFrame:
    """P(predicted source | true source) for one evaluator (rows sum to 1).

    Diagonal = recall on each source's texts; off-diagonal shows mislabeling
    patterns. Differs from `evaluator_guess_breakdown`, which marginalizes over
    true source (hence high self-guess rate can coexist with perfect diagonal).
    """
    d = prepare(df)
    if condition is not None and not d.empty:
        d = d[d["condition"] == condition]
    d = _scored(d)
    d = d[(d["evaluator_persona"] == evaluator) & d["predicted_source"].notna()]
    if d.empty:
        return pd.DataFrame()
    cm = pd.crosstab(d["source_persona"], d["predicted_source"], normalize="index")
    labels = sorted(set(cm.index) | set(cm.columns))
    cm = cm.reindex(index=labels, columns=labels, fill_value=0.0)
    cm.index.name, cm.columns.name = "true_source", "predicted_source"
    return cm


def evaluator_classification_rates(df: pd.DataFrame,
                                   condition: str = ACTIVE,
                                   task_set: str | None = None) -> pd.DataFrame:
    """Per active evaluator: self-hit, false-self, overall, and non-self accuracy.

    All rates are top-1 (predicted_source vs source_persona) on scored trials:

      self_hit_rate     P(predicted = evaluator | source = evaluator)
      false_self_rate   P(predicted = evaluator | source ≠ evaluator)
      overall_accuracy  P(predicted = source) over all trials for this evaluator
      nonself_accuracy  P(predicted = source | source ≠ evaluator)
      self_discrimination  self_hit_rate − false_self_rate

    Also includes trial counts, frac_self_trials, rate_guess_self, and chance.

    `task_set`, if given, restricts to one generation group (task-set name stored in
    `task_set` or `dataset`).
    """
    d = prepare(df)
    if task_set is not None and not d.empty:
        gcol = _group_col(d)
        d = d[d[gcol] == task_set]
    if condition is not None and not d.empty:
        d = d[d["condition"] == condition]
    d = _scored(d)
    d = d[d["evaluator_persona"].notna() & d["predicted_source"].notna()]
    if d.empty:
        return pd.DataFrame()

    chance = _chance(d)
    rows = []
    for ev, g in d.groupby("evaluator_persona"):
        is_self = g["source_persona"] == ev
        n = len(g)
        n_self = int(is_self.sum())
        n_non = int((~is_self).sum())
        self_rows = g[is_self]
        non_rows = g[~is_self]

        def _acc(sub):
            return float(sub["is_correct"].mean()) if len(sub) else np.nan

        def _pred_self(sub):
            return float((sub["predicted_source"] == ev).mean()) if len(sub) else np.nan

        rows.append({
            "evaluator_persona": ev,
            "n_total": n,
            "n_self_trials": n_self,
            "n_nonself_trials": n_non,
            "frac_self_trials": n_self / n if n else np.nan,
            "self_hit_rate": _pred_self(self_rows),
            "false_self_rate": _pred_self(non_rows),
            "overall_accuracy": _acc(g),
            "nonself_accuracy": _acc(non_rows),
            "chance": chance,
        })
    out = pd.DataFrame(rows).set_index("evaluator_persona").sort_index()
    # Unconditional self-guess rate (for back-compat / decomposition plots).
    out["rate_guess_self"] = (
        out["frac_self_trials"] * out["self_hit_rate"]
        + (1 - out["frac_self_trials"]) * out["false_self_rate"]
    )
    out["self_discrimination"] = out["self_hit_rate"] - out["false_self_rate"]
    return out


def guess_self_decomposition(df: pd.DataFrame, evaluator: str,
                           condition: str = ACTIVE) -> dict:
    """Per-evaluator subset of `evaluator_classification_rates` (one row as dict).

    Kept for notebook cells that inspect a single persona; prefer the table
    function for the canonical metric names.
    """
    rates = evaluator_classification_rates(df, condition=condition)
    if evaluator not in rates.index:
        return {}
    row = rates.loc[evaluator]
    return {
        "evaluator": evaluator,
        "n_total": int(row["n_total"]),
        "n_self_trials": int(row["n_self_trials"]),
        "n_nonself_trials": int(row["n_nonself_trials"]),
        "frac_self_trials": float(row["frac_self_trials"]),
        "self_hit_rate": float(row["self_hit_rate"]),
        "false_self_rate": float(row["false_self_rate"]),
        "overall_accuracy": float(row["overall_accuracy"]),
        "nonself_accuracy": float(row["nonself_accuracy"]),
        "self_discrimination": float(row["self_discrimination"]),
        "rate_guess_self": float(row["rate_guess_self"]),
        # Legacy aliases used by earlier notebook cells.
        "rate_guess_self_on_self_trials": float(row["self_hit_rate"]),
        "rate_guess_self_on_nonself_trials": float(row["false_self_rate"]),
        "accuracy_self_trials": float(row["self_hit_rate"]),
        "accuracy_nonself_trials": float(row["nonself_accuracy"]),
    }


def evaluator_guess_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per active evaluator: top-1 guess counts/rates for each candidate persona.

    Columns include `n_trials`, `guess_self`, `rate_guess_self`, and for every
    persona P that appears as a predicted source: `guess_P` / `rate_P`.
    """
    d = prepare(df)
    d = _scored(d[d["condition"] == ACTIVE]) if not d.empty else d
    d = d[d["evaluator_persona"].notna() & d["predicted_source"].notna()]
    if d.empty:
        return pd.DataFrame()

    personas = sorted(set(d["evaluator_persona"]) | set(d["predicted_source"]))
    rows = []
    for ev, g in d.groupby("evaluator_persona"):
        n = int(len(g))
        counts = g["predicted_source"].value_counts()
        row: dict = {"evaluator_persona": ev, "n_trials": n,
                     "guess_self": int(counts.get(ev, 0)),
                     "rate_guess_self": float(counts.get(ev, 0)) / n}
        for p in personas:
            c = int(counts.get(p, 0))
            row[f"guess_{p}"] = c
            row[f"rate_{p}"] = c / n
        rows.append(row)
    return pd.DataFrame(rows).set_index("evaluator_persona").sort_index()


def active_vs_neutral(df: pd.DataFrame) -> pd.DataFrame:
    """Per-source comparison of the active-self cell against the neutral baseline.

    For each source persona S:
      active_self_*  — the active evaluator that IS S classifying S's texts;
      active_other_* — the other active evaluators (≠S) on S's texts;
      neutral_*      — the no-persona description_only baseline on S's texts.
    The `active_self_minus_neutral_*` deltas isolate any active-self gain over
    ordinary prompt-based classification (which the neutral baseline measures).
    """
    d = prepare(df)
    if d.empty:
        return pd.DataFrame()
    s = _scored(d)
    act = s[s["condition"] == ACTIVE]
    neu = s[s["condition"] == NEUTRAL]
    sources = sorted(set(s["source_persona"]))
    rows = []
    for src in sources:
        a = act[act["source_persona"] == src]
        a_self = a[a["self_label"] == True]    # noqa: E712
        a_other = a[a["self_label"] == False]  # noqa: E712
        n = neu[neu["source_persona"] == src]

        def _m(x, col):
            return float(x[col].mean()) if len(x) else np.nan

        rows.append({
            "source_persona": src,
            "active_self_accuracy": _m(a_self, "is_correct"),
            "active_other_accuracy": _m(a_other, "is_correct"),
            "neutral_accuracy": _m(n, "is_correct"),
            "active_self_minus_neutral_acc": _m(a_self, "is_correct") - _m(n, "is_correct"),
            "active_self_correct_probability": _m(a_self, "correct_probability"),
            "neutral_correct_probability": _m(n, "correct_probability"),
            "active_self_minus_neutral_prob": (_m(a_self, "correct_probability")
                                               - _m(n, "correct_probability")),
            "n_active_self": int(len(a_self)),
            "n_neutral": int(len(n)),
        })
    return pd.DataFrame(rows).set_index("source_persona")


# ── Run-level summary (CSVs + markdown), mirroring paper_replication ─────────

def summarize_run(jsonl_path: str | Path, run_dir: str | Path) -> dict:
    """Load a descriptions eval run, write analysis CSVs + summary.md, and return
    a metrics dict. Mirrors paper_replication_analysis.summarize_run."""
    from core.results_logger import load_results

    jsonl_path, run_dir = Path(jsonl_path), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(jsonl_path)
    d = prepare(df)
    if d.empty:
        return {"error": "no all_persona_descriptions trials"}

    cond = condition_summary(df)
    by_eval = accuracy_by(df, "evaluator_persona", condition=ACTIVE)
    by_source = accuracy_by(df, "source_persona", condition=ACTIVE)
    by_src_adv = self_advantage_by_source(df)
    by_group_adv = self_advantage_by_group(df)
    avn = active_vs_neutral(df)
    adv = self_advantage(df)
    cm_active = confusion_matrix(df, condition=ACTIVE)
    cm_neutral = confusion_matrix(df, condition=NEUTRAL)
    guess_by_cond = guess_counts_by_condition(df)
    prob_active = mean_probability_by_persona(df, condition=ACTIVE)
    prob_neutral = mean_probability_by_persona(df, condition=NEUTRAL)
    eval_guesses = evaluator_guess_breakdown(df)
    class_rates = evaluator_classification_rates(df)

    cond.to_csv(run_dir / "descriptions_condition_summary.csv")
    by_eval.to_csv(run_dir / "descriptions_accuracy_by_evaluator.csv")
    by_source.to_csv(run_dir / "descriptions_accuracy_by_source.csv")
    by_src_adv.to_csv(run_dir / "descriptions_self_advantage_by_source.csv")
    if not by_group_adv.empty:
        by_group_adv.to_csv(run_dir / "descriptions_self_advantage_by_task_set.csv")
    avn.to_csv(run_dir / "descriptions_active_vs_neutral.csv")
    if not cm_active.empty:
        cm_active.to_csv(run_dir / "descriptions_confusion_active.csv")
    if not cm_neutral.empty:
        cm_neutral.to_csv(run_dir / "descriptions_confusion_neutral.csv")
    if not guess_by_cond.empty:
        guess_by_cond.to_csv(run_dir / "descriptions_guess_counts_by_condition.csv")
    if not prob_active.empty:
        prob_active.to_csv(run_dir / "descriptions_mean_prob_by_persona_active.csv")
    if not prob_neutral.empty:
        prob_neutral.to_csv(run_dir / "descriptions_mean_prob_by_persona_neutral.csv")
    if not eval_guesses.empty:
        eval_guesses.to_csv(run_dir / "descriptions_evaluator_guess_breakdown.csv")
    if not class_rates.empty:
        class_rates.to_csv(run_dir / "descriptions_evaluator_classification_rates.csv")

    metrics = {
        "model": str(df["model"].iloc[0]) if "model" in df else None,
        "candidate_display_mode": (str(d["candidate_display_mode"].iloc[0])
                                   if "candidate_display_mode" in d else None),
        "condition_summary": cond.reset_index().to_dict("records"),
        "self_advantage": adv,
        "accuracy_by_evaluator": by_eval.reset_index().to_dict("records"),
        "self_advantage_by_source": by_src_adv.reset_index().to_dict("records"),
        "self_advantage_by_task_set": by_group_adv.reset_index().to_dict("records"),
    }
    _write_markdown(run_dir / "descriptions_summary.md", metrics, cond, adv, by_src_adv, avn)
    return metrics


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) and pd.notna(v) else "n/a"


def _write_markdown(path: Path, m: dict, cond: pd.DataFrame, adv: dict,
                    by_src_adv: pd.DataFrame, avn: pd.DataFrame) -> None:
    lines = [
        f"# Persona-source classification (all_persona_descriptions) — {m.get('model')}",
        "",
        f"- candidate display mode: **{m.get('candidate_display_mode')}**",
        "",
        "Each generated text is shown with a shuffled candidate set (A/B/C/D…); the",
        "evaluator returns a probability distribution over which candidate wrote it.",
        "",
        "## Per-condition overview",
        "",
        "| condition | n | parse-fail | top-1 acc | correct-prob | chance |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c, r in cond.iterrows():
        lines.append(f"| {c} | {int(r['n_trials'])} | {_fmt(r['parse_failure_rate'])} | "
                     f"{_fmt(r['top1_accuracy'])} | {_fmt(r['mean_correct_probability'])} | "
                     f"{_fmt(r['chance'])} |")
    lines += [
        "",
        "## Self-advantage (active-persona rows)",
        "",
        "Does an evaluator classify its OWN generations better than other "
        "evaluators classify that same source?",
        "",
        f"- self-advantage (accuracy): **{_fmt(adv.get('self_advantage_acc'))}**  "
        f"(self {_fmt(adv.get('self_accuracy'))} vs nonself {_fmt(adv.get('nonself_accuracy'))}; "
        f"n_self={adv.get('n_self')}, n_nonself={adv.get('n_nonself')})",
        f"- self-advantage (correct-prob): **{_fmt(adv.get('self_advantage_prob'))}**  "
        f"(self {_fmt(adv.get('self_correct_probability'))} vs nonself "
        f"{_fmt(adv.get('nonself_correct_probability'))})",
        "",
        "### By source persona (self evaluator vs other evaluators)",
        "",
        "| source | self acc | other acc | Δacc | self prob | other prob | Δprob |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for src, r in by_src_adv.iterrows():
        lines.append(f"| {src} | {_fmt(r['self_accuracy'])} | {_fmt(r['other_accuracy'])} | "
                     f"{_fmt(r['self_advantage_acc'])} | {_fmt(r['self_correct_probability'])} | "
                     f"{_fmt(r['other_correct_probability'])} | {_fmt(r['self_advantage_prob'])} |")
    lines += [
        "",
        "## Active-self vs neutral description-only baseline",
        "",
        "Isolates any active-self gain over ordinary prompt-based classification.",
        "",
        "| source | active-self acc | neutral acc | Δacc | active-self prob | neutral prob | Δprob |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for src, r in avn.iterrows():
        lines.append(
            f"| {src} | {_fmt(r['active_self_accuracy'])} | {_fmt(r['neutral_accuracy'])} | "
            f"{_fmt(r['active_self_minus_neutral_acc'])} | "
            f"{_fmt(r['active_self_correct_probability'])} | "
            f"{_fmt(r['neutral_correct_probability'])} | "
            f"{_fmt(r['active_self_minus_neutral_prob'])} |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")
