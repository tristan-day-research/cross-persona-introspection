"""Analysis utilities for summarizing experiment results."""

import json
from pathlib import Path

import pandas as pd

from cross_persona_introspection.core.results_logger import load_results


def summarize_cross_persona(results_path: str | Path) -> dict:
    """Summarize cross-persona prediction results."""
    df = load_results(results_path)
    df_valid = df[df["error"].isna()]

    if df_valid.empty:
        return {"error": "no valid trials"}

    summary = {
        "total_trials": len(df_valid),
        "total_errors": len(df) - len(df_valid),
        "overall_choice_match": df_valid["choice_match"].mean(),
        "overall_confidence_error": df_valid["confidence_error"].abs().mean(),
    }

    # Per reporter-target pair
    pair_stats = {}
    for (r, t), group in df_valid.groupby(["reporter_persona", "source_persona"]):
        pair_stats[f"{r}->{t}"] = {
            "n": len(group),
            "choice_match": group["choice_match"].mean(),
            "confidence_error": group["confidence_error"].abs().mean(),
        }
    summary["per_pair"] = pair_stats

    # Per task set
    task_stats = {}
    for ts, group in df_valid.groupby("task_set"):
        task_stats[ts] = {
            "n": len(group),
            "choice_match": group["choice_match"].mean(),
        }
    summary["per_task_set"] = task_stats

    return summary


def summarize_source_reporter(results_path: str | Path) -> dict:
    """Summarize source-reporter matrix results."""
    df = load_results(results_path)

    summary = {}
    for exp_type in df["experiment"].unique():
        sub = df[df["experiment"] == exp_type]
        sub_valid = sub[sub["error"].isna()]

        if sub_valid.empty:
            summary[exp_type] = {"n": 0}
            continue

        cells = {}
        for (s, r), group in sub_valid.groupby(["source_persona", "reporter_persona"]):
            cells[f"{s}->{r}"] = {
                "n": len(group),
                "choice_match": group["choice_match"].mean(),
                "confidence_error": group["confidence_error"].abs().mean() if "confidence_error" in group else None,
            }

        summary[exp_type] = {
            "n_trials": len(sub_valid),
            "overall_choice_match": sub_valid["choice_match"].mean(),
            "cells": cells,
        }

    # Compute cache advantage: matrix accuracy - baseline accuracy
    matrix_acc = summary.get("source_reporter_matrix", {}).get("overall_choice_match")
    baseline_acc = summary.get("source_reporter_matrix_baseline", {}).get("overall_choice_match")
    if matrix_acc is not None and baseline_acc is not None:
        summary["cache_advantage"] = matrix_acc - baseline_acc

    return summary


def print_summary(summary: dict, indent: int = 2) -> None:
    """Pretty-print a summary dict."""
    print(json.dumps(summary, indent=indent, default=str))
