"""Evaluation: calibration metrics."""

from typing import Optional
import pandas as pd


def confidence_bin_error(df: pd.DataFrame) -> Optional[float]:
    """Mean absolute error between reported confidence and source confidence proxy.

    Expects columns: reporter_confidence, source_metrics (dict with confidence_proxy_bin).
    """
    errors = []
    for _, row in df.iterrows():
        rc = row.get("reporter_confidence")
        sm = row.get("source_metrics")
        if rc is not None and isinstance(sm, dict):
            sc = sm.get("confidence_proxy_bin")
            if sc is not None:
                errors.append(abs(rc - sc))
    return sum(errors) / len(errors) if errors else None


def confidence_correlation(df: pd.DataFrame) -> Optional[float]:
    """Pearson correlation between reporter confidence and source confidence proxy."""
    reporter_confs = []
    source_confs = []

    for _, row in df.iterrows():
        rc = row.get("reporter_confidence")
        sm = row.get("source_metrics")
        if rc is not None and isinstance(sm, dict):
            sc = sm.get("confidence_proxy_bin")
            if sc is not None:
                reporter_confs.append(rc)
                source_confs.append(sc)

    if len(reporter_confs) < 3:
        return None

    r_series = pd.Series(reporter_confs)
    s_series = pd.Series(source_confs)
    return r_series.corr(s_series)
