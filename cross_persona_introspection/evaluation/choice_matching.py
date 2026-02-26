"""Evaluation: choice matching and scoring."""

from cross_persona_introspection.schemas import TrialRecord


def score_trial(record: TrialRecord) -> TrialRecord:
    """Score a cross-persona prediction trial.

    Computes:
    - choice_match: exact match of predicted vs actual answer
    - confidence_error: absolute difference of predicted vs actual confidence
    - refuse_match: exact match of predicted vs actual refusal
    """
    if record.predicted_answer is not None and record.actual_answer is not None:
        record.choice_match = record.predicted_answer.upper() == record.actual_answer.upper()

    if record.predicted_confidence is not None and record.actual_confidence is not None:
        record.confidence_error = abs(record.predicted_confidence - record.actual_confidence)

    if record.predicted_refuse is not None and record.actual_refuse is not None:
        record.refuse_match = record.predicted_refuse == record.actual_refuse

    return record


def score_reporter_trial(record: TrialRecord) -> TrialRecord:
    """Score a source-reporter matrix trial.

    Compares reporter's report to source-state metrics.
    """
    if record.source_metrics and record.reporter_answer:
        source_top1 = record.source_metrics.get("top1_answer")
        if source_top1:
            record.choice_match = record.reporter_answer.upper() == source_top1.upper()

    if record.source_metrics and record.reporter_confidence is not None:
        source_conf = record.source_metrics.get("confidence_proxy_bin")
        if source_conf is not None:
            record.confidence_error = abs(record.reporter_confidence - source_conf)

    return record
