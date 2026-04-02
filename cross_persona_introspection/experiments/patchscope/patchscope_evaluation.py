"""Aggregate metrics for a completed Patchscope run.

Computes accuracy / agreement rates per template and condition from
a list of ``PatchscopeRecord`` objects.  Stateless — receives the
record list and returns a plain dict of metrics.
"""

from cross_persona_introspection.schemas import PatchscopeRecord


def compute_metrics(records: list[PatchscopeRecord]) -> dict:
    """Return aggregate accuracy metrics broken down by template × condition.

    Metrics produced (when the relevant records exist):

    * ``answer_extraction_{condition}_accuracy`` — fraction of records where
      the reporter's answer matches the source's direct answer.
    * ``persona_probe_{condition}_accuracy`` — fraction where the reporter
      correctly identifies the source persona's political leaning.
    * ``self_recognition_{condition}_accuracy`` — fraction where the reporter
      correctly labels the activation as SELF vs OTHER.
    * ``mean_relevancy`` — average SelfIE relevancy score across all records
      that have one.
    """
    if not records:
        return {"status": "no records"}

    total = len(records)
    errors = sum(1 for r in records if r.error)
    constrained = [r for r in records if r.reporter_parse_success]

    metrics = {
        "total_records": total,
        "errors": errors,
        "parsed_constrained": len(constrained),
    }

    # -- answer_extraction: reporter answer == source direct answer
    for condition in ("real", "text_only_baseline", "shuffled"):
        recs = [
            r for r in constrained
            if r.template_name == "answer_extraction"
            and r.condition == condition
            and r.source_last_prefill_answer is not None
        ]
        if recs:
            correct = sum(
                1 for r in recs
                if r.reporter_parsed_answer and r.reporter_parsed_answer[0] == r.source_last_prefill_answer
            )
            metrics[f"answer_extraction_{condition}_accuracy"] = round(
                correct / len(recs), 4
            )
            metrics[f"answer_extraction_{condition}_n"] = len(recs)

    # -- persona_probe: political leaning detection
    for condition in ("real", "text_only_baseline", "shuffled"):
        recs = [
            r for r in constrained
            if r.template_name == "persona_probe" and r.condition == condition
        ]
        if recs:
            correct = 0
            for r in recs:
                expected = (
                    "CONSERVATIVE" if "conservative" in r.source_persona
                    else "PROGRESSIVE"
                )
                if r.reporter_parsed_answer and expected.startswith(r.reporter_parsed_answer[:4]):
                    correct += 1
            metrics[f"persona_probe_{condition}_accuracy"] = round(
                correct / len(recs), 4
            )
            metrics[f"persona_probe_{condition}_n"] = len(recs)

    # -- self_recognition: SELF when source == reporter, OTHER otherwise
    for condition in ("real",):
        recs = [
            r for r in constrained
            if r.template_name == "self_recognition" and r.condition == condition
        ]
        if recs:
            correct = sum(
                1 for r in recs
                if r.reporter_parsed_answer == ("SELF" if r.source_persona == r.reporter_persona else "OTHER")
            )
            metrics[f"self_recognition_{condition}_accuracy"] = round(
                correct / len(recs), 4
            )
            metrics[f"self_recognition_{condition}_n"] = len(recs)

    # -- mean relevancy (SelfIE)
    rel_recs = [r for r in records if r.reporter_mean_relevancy is not None]
    if rel_recs:
        metrics["reporter_mean_relevancy"] = round(
            sum(r.reporter_mean_relevancy for r in rel_recs) / len(rel_recs), 6
        )

    return metrics
