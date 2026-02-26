"""Parse structured responses from model outputs."""

import re
from cross_persona_introspection.schemas import ParsedResponse


def parse_constrained_response(raw_text: str) -> ParsedResponse:
    """Parse a constrained-format response.

    Expected format (flexible about whitespace/casing):
        Answer: B
        Confidence: 4
        Refuse: no
    """
    result = ParsedResponse(raw_text=raw_text)

    # Parse answer — look for "Answer: X" pattern
    answer_match = re.search(r"answer\s*:\s*([A-Da-d])\b", raw_text, re.IGNORECASE)
    if answer_match:
        result.answer = answer_match.group(1).upper()

    # Parse confidence — look for "Confidence: N"
    conf_match = re.search(r"confidence\s*:\s*([1-5])", raw_text, re.IGNORECASE)
    if conf_match:
        result.confidence = int(conf_match.group(1))

    # Parse refuse — look for "Refuse: yes/no"
    refuse_match = re.search(r"refuse\s*:\s*(yes|no)\b", raw_text, re.IGNORECASE)
    if refuse_match:
        result.refuse = refuse_match.group(1).lower() == "yes"

    result.parse_success = result.answer is not None
    return result


def parse_free_form_response(raw_text: str) -> ParsedResponse:
    """Parse a free-form response. Just wraps the raw text."""
    return ParsedResponse(
        raw_text=raw_text,
        parse_success=True,
    )


def parse_response(raw_text: str, format: str = "constrained") -> ParsedResponse:
    """Route to the appropriate parser based on format."""
    if format == "constrained":
        return parse_constrained_response(raw_text)
    return parse_free_form_response(raw_text)
