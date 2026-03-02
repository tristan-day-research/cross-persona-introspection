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


def parse_json_response(raw_text: str) -> tuple[str | None, float | None, bool]:
    """Parse a JSON-formatted response: {"answer": "A", "confidence_prob": 0.73}.

    Returns (answer, confidence_prob, parse_success).
    Tries direct JSON parse first, then regex fallback for malformed output.
    """
    import json

    answer = None
    confidence_prob = None

    # Try to extract a JSON object from the text
    json_str = None

    # First: try the full text as JSON
    stripped = raw_text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            json_str = stripped
        except json.JSONDecodeError:
            pass

    # Second: find first {...} block in the text
    if json_str is None:
        brace_match = re.search(r"\{[^{}]+\}", raw_text)
        if brace_match:
            try:
                data = json.loads(brace_match.group())
                json_str = brace_match.group()
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    # Extract fields from parsed JSON
    if isinstance(data, dict):
        raw_answer = data.get("answer", "")
        if isinstance(raw_answer, str) and re.match(r"^[A-Ea-e]$", raw_answer.strip()):
            answer = raw_answer.strip().upper()

        raw_conf = data.get("confidence_prob")
        if raw_conf is not None:
            try:
                confidence_prob = float(raw_conf)
                if not (0.0 <= confidence_prob <= 1.0):
                    confidence_prob = max(0.0, min(1.0, confidence_prob))
            except (ValueError, TypeError):
                pass

    # Regex fallback if JSON parsing didn't find an answer
    if answer is None:
        ans_match = re.search(r'"answer"\s*:\s*"([A-Ea-e])"', raw_text, re.IGNORECASE)
        if ans_match:
            answer = ans_match.group(1).upper()

    if confidence_prob is None:
        conf_match = re.search(r'"confidence_prob"\s*:\s*([\d.]+)', raw_text)
        if conf_match:
            try:
                confidence_prob = float(conf_match.group(1))
                if not (0.0 <= confidence_prob <= 1.0):
                    confidence_prob = max(0.0, min(1.0, confidence_prob))
            except ValueError:
                pass

    parse_success = answer is not None
    return answer, confidence_prob, parse_success
