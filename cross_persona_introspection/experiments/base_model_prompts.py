"""Few-shot prompt formatting for base (non-instruct) models.

Base models don't follow instructions reliably. Instead, we use few-shot
examples to establish the expected input/output pattern, then let the model
continue the pattern via next-token prediction.

For persona induction, we prepend a short context line describing the persona
before the few-shot block. This is the base-model equivalent of a system prompt.

Adapted from external reference code for the PSM introspection framework.
"""

import random
from typing import Optional


# ============================================================================
# Few-shot example pools
# ============================================================================

# Fixed few-shot examples for MC answer extraction.
# These are trivial questions the model should get right, establishing the
# "Question → Answer: <letter>" pattern.
MC_FEW_SHOT_FIXED = [
    {
        "question": "What planet is known as the Red Planet?",
        "options": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"},
        "answer": "B",
    },
    {
        "question": "What is the chemical symbol for water?",
        "options": {"A": "CO2", "B": "NaCl", "C": "H2O", "D": "O2"},
        "answer": "C",
    },
    {
        "question": "In which year did World War II end?",
        "options": {"A": "1943", "B": "1944", "C": "1945", "D": "1946"},
        "answer": "C",
    },
]

# Fixed few-shot examples for confidence rating.
# Stratified across the S-Z scale so the model sees low, mid, and high confidence.
CONFIDENCE_FEW_SHOT_FIXED = [
    {
        "question": "What planet is known as the Red Planet?",
        "options": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"},
        "confidence": "Z",  # very easy → high confidence
    },
    {
        "question": "What is the 21 trillionth digit of pi?",
        "options": {"A": "3", "B": "7", "C": "1", "D": "9"},
        "confidence": "S",  # impossible → lowest confidence
    },
    {
        "question": "Who wrote the novel 'War and Peace'?",
        "options": {"A": "Charles Dickens", "B": "Leo Tolstoy", "C": "Mark Twain", "D": "Jane Austen"},
        "confidence": "Y",  # well-known → high confidence
    },
    {
        "question": "Which enzyme primarily catalyzes the conversion of superoxide radicals?",
        "options": {"A": "Catalase", "B": "Superoxide dismutase", "C": "Peroxidase", "D": "Glutathione reductase"},
        "confidence": "V",  # domain-specific → moderate confidence
    },
    {
        "question": "The Treaty of Tordesillas was signed in which year?",
        "options": {"A": "1492", "B": "1494", "C": "1500", "D": "1488"},
        "confidence": "W",  # somewhat obscure → mid confidence
    },
]


# ============================================================================
# Persona context lines for base models
# ============================================================================

# For base models, persona induction is done by prepending a brief context
# sentence rather than a system prompt. The model treats this as part of
# the document it's continuing.

BASE_PERSONA_CONTEXTS = {
    "default_assistant": "",  # no extra context
    "chemist": "The following answers are from an expert chemist with deep knowledge of organic, inorganic, and physical chemistry.\n\n",
    "historian": "The following answers are from an expert historian with deep knowledge of world, US, and European history.\n\n",
    "artist": "The following answers are from an expert in art history, art movements, and artistic techniques.\n\n",
    "cautious_hedging": "The following answers are from someone who is very cautious and uncertain, tending to report low confidence.\n\n",
    "bold_assertive": "The following answers are from someone who is very confident and assertive, tending to report high confidence.\n\n",
    "five_year_old": "The following answers are from a 5-year-old child who uses simple words and sometimes gets confused.\n\n",
}


def get_persona_context(persona_name: str) -> str:
    """Get the base-model persona context prefix for a given persona name."""
    return BASE_PERSONA_CONTEXTS.get(persona_name, "")


# ============================================================================
# MC answer prompt formatting
# ============================================================================

def _format_mc_few_shot_block(examples: list[dict]) -> str:
    """Format a list of MC few-shot examples into text."""
    text = ""
    for ex in examples:
        text += "Question: " + ex["question"] + "\n"
        for key, value in ex["options"].items():
            text += f"  {key}: {value}\n"
        text += "Answer: " + ex["answer"] + "\n\n"
    return text


def format_mc_prompt_base(
    question_text: str,
    options: dict[str, str],
    persona_name: str = "default_assistant",
) -> str:
    """Format a multiple-choice question for a base model.

    Returns a complete prompt string ending with "Answer:" that the model
    should complete with a single letter (A/B/C/D).

    Args:
        question_text: The question text (without options).
        options: Dict mapping option letters to option text, e.g. {"A": "...", ...}.
        persona_name: Name of the persona for context prefix.

    Returns:
        Complete prompt string ready for text completion.
    """
    persona_ctx = get_persona_context(persona_name)

    prompt = persona_ctx
    prompt += "The following are multiple choice questions with answers.\n\n"
    prompt += _format_mc_few_shot_block(MC_FEW_SHOT_FIXED)

    # Now the actual question
    prompt += "Question: " + question_text + "\n"
    for key, value in options.items():
        prompt += f"  {key}: {value}\n"
    prompt += "Answer:"

    return prompt


# ============================================================================
# Confidence rating prompt formatting
# ============================================================================

def _format_confidence_few_shot_block(examples: list[dict]) -> str:
    """Format a list of confidence few-shot examples into text."""
    text = ""
    for ex in examples:
        text += "Question: " + ex["question"] + "\n"
        for key, value in ex["options"].items():
            text += f"  {key}: {value}\n"
        text += "Confidence: " + ex["confidence"] + "\n\n"
    return text


def format_confidence_prompt_base(
    question_text: str,
    options: dict[str, str],
    persona_name: str = "default_assistant",
) -> str:
    """Format a confidence rating question for a base model.

    Returns a prompt ending with "Confidence:" that the model should
    complete with a single letter (S-Z).

    The confidence scale:
        S: <5%, T: 5-10%, U: 10-20%, V: 20-40%,
        W: 40-60%, X: 60-80%, Y: 80-90%, Z: >90%

    Args:
        question_text: The question text (without options).
        options: Dict mapping option letters to option text.
        persona_name: Name of the persona for context prefix.

    Returns:
        Complete prompt string ready for text completion.
    """
    persona_ctx = get_persona_context(persona_name)

    prompt = persona_ctx
    prompt += "For each question, rate your confidence that you know the correct answer.\n"
    prompt += "S: <5%, T: 5-10%, U: 10-20%, V: 20-40%, W: 40-60%, X: 60-80%, Y: 80-90%, Z: >90%\n\n"
    prompt += _format_confidence_few_shot_block(CONFIDENCE_FEW_SHOT_FIXED)

    # Now the actual question
    prompt += "Question: " + question_text + "\n"
    for key, value in options.items():
        prompt += f"  {key}: {value}\n"
    prompt += "Confidence:"

    return prompt
