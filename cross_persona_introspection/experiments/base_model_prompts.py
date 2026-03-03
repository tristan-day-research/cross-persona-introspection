"""Few-shot prompt formatting for base (non-instruct) models.

Base models don't follow instructions reliably. Instead, we use few-shot
examples to establish the expected input/output pattern, then let the model
continue the pattern via next-token prediction.

For persona induction, we prepend a short context line describing the persona
before the few-shot block. This is the base-model equivalent of a system prompt.

Adapted from external reference code for the PSM introspection framework.
"""

import random as _random
from typing import Optional


# ============================================================================
# Few-shot example pools
# ============================================================================

# Fixed few-shot examples for MC answer extraction (used in "fixed" mode).
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

# Larger pool for random sampling. Covers different domains and answer positions
# (A/B/C/D) to avoid positional bias in the few-shot examples.
MC_FEW_SHOT_POOL = [
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
    {
        "question": "What is the largest organ in the human body?",
        "options": {"A": "Skin", "B": "Liver", "C": "Brain", "D": "Heart"},
        "answer": "A",
    },
    {
        "question": "Who painted the Mona Lisa?",
        "options": {"A": "Michelangelo", "B": "Raphael", "C": "Rembrandt", "D": "Leonardo da Vinci"},
        "answer": "D",
    },
    {
        "question": "What is the speed of light in a vacuum approximately?",
        "options": {"A": "300,000 km/s", "B": "150,000 km/s", "C": "500,000 km/s", "D": "100,000 km/s"},
        "answer": "A",
    },
    {
        "question": "Which element has the atomic number 1?",
        "options": {"A": "Helium", "B": "Oxygen", "C": "Carbon", "D": "Hydrogen"},
        "answer": "D",
    },
    {
        "question": "What is the capital of Japan?",
        "options": {"A": "Seoul", "B": "Tokyo", "C": "Beijing", "D": "Bangkok"},
        "answer": "B",
    },
    {
        "question": "Which Shakespeare play features the character Ophelia?",
        "options": {"A": "Macbeth", "B": "Othello", "C": "Hamlet", "D": "King Lear"},
        "answer": "C",
    },
    {
        "question": "What gas do plants absorb from the atmosphere during photosynthesis?",
        "options": {"A": "Oxygen", "B": "Nitrogen", "C": "Hydrogen", "D": "Carbon dioxide"},
        "answer": "D",
    },
    {
        "question": "The Great Wall of China was primarily built to protect against invasions from which direction?",
        "options": {"A": "North", "B": "South", "C": "East", "D": "West"},
        "answer": "A",
    },
    {
        "question": "What is the boiling point of water at sea level in Celsius?",
        "options": {"A": "90°C", "B": "100°C", "C": "110°C", "D": "120°C"},
        "answer": "B",
    },
]

# Fixed few-shot examples for confidence rating (used in "fixed" mode).
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

# Larger pool for random sampling. Covers the full S-Z confidence range.
# Labels are assigned based on how well-known the question is (not model-specific).
CONFIDENCE_FEW_SHOT_POOL = [
    # S: <5% — effectively impossible/random guessing
    {
        "question": "What is the 21 trillionth digit of pi?",
        "options": {"A": "3", "B": "7", "C": "1", "D": "9"},
        "confidence": "S",
    },
    {
        "question": "How many grains of sand are on Bondi Beach to the nearest billion?",
        "options": {"A": "12 billion", "B": "87 billion", "C": "204 billion", "D": "531 billion"},
        "confidence": "S",
    },
    # T: 5-10% — very obscure, slightly better than random
    {
        "question": "What is the exact population of Liechtenstein as of January 1, 2023?",
        "options": {"A": "39,327", "B": "38,254", "C": "40,101", "D": "37,910"},
        "confidence": "T",
    },
    {
        "question": "In what year was the first rubber band produced commercially?",
        "options": {"A": "1845", "B": "1823", "C": "1860", "D": "1833"},
        "confidence": "T",
    },
    # U: 10-20% — obscure but recognizable domain
    {
        "question": "Which enzyme primarily catalyzes the conversion of superoxide radicals?",
        "options": {"A": "Catalase", "B": "Superoxide dismutase", "C": "Peroxidase", "D": "Glutathione reductase"},
        "confidence": "U",
    },
    {
        "question": "Who composed the opera 'Rusalka'?",
        "options": {"A": "Smetana", "B": "Janáček", "C": "Dvořák", "D": "Martinů"},
        "confidence": "U",
    },
    # V: 20-40% — moderately difficult, some clues available
    {
        "question": "The Treaty of Tordesillas was signed in which year?",
        "options": {"A": "1492", "B": "1494", "C": "1500", "D": "1488"},
        "confidence": "V",
    },
    {
        "question": "What is the half-life of Carbon-14?",
        "options": {"A": "2,730 years", "B": "5,730 years", "C": "8,730 years", "D": "11,730 years"},
        "confidence": "V",
    },
    # W: 40-60% — coin flip territory, might know it
    {
        "question": "Which country has the longest coastline in the world?",
        "options": {"A": "Russia", "B": "Canada", "C": "Indonesia", "D": "Australia"},
        "confidence": "W",
    },
    {
        "question": "What year was the first photograph taken?",
        "options": {"A": "1816", "B": "1826", "C": "1839", "D": "1842"},
        "confidence": "W",
    },
    # X: 60-80% — fairly confident but not certain
    {
        "question": "What is the capital of Australia?",
        "options": {"A": "Sydney", "B": "Melbourne", "C": "Canberra", "D": "Brisbane"},
        "confidence": "X",
    },
    {
        "question": "Who wrote 'One Hundred Years of Solitude'?",
        "options": {"A": "Jorge Luis Borges", "B": "Gabriel García Márquez", "C": "Pablo Neruda", "D": "Isabel Allende"},
        "confidence": "X",
    },
    # Y: 80-90% — quite confident
    {
        "question": "Who wrote the novel 'War and Peace'?",
        "options": {"A": "Charles Dickens", "B": "Leo Tolstoy", "C": "Mark Twain", "D": "Jane Austen"},
        "confidence": "Y",
    },
    {
        "question": "What is the largest planet in our solar system?",
        "options": {"A": "Saturn", "B": "Neptune", "C": "Jupiter", "D": "Uranus"},
        "confidence": "Y",
    },
    # Z: >90% — near certain
    {
        "question": "What planet is known as the Red Planet?",
        "options": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"},
        "confidence": "Z",
    },
    {
        "question": "What is the boiling point of water at sea level in Celsius?",
        "options": {"A": "90°C", "B": "100°C", "C": "110°C", "D": "120°C"},
        "confidence": "Z",
    },
]


def _sample_stratified(pool: list[dict], n: int, key: str) -> list[dict]:
    """Sample n examples from pool, stratified by the given key.

    Tries to cover as many distinct values of key as possible,
    then fills remaining slots randomly. Shuffles the result.
    """
    by_value: dict[str, list[dict]] = {}
    for ex in pool:
        val = ex[key]
        by_value.setdefault(val, []).append(ex)

    groups = list(by_value.keys())
    _random.shuffle(groups)

    selected = []
    used_indices = set()
    for group_key in groups:
        if len(selected) >= n:
            break
        choice = _random.choice(by_value[group_key])
        selected.append(choice)
        used_indices.add(id(choice))

    if len(selected) < n:
        remaining = [ex for ex in pool if id(ex) not in used_indices]
        if remaining:
            extra = _random.sample(remaining, min(n - len(selected), len(remaining)))
            selected.extend(extra)

    _random.shuffle(selected)
    return selected


def _sample_balanced(pool: list[dict], key: str, target_values: list[str]) -> list[dict]:
    """Sample exactly one example per target value, covering the full scale.

    For confidence: target_values = ["S","T","U","V","W","X","Y","Z"] → 8 examples.
    For MC answers: target_values = ["A","B","C","D"] → 4 examples.

    If a target value has no examples in the pool, finds the nearest
    neighbor value as a substitute (labeled with the target value).
    Shuffles the final order to avoid positional bias.

    Returns list of (example_dict, label) where label is the target value
    to use (may differ from the example's original value if substituted).
    The returned dicts have the key overwritten with the target label.
    """
    by_value: dict[str, list[dict]] = {}
    for ex in pool:
        val = ex[key]
        by_value.setdefault(val, []).append(ex)

    selected = []
    for target in target_values:
        if by_value.get(target):
            ex = _random.choice(by_value[target])
        else:
            # Find nearest neighbor in target_values ordering
            idx = target_values.index(target)
            ex = None
            for offset in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7]:
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(target_values):
                    neighbor = target_values[neighbor_idx]
                    if by_value.get(neighbor):
                        ex = _random.choice(by_value[neighbor])
                        break
            if ex is None:
                continue  # pool has no usable examples at all

        # Copy and relabel with target value so the model sees uniform coverage
        relabeled = dict(ex)
        relabeled[key] = target
        selected.append(relabeled)

    _random.shuffle(selected)
    return selected


# ============================================================================
# Persona context lines for base models
# ============================================================================

BASE_PERSONA_CONTEXTS = {
    "default_assistant": "",
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
    mode: str = "fixed",
) -> str:
    """Format a multiple-choice question for a base model.

    Returns a complete prompt string ending with "Answer: " (with trailing
    space) so the model's next-token prediction is the bare letter (A/B/C/D)
    rather than " A" (space+letter), which is a different token.

    Args:
        question_text: The question text (without options).
        options: Dict mapping option letters to option text, e.g. {"A": "...", ...}.
        persona_name: Name of the persona for context prefix.
        mode: "fixed" uses the same 3 examples every time.
              "random" samples 3 from a larger pool (different per call).
              "balanced" shows one example per answer position (A/B/C/D = 4
              examples), randomly selected and shuffled each call.

    Returns:
        Complete prompt string ready for text completion.
    """
    persona_ctx = get_persona_context(persona_name)

    if mode == "fixed":
        examples = MC_FEW_SHOT_FIXED
    elif mode == "random":
        examples = _sample_stratified(MC_FEW_SHOT_POOL, 3, key="answer")
    elif mode == "balanced":
        examples = _sample_balanced(MC_FEW_SHOT_POOL, key="answer",
                                    target_values=["A", "B", "C", "D"])
    else:
        raise ValueError(f"Unknown few_shot_mode: {mode!r}. Use 'fixed', 'random', or 'balanced'.")

    prompt = persona_ctx
    prompt += "The following are multiple choice questions with answers.\n\n"
    prompt += _format_mc_few_shot_block(examples)

    # Now the actual question
    prompt += "Question: " + question_text + "\n"
    for key, value in options.items():
        prompt += f"  {key}: {value}\n"
    # Trailing space is critical: makes the model predict the bare letter
    # token ("B") instead of space+letter (" B"), which is a different token
    # in Llama's vocabulary. Without it, logprob extraction looks up the wrong token.
    prompt += "Answer: "

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
    mode: str = "fixed",
) -> str:
    """Format a confidence rating question for a base model.

    Returns a prompt ending with "Confidence: " (trailing space) so the
    model's next-token prediction is the bare letter (S-Z).

    The confidence scale:
        S: <5%, T: 5-10%, U: 10-20%, V: 20-40%,
        W: 40-60%, X: 60-80%, Y: 80-90%, Z: >90%

    Args:
        question_text: The question text (without options).
        options: Dict mapping option letters to option text.
        persona_name: Name of the persona for context prefix.
        mode: "fixed" uses the same 5 examples every time.
              "random" samples 3 from a larger pool (different per call).
              "balanced" shows one example per confidence level (S-Z = 8
              examples), randomly selected and shuffled each call. This
              gives the model uniform exposure to every confidence level.

    Returns:
        Complete prompt string ready for text completion.
    """
    persona_ctx = get_persona_context(persona_name)

    if mode == "fixed":
        examples = CONFIDENCE_FEW_SHOT_FIXED
    elif mode == "random":
        examples = _sample_stratified(CONFIDENCE_FEW_SHOT_POOL, 3, key="confidence")
    elif mode == "balanced":
        examples = _sample_balanced(CONFIDENCE_FEW_SHOT_POOL, key="confidence",
                                    target_values=["S", "T", "U", "V", "W", "X", "Y", "Z"])
    else:
        raise ValueError(f"Unknown few_shot_mode: {mode!r}. Use 'fixed', 'random', or 'balanced'.")

    prompt = persona_ctx
    prompt += "For each question, rate your confidence that you know the correct answer.\n"
    prompt += "S: <5%, T: 5-10%, U: 10-20%, V: 20-40%, W: 40-60%, X: 60-80%, Y: 80-90%, Z: >90%\n\n"
    prompt += _format_confidence_few_shot_block(examples)

    # Now the actual question
    prompt += "Question: " + question_text + "\n"
    for key, value in options.items():
        prompt += f"  {key}: {value}\n"
    # Trailing space — same rationale as MC prompt (see format_mc_prompt_base)
    prompt += "Confidence: "

    return prompt
