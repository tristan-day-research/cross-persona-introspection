"""Build task sets from HuggingFace datasets for the confidence-vs-entropy experiment.

Loads questions from MMLU, CommonsenseQA, TruthfulQA, and a local art JSON file,
normalizes them all to the shared TaskItem schema, and caches the result locally.

Scientific note: All personas answer the SAME shared question set.
Do NOT create separate datasets per persona.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

from cross_persona_introspection.schemas import TaskItem

logger = logging.getLogger(__name__)


def format_mc_prompt(question: str, choices: dict[str, str]) -> str:
    """Format a question + choices into inline MCQ format matching existing conventions.

    Args:
        question: The question text.
        choices: Mapping like {"A": "answer text", "B": "answer text", ...}.

    Returns:
        Formatted string like:
            What is X?

            A) answer one
            B) answer two
            ...
    """
    lines = [question, ""]
    for label in sorted(choices.keys()):
        lines.append(f"{label}) {choices[label]}")
    return "\n".join(lines)


def _ensure_balanced_answers(tasks: list[TaskItem], seed: int = 42) -> list[TaskItem]:
    """Shuffle tasks so that correct answers are not all the same letter.

    If more than 40% of answers share one letter, re-shuffle choice positions
    for a random subset to rebalance. This modifies tasks in place.
    """
    if not tasks:
        return tasks

    # Check distribution
    from collections import Counter
    answer_counts = Counter(t.expected_answer for t in tasks if t.expected_answer)
    if not answer_counts:
        return tasks

    total = sum(answer_counts.values())
    max_frac = max(answer_counts.values()) / total if total > 0 else 0

    if max_frac <= 0.40:
        return tasks  # already balanced enough

    logger.info(f"Answer distribution imbalanced ({dict(answer_counts)}), reshuffling choices")
    rng = random.Random(seed)
    letters = ["A", "B", "C", "D"]

    for task in tasks:
        if not task.expected_answer or not task.metadata.get("_choice_texts"):
            continue

        choice_texts = task.metadata["_choice_texts"]
        correct_idx = letters.index(task.expected_answer) if task.expected_answer in letters else None
        if correct_idx is None:
            continue

        # Shuffle the choice order
        indices = list(range(len(choice_texts)))
        rng.shuffle(indices)

        new_choices = {letters[i]: choice_texts[idx] for i, idx in enumerate(indices)}
        new_correct = letters[indices.index(correct_idx)]

        task.prompt = format_mc_prompt(task.metadata.get("_question_text", ""), new_choices)
        task.expected_answer = new_correct
        task.choices = sorted(new_choices.keys())

    # Clean up temporary metadata
    for task in tasks:
        task.metadata.pop("_choice_texts", None)
        task.metadata.pop("_question_text", None)

    return tasks


def build_mmlu_tasks(
    subjects: list[tuple[str, int]],
    domain: str,
    id_prefix: str,
    seed: int = 42,
) -> list[TaskItem]:
    """Load MMLU questions for given subjects, normalize to TaskItem.

    Args:
        subjects: List of (subject_name, num_questions) tuples.
        domain: Domain tag (e.g., "chemistry", "history").
        id_prefix: Prefix for task IDs (e.g., "chem", "hist").
        seed: Random seed for shuffling before selection.

    Returns:
        List of TaskItem objects.
    """
    from datasets import load_dataset

    letters = ["A", "B", "C", "D"]
    all_tasks = []
    task_counter = 0

    for subject, num_questions in subjects:
        logger.info(f"Loading MMLU subject: {subject} (need {num_questions})")
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception:
            # Fallback to older dataset ID
            dataset = load_dataset("lukaemon/mmlu", subject, split="test", trust_remote_code=True)

        # Shuffle with seed for reproducible selection
        indices = list(range(len(dataset)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        selected = indices[:num_questions]

        for idx in selected:
            row = dataset[idx]
            question_text = row["question"]
            choice_texts = row["choices"]
            correct_idx = row["answer"]  # 0-3 index

            choices_dict = {letters[i]: choice_texts[i] for i in range(len(choice_texts))}
            prompt = format_mc_prompt(question_text, choices_dict)

            task = TaskItem(
                task_id=f"{id_prefix}_{task_counter:03d}",
                prompt=prompt,
                task_set="confidence_entropy",
                format="constrained",
                choices=letters[:len(choice_texts)],
                expected_answer=letters[correct_idx],
                metadata={
                    "domain": domain,
                    "source_dataset": f"mmlu_{subject}",
                    "subject": subject,
                    "_choice_texts": choice_texts,
                    "_question_text": question_text,
                },
            )
            all_tasks.append(task)
            task_counter += 1

    return all_tasks


def build_commonsenseqa_tasks(
    num_questions: int = 25,
    id_prefix: str = "gen",
    seed: int = 42,
) -> list[TaskItem]:
    """Load CommonsenseQA validation split, normalize to TaskItem.

    CommonsenseQA has 5 choices (A-E). We keep all 5 as-is.
    """
    from datasets import load_dataset

    logger.info(f"Loading CommonsenseQA (need {num_questions})")
    dataset = load_dataset("commonsense_qa", split="validation", trust_remote_code=True)

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected = indices[:num_questions]

    tasks = []
    for i, idx in enumerate(selected):
        row = dataset[idx]
        question_text = row["question"]
        choice_labels = row["choices"]["label"]  # ["A", "B", "C", "D", "E"]
        choice_texts = row["choices"]["text"]
        correct_answer = row["answerKey"]  # "A"-"E"

        choices_dict = {label: text for label, text in zip(choice_labels, choice_texts)}
        prompt = format_mc_prompt(question_text, choices_dict)

        task = TaskItem(
            task_id=f"{id_prefix}_{i:03d}",
            prompt=prompt,
            task_set="confidence_entropy",
            format="constrained",
            choices=choice_labels,
            expected_answer=correct_answer,
            metadata={
                "domain": "general",
                "source_dataset": "commonsenseqa",
                "_choice_texts": choice_texts,
                "_question_text": question_text,
            },
        )
        tasks.append(task)

    return tasks


def build_truthfulqa_tasks(
    num_questions: int = 20,
    id_prefix: str = "misc",
    seed: int = 42,
) -> list[TaskItem]:
    """Load TruthfulQA multiple-choice questions, normalize to TaskItem.

    Uses the mc1_targets format (single correct answer).
    If loading fails, returns a placeholder list with a TODO note.
    """
    from datasets import load_dataset

    logger.info(f"Loading TruthfulQA MC (need {num_questions})")
    letters = ["A", "B", "C", "D"]

    try:
        dataset = load_dataset("truthful_qa", "multiple_choice", split="validation", trust_remote_code=True)
    except Exception as exc:
        logger.warning(f"Could not load TruthfulQA: {exc}. Using placeholder.")
        # TODO: Fill in TruthfulQA questions manually or fix loader
        return _truthfulqa_placeholder(num_questions, id_prefix)

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    tasks = []
    for idx in indices:
        if len(tasks) >= num_questions:
            break

        row = dataset[idx]
        question_text = row["question"]
        mc1_choices = row["mc1_targets"]["choices"]
        mc1_labels = row["mc1_targets"]["labels"]  # 1 for correct, 0 for wrong

        # Take exactly 4 choices: the correct one + 3 wrong ones
        correct_indices = [i for i, label in enumerate(mc1_labels) if label == 1]
        wrong_indices = [i for i, label in enumerate(mc1_labels) if label == 0]

        if not correct_indices or len(wrong_indices) < 3:
            continue

        selected_wrong = rng.sample(wrong_indices, 3)
        selected_indices = correct_indices[:1] + selected_wrong
        rng.shuffle(selected_indices)

        choice_texts = [mc1_choices[i] for i in selected_indices]
        correct_pos = selected_indices.index(correct_indices[0])

        choices_dict = {letters[i]: choice_texts[i] for i in range(4)}
        prompt = format_mc_prompt(question_text, choices_dict)

        task = TaskItem(
            task_id=f"{id_prefix}_{len(tasks):03d}",
            prompt=prompt,
            task_set="confidence_entropy",
            format="constrained",
            choices=letters,
            expected_answer=letters[correct_pos],
            metadata={
                "domain": "misconception",
                "source_dataset": "truthfulqa",
                "_choice_texts": choice_texts,
                "_question_text": question_text,
            },
        )
        tasks.append(task)

    if len(tasks) < num_questions:
        logger.warning(f"TruthfulQA: only got {len(tasks)}/{num_questions} valid questions")

    return tasks


def _truthfulqa_placeholder(num_questions: int, id_prefix: str) -> list[TaskItem]:
    """Return placeholder tasks when TruthfulQA can't be loaded."""
    # TODO: Replace with actual TruthfulQA questions or fix the loader
    tasks = []
    for i in range(num_questions):
        task = TaskItem(
            task_id=f"{id_prefix}_{i:03d}",
            prompt=f"[PLACEHOLDER] TruthfulQA question {i+1}\n\nA) Option A\nB) Option B\nC) Option C\nD) Option D",
            task_set="confidence_entropy",
            format="constrained",
            choices=["A", "B", "C", "D"],
            expected_answer="A",
            metadata={
                "domain": "misconception",
                "source_dataset": "truthfulqa_placeholder",
                "notes": "TODO: Replace with real TruthfulQA questions",
            },
        )
        tasks.append(task)
    return tasks


def load_art_tasks(path: Path) -> list[TaskItem]:
    """Load hand-authored art questions from a local JSON file.

    The file should use the standard TaskItem JSON format (same as tasks/*.json).
    """
    from cross_persona_introspection.core.task_loader import load_task_file

    logger.info(f"Loading art tasks from: {path}")
    tasks = load_task_file(path)

    # Ensure domain metadata is set
    for task in tasks:
        task.metadata.setdefault("domain", "art")
        task.metadata.setdefault("source_dataset", "manual")
        task.task_set = "confidence_entropy"

    return tasks


def build_confidence_entropy_dataset(
    art_path: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    seed: int = 42,
) -> list[TaskItem]:
    """Build the full ~150-question dataset for the confidence-entropy experiment.

    Loads from cache if available, otherwise builds fresh and caches.

    Dataset composition:
        Chemistry: 40 (MMLU high_school_chemistry + college_chemistry)
        History:   40 (MMLU us/world/european history)
        Art:       up to 25 (manual JSON file)
        General:   25 (CommonsenseQA)
        Misconception: 20 (TruthfulQA MC)

    Args:
        art_path: Path to hand-authored art questions JSON. None to skip art.
        cache_path: Path to cache the built dataset. None to skip caching.
        seed: Random seed for reproducible dataset construction.
    """
    # Try loading from cache first
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached dataset from: {cache_path}")
        from cross_persona_introspection.core.task_loader import load_task_file
        return load_task_file(cache_path)

    logger.info("Building confidence-entropy dataset from source datasets...")

    # Chemistry: 40 questions
    chemistry_tasks = build_mmlu_tasks(
        subjects=[("high_school_chemistry", 20), ("college_chemistry", 20)],
        domain="chemistry",
        id_prefix="chem",
        seed=seed,
    )

    # History: 40 questions
    history_tasks = build_mmlu_tasks(
        subjects=[
            ("high_school_us_history", 15),
            ("high_school_world_history", 15),
            ("high_school_european_history", 10),
        ],
        domain="history",
        id_prefix="hist",
        seed=seed,
    )

    # Art: up to 25 questions (manual)
    art_tasks = []
    if art_path and art_path.exists():
        art_tasks = load_art_tasks(art_path)
        logger.info(f"Loaded {len(art_tasks)} art tasks")
    else:
        logger.warning("No art tasks file found. Skipping art domain.")

    # General: 25 questions (CommonsenseQA)
    general_tasks = build_commonsenseqa_tasks(
        num_questions=25,
        id_prefix="gen",
        seed=seed,
    )

    # Misconception: 20 questions (TruthfulQA)
    misconception_tasks = build_truthfulqa_tasks(
        num_questions=20,
        id_prefix="misc",
        seed=seed,
    )

    all_tasks = chemistry_tasks + history_tasks + art_tasks + general_tasks + misconception_tasks

    # Ensure answer distribution is not too skewed
    all_tasks = _ensure_balanced_answers(all_tasks, seed=seed)

    logger.info(
        f"Built dataset: {len(all_tasks)} total — "
        f"chem={len(chemistry_tasks)}, hist={len(history_tasks)}, "
        f"art={len(art_tasks)}, gen={len(general_tasks)}, "
        f"misc={len(misconception_tasks)}"
    )

    # Cache to disk for reproducibility
    if cache_path:
        _cache_dataset(all_tasks, cache_path)

    return all_tasks


def _cache_dataset(tasks: list[TaskItem], cache_path: Path) -> None:
    """Save built dataset to JSON for reproducible reruns."""
    from dataclasses import asdict

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for task in tasks:
        data = asdict(task)
        # Remove internal-only metadata keys
        data["metadata"] = {
            k: v for k, v in data["metadata"].items()
            if not k.startswith("_")
        }
        records.append(data)

    with open(cache_path, "w") as f:
        json.dump(records, f, indent=2)

    logger.info(f"Cached dataset to: {cache_path}")
