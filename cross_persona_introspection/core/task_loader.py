"""Task loading and sampling."""

import json
import random
from pathlib import Path
from typing import Optional

from cross_persona_introspection.schemas import TaskItem

TASKS_DIR = Path(__file__).parent.parent.parent / "tasks"


def load_task_file(path: Path) -> list[TaskItem]:
    """Load a JSON task file into a list of TaskItems."""
    with open(path) as f:
        data = json.load(f)

    items = []
    for entry in data:
        items.append(TaskItem(
            task_id=entry["task_id"],
            prompt=entry["prompt"],
            task_set=entry.get("task_set", path.stem),
            format=entry.get("format", "constrained"),
            choices=entry.get("choices", []),
            expected_answer=entry.get("expected_answer"),
            metadata=entry.get("metadata", {}),
        ))
    return items


def load_tasks(
    task_sets: list[str],
    tasks_dir: Optional[Path] = None,
) -> list[TaskItem]:
    """Load tasks from named task sets.

    Task set names map to JSON files: "factual" -> tasks/factual_tasks.json
    """
    tasks_dir = tasks_dir or TASKS_DIR
    all_tasks = []

    for name in task_sets:
        path = tasks_dir / f"{name}_tasks.json"
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {path}")
        all_tasks.extend(load_task_file(path))

    return all_tasks


def sample_tasks(
    tasks: list[TaskItem],
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> list[TaskItem]:
    """Optionally sample a subset of tasks with a fixed seed."""
    if sample_size is None or sample_size >= len(tasks):
        return tasks
    rng = random.Random(seed)
    return rng.sample(tasks, sample_size)
