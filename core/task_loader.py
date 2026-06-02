"""Task loading and sampling."""

import json
import random
from pathlib import Path
from typing import Optional

from core.schemas import TaskItem

TASKS_DIR = Path(__file__).parent.parent / "tasks"


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
    task_sets: list,
    tasks_dir: Optional[Path] = None,
) -> list[TaskItem]:
    """Load tasks from named task sets, optionally filtered to specific task_ids.

    Set names map to JSON files: "factual" -> tasks/factual_tasks.json.

    Each entry in `task_sets` may be either:
      - a string set name              -> load the *entire* set, e.g. "neutral"
      - a single-key mapping            -> load only the listed task_ids from that
        {set_name: [task_id, ...]}         set, e.g. {"misaligned": ["sr_22", "sr_25"]}
        (a null/empty value means the whole set, same as the string form)

    Entries may be mixed freely, so a config can take an entire set for one and
    specific questions for the other:

        task_sets:
          - neutral                       # all of the neutral set
          - misaligned: [sr_22, sr_25]    # only these two from the misaligned set
    """
    tasks_dir = tasks_dir or TASKS_DIR
    all_tasks = []

    for entry in task_sets:
        if isinstance(entry, str):
            name, wanted_ids = entry, None
        elif isinstance(entry, dict):
            if len(entry) != 1:
                raise ValueError(
                    "Each task_sets mapping entry must have exactly one set name, "
                    f"got: {entry!r}"
                )
            (name, wanted_ids), = entry.items()
            if wanted_ids is not None and not isinstance(wanted_ids, list):
                raise ValueError(
                    f"task_ids for set {name!r} must be a list of ids, got: {wanted_ids!r}"
                )
        else:
            raise TypeError(
                "task_sets entries must be a string set name or a single-key "
                f"{{set_name: [task_ids]}} mapping, got: {entry!r}"
            )

        path = tasks_dir / f"{name}_tasks.json"
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {path}")
        items = load_task_file(path)

        if wanted_ids:
            by_id = {t.task_id: t for t in items}
            missing = [tid for tid in wanted_ids if tid not in by_id]
            if missing:
                raise ValueError(f"task_ids not found in set {name!r}: {missing}")
            items = [by_id[tid] for tid in wanted_ids]

        all_tasks.extend(items)

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
