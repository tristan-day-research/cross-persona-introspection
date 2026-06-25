"""Small run-level utilities shared across the standalone experiment scripts
(generation, evaluation, …).

These were duplicated verbatim in generate_text.py and
evaluate_self_recognition.py; lifting them here keeps a single source of truth
for the filesystem-safe model tag, the provenance git stamp, and the
torch-dtype / GPU-count resolution every multi-GPU script needs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def model_slug(model_name: str) -> str:
    """Short, filesystem-safe tag for a model name (last path segment)."""
    return (model_name or "model").split("/")[-1].replace(".", "-")


def git_commit() -> str:
    """Short HEAD commit for manifest provenance, or "(unknown)" outside a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "(unknown)"


def resolve_dtype(model_dtype: str | None):
    """Map a config dtype string to a torch dtype (None → backend default)."""
    import torch
    return {
        "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32,
    }.get(model_dtype) if model_dtype else None


def resolve_num_gpus(run_config) -> int:
    """GPU fan-out: explicit `num_gpus` in the config, else CUDA device count, else 1."""
    if run_config.num_gpus is not None:
        return max(1, int(run_config.num_gpus))
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1
