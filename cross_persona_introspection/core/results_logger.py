"""Incremental JSONL results logging."""

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from cross_persona_introspection.schemas import TrialRecord

logger = logging.getLogger(__name__)


class ResultsLogger:
    """Append-only JSONL logger. Each trial is written immediately so
    prior successful trials are never lost if a later trial fails."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log_trial(self, record: TrialRecord) -> None:
        """Append one trial record as a JSONL line."""
        if record.timestamp is None:
            record.timestamp = datetime.now(timezone.utc).isoformat()

        data = asdict(record)
        with open(self.output_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def log_error(self, experiment: str, task_id: str, error: str, **extra) -> None:
        """Log a failed trial with error info."""
        record = TrialRecord(
            experiment=experiment,
            model="",
            task_id=task_id,
            task_set="",
            error=error,
            **extra,
        )
        self.log_trial(record)
        logger.error(f"Trial error [{experiment}/{task_id}]: {error}")


def load_results(path: str | Path) -> pd.DataFrame:
    """Load a JSONL results file into a pandas DataFrame."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)
