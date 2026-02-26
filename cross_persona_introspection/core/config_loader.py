"""Load config YAML files for prompts and scoring thresholds."""

from pathlib import Path
from functools import lru_cache

import yaml

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


@lru_cache
def load_prompts() -> dict:
    """Load prompt templates from config/prompts.yaml."""
    with open(CONFIG_DIR / "prompts.yaml") as f:
        return yaml.safe_load(f)


@lru_cache
def load_scoring() -> dict:
    """Load scoring config from config/scoring.yaml."""
    with open(CONFIG_DIR / "scoring.yaml") as f:
        return yaml.safe_load(f)


def get_response_format(choices: list[str]) -> str:
    """Build the response format string with actual choices filled in."""
    prompts = load_prompts()
    choice_str = ", ".join(choices) if choices else "your best answer"
    return prompts["response_format"].format(choice_str=choice_str).strip()


def prob_to_confidence_bin(top1_prob: float) -> int:
    """Map a top-1 probability to a confidence bin using config thresholds."""
    scoring = load_scoring()
    for threshold, bin_value in scoring["confidence_bins"]:
        if top1_prob >= threshold:
            return bin_value
    return 1
