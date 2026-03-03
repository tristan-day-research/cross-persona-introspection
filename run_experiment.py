"""CLI entrypoint for running PSM introspection experiments."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_project_root = str(Path(__file__).parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml

from cross_persona_introspection.schemas import PersonaConfig, RunConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "config"


def load_personas(path: Path | None = None) -> dict[str, PersonaConfig]:
    """Load persona configs from YAML."""
    path = path or CONFIG_DIR / "personas.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    personas = {}
    for name, cfg in data.items():
        personas[name] = PersonaConfig(
            name=name,
            system_prompt=cfg.get("system_prompt", ""),
            description=cfg.get("description", ""),
        )
    return personas


def load_experiment_config(name: str, path: Path | None = None) -> dict:
    """Load a named experiment config from YAML."""
    path = path or CONFIG_DIR / "experiments.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    if name not in data:
        available = ", ".join(data.keys())
        raise ValueError(f"Experiment config '{name}' not found. Available: {available}")
    return data[name]


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply simple dot-path overrides like 'sample_size=5'."""
    for override in overrides:
        if "=" not in override:
            logger.warning(f"Ignoring malformed override: {override}")
            continue
        key, value = override.split("=", 1)

        # Try to parse value as JSON for proper typing
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # keep as string

        # Support dot paths like tasks.sample_size
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return config


def build_run_config(exp_config: dict) -> RunConfig:
    """Convert a raw config dict to a RunConfig."""
    return RunConfig(
        experiment_name=exp_config["experiment_name"],
        model_name=exp_config["model_name"],
        personas=exp_config["personas"],
        task_sets=exp_config["task_sets"],
        sample_size=exp_config.get("sample_size"),
        seed=exp_config.get("seed", 42),
        max_new_tokens=exp_config.get("max_new_tokens", 256),
        temperature=exp_config.get("temperature", 0.0),
        output_dir=exp_config.get("output_dir", "results/raw"),
        pause_cue=exp_config.get("pause_cue", "\n[PAUSE: Before answering, note your current inclination.]\n"),
        few_shot_mode=exp_config.get("few_shot_mode", "fixed"),
        use_persona_suffixes=exp_config.get("use_persona_suffixes", False),
        openrouter_model=exp_config.get("openrouter_model"),
        openrouter_api_key=exp_config.get("openrouter_api_key"),
    )


def run_one(config_name: str, overrides: list[str] | None = None) -> None:
    """Run a single named experiment configuration."""
    exp_config = load_experiment_config(config_name)
    if overrides:
        exp_config = apply_overrides(exp_config, overrides)

    run_config = build_run_config(exp_config)
    personas = load_personas()

    # Filter to only requested personas
    used_personas = {k: v for k, v in personas.items() if k in run_config.personas}
    missing = set(run_config.personas) - set(used_personas.keys())
    if missing:
        raise ValueError(f"Personas not found in personas.yaml: {missing}")

    experiment_name = run_config.experiment_name

    if experiment_name == "cross_persona_prediction":
        from cross_persona_introspection.experiments.cross_persona_prediction import CrossPersonaPrediction
        experiment = CrossPersonaPrediction(run_config, used_personas)
    elif experiment_name == "source_reporter_matrix":
        from cross_persona_introspection.experiments.source_reporter_matrix import SourceReporterMatrix
        experiment = SourceReporterMatrix(run_config, used_personas)
    elif experiment_name == "confidence_entropy":
        from cross_persona_introspection.experiments.confidence_entropy import ConfidenceEntropy
        experiment = ConfidenceEntropy(run_config, used_personas)
    elif experiment_name == "confidence_entropy_base":
        from cross_persona_introspection.experiments.confidence_entropy_base import ConfidenceEntropyBase
        experiment = ConfidenceEntropyBase(run_config, used_personas)
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    logger.info(f"Running experiment: {config_name} ({experiment_name})")
    experiment.setup()
    experiment.run()

    metrics = experiment.evaluate()
    output_path = experiment.save_results()

    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Summary metrics: {json.dumps(metrics, indent=2, default=str)}")


def main():
    parser = argparse.ArgumentParser(description="Run PSM introspection experiments")
    parser.add_argument("config", nargs="?", help="Named experiment config from experiments.yaml")
    parser.add_argument("--all", action="store_true", help="Run all experiment configs")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides like sample_size=5")
    parser.add_argument("--list", action="store_true", help="List available experiment configs")

    args = parser.parse_args()

    if args.list:
        path = CONFIG_DIR / "experiments.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        for name in data:
            print(f"  {name}")
        return

    if not args.list and not args.all and not args.config:
        parser.error("config is required unless --list or --all is specified")

    if args.all:
        path = CONFIG_DIR / "experiments.yaml"
        with open(path) as f:
            data = yaml.safe_load(f)
        for name in data:
            try:
                run_one(name, args.override)
            except Exception as e:
                logger.error(f"Experiment '{name}' failed: {e}")
    else:
        run_one(args.config, args.override)


if __name__ == "__main__":
    main()
