"""CLI entrypoint for running experiments.

Usage:
    python run.py <config_name>                    # run a named config
    python run.py --list                           # list all available configs
    python run.py <name> --override sample_size=2  # override one or more fields

Each experiment lives at experiments/<name>/ with a single `config.yaml`
holding everything the experiment needs:

    personas:                # registry of personas this experiment uses
      <persona_name>:
        system_prompt: "..."
        description: "..."

    prompts:                 # optional — prompt templates if any
      <key>: "..."

    configs:                 # named run configs
      <config_name>:
        model_name: "..."
        personas: [<persona_name>, ...]
        ...

This script discovers configs by walking experiments/*/config.yaml.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.schemas import PersonaConfig, RunConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Silence the harmless `max_new_tokens (=N) and max_length (=M) seem to have been set`
# warning that transformers prints on every generate() call.
try:
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    pass

EXPERIMENTS_DIR = ROOT / "experiments"


# ── Loading ──────────────────────────────────────────────────────────────


def _load_experiment_yaml(exp_dir: Path) -> dict:
    """Load and validate one experiment's config.yaml."""
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        data = yaml.safe_load(f) or {}
    return data


def discover_configs() -> dict[str, tuple[str, dict]]:
    """Scan experiments/*/config.yaml. Return {config_name: (experiment_folder, run_config_dict)}."""
    configs: dict[str, tuple[str, dict]] = {}
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        data = _load_experiment_yaml(exp_dir)
        for name, cfg in (data.get("configs") or {}).items():
            if name in configs:
                raise ValueError(f"Duplicate config name '{name}' in {exp_dir}")
            configs[name] = (exp_dir.name, cfg)
    return configs


def load_personas_for(experiment_folder: str) -> dict[str, PersonaConfig]:
    """Load this experiment's persona registry from its own config.yaml."""
    data = _load_experiment_yaml(EXPERIMENTS_DIR / experiment_folder)
    return {
        name: PersonaConfig(
            name=name,
            system_prompt=cfg.get("system_prompt", ""),
            description=cfg.get("description", ""),
        )
        for name, cfg in (data.get("personas") or {}).items()
    }


def load_prompts_for(experiment_folder: str) -> dict[str, str]:
    """Load this experiment's prompt templates from its own config.yaml."""
    data = _load_experiment_yaml(EXPERIMENTS_DIR / experiment_folder)
    return data.get("prompts") or {}


# ── Override / RunConfig assembly ────────────────────────────────────────


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply CLI overrides like 'sample_size=5'. JSON-parsed when possible."""
    for override in overrides:
        if "=" not in override:
            logger.warning(f"Ignoring malformed override: {override}")
            continue
        key, value = override.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config


def build_run_config(experiment_name: str, exp_config: dict) -> RunConfig:
    """Convert a raw config dict to a RunConfig. experiment_name comes from the folder."""
    return RunConfig(
        experiment_name=experiment_name,
        model_name=exp_config["model_name"],
        personas=exp_config.get("personas", []),
        task_sets=exp_config.get("task_sets", []),
        sample_size=exp_config.get("sample_size"),
        seed=exp_config.get("seed", 42),
        max_new_tokens=exp_config.get("max_new_tokens", 256),
        temperature=exp_config.get("temperature", 0.0),
        output_dir=exp_config.get("output_dir", f"experiments/{experiment_name}/results"),
        pause_cue=exp_config.get("pause_cue", ""),
        few_shot_mode=exp_config.get("few_shot_mode", "fixed"),
        use_persona_suffixes=exp_config.get("use_persona_suffixes", False),
        confidence_legend=exp_config.get("confidence_legend", "bins"),
        task_file=exp_config.get("task_file"),
        patchscope_config=exp_config.get("patchscope_config"),
        model_dtype=exp_config.get("model_dtype"),
        batch_size=exp_config.get("batch_size", 1),
        openrouter_model=exp_config.get("openrouter_model"),
        openrouter_api_key=exp_config.get("openrouter_api_key"),
    )


# ── Dispatch ─────────────────────────────────────────────────────────────


def build_experiment(experiment_name: str, run_config: RunConfig, exp_config: dict, personas: dict[str, PersonaConfig]):
    """Import experiments/<name>/experiment.py and instantiate the right class."""
    module = importlib.import_module(f"experiments.{experiment_name}.experiment")

    used_personas = {k: v for k, v in personas.items() if k in run_config.personas}
    missing = set(run_config.personas) - set(used_personas)
    if missing:
        raise ValueError(f"Personas not found in {experiment_name}/config.yaml: {missing}")

    if experiment_name == "confidence_entropy":
        # Two variants in one folder: instruct (experiment.py) and base (experiment_base.py).
        variant = exp_config.get("variant", "instruct")
        if variant == "base":
            base_module = importlib.import_module("experiments.confidence_entropy.experiment_base")
            return base_module.ConfidenceEntropyBase(run_config, used_personas)
        return module.ConfidenceEntropy(run_config, used_personas)
    if experiment_name == "activation_probing":
        return module.ActivationProbing(run_config, used_personas)
    if experiment_name == "patchscope":
        return module.PatchscopeExperiment(run_config, personas)  # patchscope uses full registry
    if experiment_name == "persona_self_recognition":
        return module.PersonaSelfRecognition(run_config, used_personas)
    raise ValueError(f"Unknown experiment folder: {experiment_name}")


def run_one(config_name: str, overrides: list[str] | None = None) -> None:
    configs = discover_configs()
    if config_name not in configs:
        available = "\n  ".join(sorted(configs))
        raise ValueError(f"Config '{config_name}' not found. Available:\n  {available}")
    experiment_name, exp_config = configs[config_name]
    if overrides:
        exp_config = apply_overrides(exp_config, overrides)

    run_config = build_run_config(experiment_name, exp_config)
    personas = load_personas_for(experiment_name)

    logger.info(f"Running config '{config_name}' (experiment: {experiment_name})")
    experiment = build_experiment(experiment_name, run_config, exp_config, personas)
    experiment.setup()
    experiment.run()

    metrics = experiment.evaluate()
    output_path = experiment.save_results()
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Summary metrics: {json.dumps(metrics, indent=2, default=str)}")


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument("config", nargs="?", help="Config name (top-level key under `configs:` in some experiments/*/config.yaml)")
    parser.add_argument("--list", action="store_true", help="List all available configs and exit")
    parser.add_argument("--all", action="store_true", help="Run every config (rarely useful)")
    parser.add_argument("--override", nargs="*", default=[], help="Field overrides like sample_size=2")
    args = parser.parse_args()

    if args.list:
        for name, (exp, _) in sorted(discover_configs().items()):
            print(f"  {name}    [{exp}]")
        return

    if args.all:
        for name in sorted(discover_configs()):
            try:
                run_one(name, args.override)
            except Exception as e:
                logger.error(f"Config '{name}' failed: {e}")
        return

    if not args.config:
        parser.error("config is required (or pass --list)")

    run_one(args.config, args.override)


if __name__ == "__main__":
    main()
