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
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _freeze_project_imports() -> None:
    """Eagerly import every .py file under core/ and experiments/ so that
    later lazy imports inside functions become sys.modules cache hits and
    never re-read disk. This makes the run resilient to source edits made
    while the experiment is in flight — the in-memory bytecode is locked
    in at process start, regardless of what happens to the .py files later.

    Skipped subpaths: __pycache__, .ipynb_checkpoints, anything starting
    with '.' or '_'. Files that fail to import are reported but don't abort
    startup (e.g. helper scripts with their own __main__-only side effects).
    """
    skip_parts = {"__pycache__", ".ipynb_checkpoints"}
    for pkg in ("core", "experiments"):
        pkg_root = ROOT / pkg
        if not pkg_root.is_dir():
            continue
        for py in sorted(pkg_root.rglob("*.py")):
            rel = py.relative_to(ROOT)
            if any(p in skip_parts or p.startswith(".") for p in rel.parts):
                continue
            mod_name = ".".join(rel.with_suffix("").parts)
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                # Don't kill startup over an unrelated module failing to load.
                print(f"[freeze] skipped {mod_name}: {type(e).__name__}: {e}", file=sys.stderr)


_freeze_project_imports()

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


def _find_experiment_config(exp_dir: Path) -> Path | None:
    """Locate an experiment's config YAML. Accepts `config.yaml` or a prefixed
    name like `<prefix>_config.yaml` (e.g. `self_recognition_config.yaml`)."""
    candidate = exp_dir / "config.yaml"
    if candidate.exists():
        return candidate
    matches = sorted(p for p in exp_dir.glob("*_config.yaml") if p.is_file())
    if len(matches) > 1:
        raise ValueError(
            f"{exp_dir.name}: multiple *_config.yaml files found, expected one: {matches}"
        )
    return matches[0] if matches else None


def _find_experiment_module_name(exp_dir: Path) -> str | None:
    """Return the python module name (relative to the experiment folder) for
    this experiment. Prefers `experiment` and falls back to a prefixed
    `<prefix>_experiment` file."""
    if (exp_dir / "experiment.py").exists():
        return "experiment"
    matches = sorted(p.stem for p in exp_dir.glob("*_experiment.py") if p.is_file())
    if len(matches) > 1:
        raise ValueError(
            f"{exp_dir.name}: multiple *_experiment.py files found, expected one: {matches}"
        )
    return matches[0] if matches else None


def _load_experiment_yaml(exp_dir: Path) -> dict:
    """Load one experiment's config YAML (either `config.yaml` or a prefixed name)."""
    cfg_path = _find_experiment_config(exp_dir)
    if cfg_path is None:
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
            system_prompt=(cfg or {}).get("system_prompt", ""),
            description=(cfg or {}).get("description", ""),
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
        adapter=exp_config.get("adapter"),
        num_gpus=exp_config.get("num_gpus"),
    )


# ── Dispatch ─────────────────────────────────────────────────────────────


def build_experiment(experiment_name: str, run_config: RunConfig, exp_config: dict, personas: dict[str, PersonaConfig], config_name: str = "", shard=None):
    """Import the experiment module and instantiate the right class.

    The module file may be named `experiment.py` or, for folders that prefix
    their files, `<prefix>_experiment.py` (e.g. `self_recognition_experiment.py`).
    """
    exp_dir = EXPERIMENTS_DIR / experiment_name
    module_basename = _find_experiment_module_name(exp_dir)
    if module_basename is None:
        raise ValueError(
            f"No experiment module found in {exp_dir} "
            f"(expected `experiment.py` or `*_experiment.py`)"
        )
    module = importlib.import_module(f"experiments.{experiment_name}.{module_basename}")

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
        return module.PersonaSelfRecognition(run_config, used_personas, config_name=config_name, shard=shard)
    raise ValueError(f"Unknown experiment folder: {experiment_name}")


# Experiments that support data-parallel multi-GPU sharding (two-stage:
# generation, then recognition). Others always run single-process.
_SHARDABLE_EXPERIMENTS = {"persona_self_recognition"}


def resolve_num_gpus(run_config: RunConfig) -> int:
    """How many worker processes / GPUs to use.

    config `num_gpus`: None → auto-detect all visible CUDA devices; else the
    explicit count (clamped to >=1). Auto-detect falls back to 1 when torch or
    CUDA is unavailable (e.g. CPU/MPS box), so behavior is unchanged there.
    """
    requested = run_config.num_gpus
    if requested is not None:
        return max(1, int(requested))
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


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

    num_gpus = resolve_num_gpus(run_config)
    if experiment_name in _SHARDABLE_EXPERIMENTS and num_gpus > 1:
        logger.info(f"Running config '{config_name}' across {num_gpus} GPUs (data-parallel)")
        run_distributed(config_name, experiment_name, exp_config, run_config, personas, num_gpus, overrides)
        return

    logger.info(f"Running config '{config_name}' (experiment: {experiment_name})")
    experiment = build_experiment(experiment_name, run_config, exp_config, personas, config_name=config_name)
    experiment.setup()
    experiment.run()

    metrics = experiment.evaluate()
    output_path = experiment.save_results()
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Summary metrics: {json.dumps(metrics, indent=2, default=str)}")


# ── Multi-GPU data-parallel orchestration ─────────────────────────────────


def run_distributed(config_name, experiment_name, exp_config, run_config, personas,
                    num_gpus: int, overrides: list[str] | None) -> None:
    """Two-stage sharded run, one worker process per GPU.

    Stage 1 (generation) and stage 2 (recognition) each fan out across GPUs;
    the recognition stage starts only after every generation shard has been
    merged, so each recognition worker sees the full text pool. Shards write to
    run_dir/_shards/<stage>_<i>.jsonl, which the parent merges into the single
    canonical run_dir/trials.jsonl.
    """
    # A shard=None instance fixes the run dir / id and exposes the canonical
    # output path — it loads no model (that happens in setup(), which we skip).
    parent = build_experiment(experiment_name, run_config, exp_config, personas, config_name=config_name)
    run_dir = Path(parent.run_dir)
    run_id = parent.run_id
    shards_dir = run_dir / "_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"

    # Stage 1: generation
    _spawn_stage(config_name, overrides, "generation", num_gpus, run_dir, run_id, generations_path=None)
    _merge_shards(shards_dir, "generation", num_gpus, trials_path, append=False)

    # Stage 2: recognition (text pool = the merged generation rows)
    _spawn_stage(config_name, overrides, "recognition", num_gpus, run_dir, run_id, generations_path=trials_path)
    _merge_shards(shards_dir, "recognition", num_gpus, trials_path, append=True)

    metrics = parent.evaluate()
    output_path = parent.save_results()
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Summary metrics: {json.dumps(metrics, indent=2, default=str)}")


def _spawn_stage(config_name, overrides, stage: str, num_gpus: int,
                 run_dir: Path, run_id: str, generations_path) -> None:
    """Launch one worker per GPU for `stage` and wait for all to finish."""
    procs = []
    for i in range(num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)  # each worker sees exactly its GPU as cuda:0
        cmd = [
            sys.executable, str(ROOT / "run.py"), config_name,
            "--shard-stage", stage,
            "--shard-index", str(i),
            "--num-shards", str(num_gpus),
            "--run-dir", str(run_dir),
            "--run-id", run_id,
        ]
        if generations_path is not None:
            cmd += ["--generations-path", str(generations_path)]
        if overrides:
            cmd += ["--override", *overrides]
        logger.info(f"[{stage}] launching shard {i}/{num_gpus} on GPU {i}")
        procs.append(subprocess.Popen(cmd, env=env))

    failures = []
    for i, p in enumerate(procs):
        if p.wait() != 0:
            failures.append((i, p.returncode))
    if failures:
        raise RuntimeError(f"{stage} stage failed for shard(s): {failures}")


def _merge_shards(shards_dir: Path, stage: str, num_shards: int, target: Path, append: bool) -> None:
    """Concatenate run_dir/_shards/<stage>_<i>.jsonl into the canonical trials file."""
    n = 0
    with open(target, "a" if append else "w") as out:
        for i in range(num_shards):
            shard_file = shards_dir / f"{stage}_{i}.jsonl"
            if not shard_file.exists():
                continue
            with open(shard_file) as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    logger.info(f"merged {n} {stage} rows -> {target}")


def run_worker(config_name: str, overrides: list[str] | None, stage: str,
               shard_index: int, num_shards: int, run_dir: str, run_id: str,
               generations_path: str | None) -> None:
    """Single-shard worker entry point (invoked as a subprocess by _spawn_stage)."""
    from experiments.persona_self_recognition.self_recognition_experiment import ShardSpec

    configs = discover_configs()
    experiment_name, exp_config = configs[config_name]
    if overrides:
        exp_config = apply_overrides(exp_config, overrides)
    run_config = build_run_config(experiment_name, exp_config)
    personas = load_personas_for(experiment_name)

    shards_dir = Path(run_dir) / "_shards"
    shard = ShardSpec(
        stage=stage,
        shard_index=shard_index,
        num_shards=num_shards,
        run_dir=Path(run_dir),
        run_id=run_id,
        output_path=shards_dir / f"{stage}_{shard_index}.jsonl",
        generations_path=Path(generations_path) if generations_path else None,
        write_manifest=(stage == "generation" and shard_index == 0),
    )
    experiment = build_experiment(experiment_name, run_config, exp_config, personas,
                                  config_name=config_name, shard=shard)
    experiment.setup()
    experiment.run()
    logger.info(f"[{stage} shard {shard_index}/{num_shards}] done -> {shard.output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument("config", nargs="?", help="Config name (top-level key under `configs:` in some experiments/*/config.yaml)")
    parser.add_argument("--list", action="store_true", help="List all available configs and exit")
    parser.add_argument("--all", action="store_true", help="Run every config (rarely useful)")
    parser.add_argument("--override", nargs="*", default=[], help="Field overrides like sample_size=2")
    # Internal: set by the multi-GPU launcher when spawning a single shard worker.
    # Not intended for manual use — `num_gpus` in the config drives sharding.
    parser.add_argument("--shard-stage", choices=["generation", "recognition"], help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--generations-path", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.shard_stage:
        if not args.config:
            parser.error("config is required for a shard worker")
        run_worker(
            args.config, args.override, args.shard_stage, args.shard_index,
            args.num_shards, args.run_dir, args.run_id, args.generations_path,
        )
        return

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
