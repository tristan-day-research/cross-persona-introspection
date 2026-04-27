"""Smoke tests: imports, config loading, parsing, and a mock dry run."""

import json
import tempfile
from pathlib import Path

import pytest


def test_imports():
    """All modules should import without error."""
    from core import schemas
    from core.backends import hf_backend, openrouter_backend
    from core import persona_inducer, task_loader, response_parser, results_logger
    from experiments import base
    from experiments.confidence_entropy import experiment as ce
    from experiments.activation_probing import experiment as ap
    from experiments.patchscope import experiment as ps
    from experiments.persona_self_recognition import experiment as psr
    from experiments.persona_self_recognition import analysis_helpers as psr_analysis


def test_schemas():
    """Schema dataclasses should instantiate correctly."""
    from core.schemas import (
        PersonaConfig, TaskItem, RunConfig, TrialRecord, ParsedResponse, SourceStateMetrics,
        PatchscopeRecord,
    )

    p = PersonaConfig(name="test", system_prompt="You are a test.", description="test persona")
    assert p.name == "test"

    t = TaskItem(task_id="t1", prompt="What?", task_set="factual", choices=["A", "B"])
    assert t.choices == ["A", "B"]

    rc = RunConfig(experiment_name="test", model_name="test-model", personas=["a"], task_sets=["factual"])
    assert rc.seed == 42

    sm = SourceStateMetrics(top1_answer="A", top1_prob=0.9, entropy=0.5)
    assert sm.confidence_proxy_bin is None


def test_response_parser():
    """Test constrained response parsing."""
    from core.response_parser import parse_constrained_response

    # Clean format
    result = parse_constrained_response("Answer: B\nConfidence: 4\nRefuse: no")
    assert result.answer == "B"
    assert result.confidence == 4
    assert result.refuse is False
    assert result.parse_success is True

    # Messy format
    result = parse_constrained_response("I think the answer is:\n  answer: c\n  confidence:  2\nrefuse:yes")
    assert result.answer == "C"
    assert result.confidence == 2
    assert result.refuse is True

    # Unparseable
    result = parse_constrained_response("I don't know what to say here.")
    assert result.parse_success is False


def test_persona_inducer():
    """Test persona induction with and without system prompt."""
    from core.persona_inducer import induce_persona
    from core.schemas import PersonaConfig

    persona = PersonaConfig(name="test", system_prompt="Be helpful.")
    messages = [{"role": "user", "content": "Hello"}]

    result = induce_persona(persona, messages)
    assert len(result) == 2
    assert result[0]["role"] == "system"

    # No system prompt
    stripped = PersonaConfig(name="stripped", system_prompt="")
    result = induce_persona(stripped, messages)
    assert len(result) == 1


def test_task_loader():
    """Test loading tasks from JSON."""
    from core.task_loader import load_task_file, sample_tasks
    from core.schemas import TaskItem

    tasks_data = [
        {"task_id": "t1", "prompt": "Q1?", "choices": ["A", "B"]},
        {"task_id": "t2", "prompt": "Q2?", "choices": ["A", "B"]},
        {"task_id": "t3", "prompt": "Q3?", "choices": ["A", "B"]},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(tasks_data, f)
        tmp_path = Path(f.name)

    items = load_task_file(tmp_path)
    assert len(items) == 3
    assert all(isinstance(i, TaskItem) for i in items)

    sampled = sample_tasks(items, sample_size=2, seed=42)
    assert len(sampled) == 2

    tmp_path.unlink()


def test_results_logger():
    """Test JSONL logging and loading."""
    from core.results_logger import ResultsLogger, load_results
    from core.schemas import TrialRecord

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp_path = Path(f.name)

    logger = ResultsLogger(tmp_path)

    record = TrialRecord(
        experiment="test",
        model="test-model",
        task_id="t1",
        task_set="factual",
        choice_match=True,
    )
    logger.log_trial(record)
    logger.log_trial(record)

    df = load_results(tmp_path)
    assert len(df) == 2
    assert df.iloc[0]["experiment"] == "test"

    tmp_path.unlink()


def test_self_recognition_record_and_analysis(tmp_path):
    """SelfRecognitionRecord serializes; analysis produces a matrix on a fixture."""
    from core.schemas import SelfRecognitionRecord
    from core.results_logger import ResultsLogger
    from experiments.persona_self_recognition.analysis_helpers import summarize_run

    jsonl_path = tmp_path / "trials.jsonl"
    log = ResultsLogger(jsonl_path)

    personas = ["a", "b"]
    for s in personas:
        log.log_trial(SelfRecognitionRecord(
            experiment="persona_self_recognition", phase="generation",
            model="m", task_id="t1", run_id="r1", source_persona=s,
            generated_text="hello", token_length=1,
        ))
    for s in personas:
        for e in personas:
            log.log_trial(SelfRecognitionRecord(
                experiment="persona_self_recognition", phase="individual",
                model="m", task_id="t1", run_id="r1",
                source_persona=s, evaluator_persona=e,
                parsed_choice="YES" if s == e else "NO",
                choice_probs={"YES": 0.9 if s == e else 0.1, "NO": 0.1 if s == e else 0.9},
                is_correct=True, has_ground_truth=True,
            ))
    log.log_trial(SelfRecognitionRecord(
        experiment="persona_self_recognition", phase="paired",
        model="m", task_id="t1", run_id="r1",
        source_persona="b", evaluator_persona="b",
        candidate_a_source="a", candidate_a_text="x",
        candidate_b_source="b", candidate_b_text="y",
        pair_order="ab", parsed_choice="B",
        is_correct=True, has_ground_truth=True,
    ))

    metrics = summarize_run(jsonl_path, tmp_path)
    assert metrics["individual"]["diagonal_mean"] == 1.0
    assert metrics["individual"]["off_diagonal_mean"] == 1.0
    assert (tmp_path / "individual_matrix.csv").exists()
    assert (tmp_path / "summary.md").exists()


def test_experiment_configs_load():
    """Each experiments/<name>/config.yaml has valid `personas` and `configs` sections."""
    import yaml

    experiments_dir = Path(__file__).parent / "experiments"
    all_named_configs: dict = {}
    for exp_dir in sorted(p for p in experiments_dir.iterdir() if p.is_dir()):
        cfg = exp_dir / "config.yaml"
        if not cfg.exists():
            continue
        with open(cfg) as f:
            data = yaml.safe_load(f) or {}
        # Each experiment defines its own persona registry
        assert "personas" in data, f"{exp_dir.name}/config.yaml missing 'personas'"
        # And one or more named configs
        assert "configs" in data, f"{exp_dir.name}/config.yaml missing 'configs'"
        for name, cfg_dict in (data["configs"] or {}).items():
            assert name not in all_named_configs, f"duplicate config name {name}"
            all_named_configs[name] = cfg_dict
            # Each named config must have a model
            assert "model_name" in cfg_dict, f"{name} missing model_name"

    assert "self_recognition_dev" in all_named_configs
    assert "patchscope_dev" in all_named_configs
    assert "confidence_entropy_dev" in all_named_configs
    assert "activation_probing_dev" in all_named_configs


def test_run_py_discovery():
    """run.py discover_configs() picks up all per-experiment configs."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from run import discover_configs, load_personas_for, load_prompts_for

    configs = discover_configs()
    assert "self_recognition_dev" in configs
    assert configs["self_recognition_dev"][0] == "persona_self_recognition"

    psr_personas = load_personas_for("persona_self_recognition")
    assert "chemist" in psr_personas

    ce_prompts = load_prompts_for("confidence_entropy")
    assert "ce_open_ended" in ce_prompts


def test_patchscope_record():
    """PatchscopeRecord should instantiate and serialize correctly."""
    from core.schemas import PatchscopeRecord
    from dataclasses import asdict

    rec = PatchscopeRecord(
        experiment="patchscope",
        model="test-model",
        question_id="q1",
        source_persona="persona_conservative",
        reporter_persona="neutral_evaluator",
        template_name="open_summary",
        interpretation_prompt_style="selfie",
        condition="real",
        source_layer=8,
        injection_layer=3,
        injection_mode="replace",
        reporter_generated_text="This represents a conservative view.",
        reporter_parsed_answer=None,
        reporter_parse_success=False,
    )
    d = asdict(rec)
    assert d["interpretation_prompt_style"] == "selfie"
    assert d["condition"] == "real"
    assert d["source_layer"] == 8
    json.dumps(d, default=str)
