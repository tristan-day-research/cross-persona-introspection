"""Smoke tests: imports, config loading, parsing, and a mock dry run."""

import json
import tempfile
from pathlib import Path


def test_imports():
    """All modules should import without error."""
    from cross_persona_introspection import schemas
    from cross_persona_introspection.backends import hf_backend, openrouter_backend
    from cross_persona_introspection.core import persona_inducer, task_loader, response_parser, kv_cache, results_logger
    from cross_persona_introspection.experiments import base, cross_persona_prediction, source_reporter_matrix, patchscope
    from cross_persona_introspection.evaluation import choice_matching, calibration, llm_judge


def test_schemas():
    """Schema dataclasses should instantiate correctly."""
    from cross_persona_introspection.schemas import (
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
    from cross_persona_introspection.core.response_parser import parse_constrained_response

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
    from cross_persona_introspection.core.persona_inducer import induce_persona
    from cross_persona_introspection.schemas import PersonaConfig

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
    from cross_persona_introspection.core.task_loader import load_task_file, sample_tasks
    from cross_persona_introspection.schemas import TaskItem

    # Create a temp task file
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
    from cross_persona_introspection.core.results_logger import ResultsLogger, load_results
    from cross_persona_introspection.schemas import TrialRecord

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


def test_choice_matching():
    """Test scoring functions."""
    from cross_persona_introspection.evaluation.choice_matching import score_trial, score_reporter_trial
    from cross_persona_introspection.schemas import TrialRecord

    record = TrialRecord(
        experiment="test", model="m", task_id="t1", task_set="factual",
        predicted_answer="B", predicted_confidence=3, predicted_refuse=False,
        actual_answer="B", actual_confidence=4, actual_refuse=False,
    )
    scored = score_trial(record)
    assert scored.choice_match is True
    assert scored.confidence_error == 1
    assert scored.refuse_match is True

    # Reporter trial
    reporter_record = TrialRecord(
        experiment="test", model="m", task_id="t1", task_set="factual",
        source_metrics={"top1_answer": "C", "confidence_proxy_bin": 4},
        reporter_answer="C", reporter_confidence=3,
    )
    scored = score_reporter_trial(reporter_record)
    assert scored.choice_match is True
    assert scored.confidence_error == 1


def test_config_loading():
    """Test that config files parse correctly."""
    import yaml

    config_dir = Path(__file__).parent.parent / "config"

    with open(config_dir / "personas.yaml") as f:
        personas = yaml.safe_load(f)
    assert "default_assistant" in personas
    assert "sycophantic" in personas

    with open(config_dir / "experiments.yaml") as f:
        experiments = yaml.safe_load(f)
    assert "cross_persona_prediction_dev" in experiments
    assert "patchscope_dev" in experiments

    with open(config_dir / "patchscope.yaml") as f:
        ps_config = yaml.safe_load(f)
    assert "extraction" in ps_config
    assert "injection" in ps_config
    assert "source_pass" in ps_config
    assert "interpretation_templates" in ps_config


def test_patchscope_record():
    """PatchscopeRecord should instantiate and serialize correctly."""
    from cross_persona_introspection.schemas import PatchscopeRecord
    from dataclasses import asdict

    rec = PatchscopeRecord(
        experiment="patchscope",
        model="test-model",
        question_id="q1",
        source_persona="persona_conservative",
        reporter_persona="neutral_evaluator",
        template_name="open_summary",
        condition="real",
        source_layer=8,
        injection_layer=3,
        injection_mode="replace",
        generated_text="This represents a conservative view.",
        parsed_answer=None,
        parse_success=False,
    )
    d = asdict(rec)
    assert d["condition"] == "real"
    assert d["source_layer"] == 8
    # Should be JSON-serializable
    json.dumps(d, default=str)


def test_patchscope_helpers():
    """Test patchscope helper functions that don't require a model."""
    from cross_persona_introspection.experiments.patchscope.patchscope_helpers import (
        _load_patchscope_config,
        _model_short_name,
        _resolve_layers,
        build_interpretation_prompt,
        describe_source_extraction_site,
        format_source_pass_user_message,
        reporter_sample_opposing_qualifies,
        resolve_extraction_token_index,
    )
    from transformers import GPT2Tokenizer

    # Layer resolution
    assert _resolve_layers("all", 32) == list(range(32))
    assert _resolve_layers("middle", 32) == list(range(10, 20))
    assert _resolve_layers(8, 32) == [8]
    assert _resolve_layers([4, 8, 12], 32) == [4, 8, 12]

    # Model short name
    assert _model_short_name("meta-llama/Llama-3.1-8B-Instruct") == "l8b"
    assert "tinyllama" in _model_short_name("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Config loading
    cfg = _load_patchscope_config("patchscope.yaml")
    inj = cfg.get("injection") or {}
    if "layer" in inj:
        assert inj["layer"] in (3, 8)  # optional when using layer_pairs only
    assert "source_pass" in cfg
    assert "user_message_template" in cfg["source_pass"]
    assert "reporting" in cfg
    assert "include_no_reporter_system_sample" in cfg["reporting"]
    assert "include_no_persona_chat_template_sample_per_layer" in cfg["reporting"]
    assert "no_persona_layer_log_system_prompt" in cfg["reporting"]
    assert "no_persona_layer_log_body" in cfg["reporting"]
    assert "opposing_sample_policy" in cfg["reporting"]
    q = {
        "question_text": "Pick one?",
        "options": {"A": "First", "C": "Third"},
    }
    out = format_source_pass_user_message(q, cfg["source_pass"]["user_message_template"])
    assert "Pick one?" in out
    assert "A) First" in out and "C) Third" in out
    assert "B)" not in out
    assert "open_summary" in cfg["interpretation_templates"]
    # Templates should have both patchscopes and selfie styles
    assert "patchscopes" in cfg["interpretation_templates"]["open_summary"]
    assert "selfie" in cfg["interpretation_templates"]["open_summary"]

    # Extraction site description (no model; local GPT-2 tokenizer)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tmpl = {
        "selfie": "HEAD {persona_prompt} MID {placeholder} TAIL",
        "decode_mode": "generate",
    }
    interp, _msgs, _ph = build_interpretation_prompt(
        tok,
        tmpl,
        "selfie",
        "",
        "?",
        1,
        {"question_text": "Q", "options": {}},
        "I am the persona.",
        use_chat_template=False,
    )
    assert interp.count("I am the persona.") == 1
    assert "HEAD I am the persona. MID" in interp

    text = "Hello world"
    site = describe_source_extraction_site(tok, text, "last")
    assert "token_index" in site and "n_tokens" in site
    assert site["token_index"] == site["n_tokens"] - 1
    assert "token_id" in site

    boundary_text = "Hello user.<|eot_id|><|start_header_id|>assistant"
    pos, meta = resolve_extraction_token_index(tok, boundary_text, "last_before_assistant")
    assert pos < resolve_extraction_token_index(tok, boundary_text, "last")[0]
    assert "boundary_char_index" in meta

    assert reporter_sample_opposing_qualifies(
        "persona_conservative", "persona_progressive", "cross_ideology"
    )
    assert not reporter_sample_opposing_qualifies(
        "persona_conservative", "neutral_evaluator", "cross_ideology"
    )
    assert reporter_sample_opposing_qualifies(
        "persona_conservative", "neutral_evaluator", "any_mismatch"
    )
