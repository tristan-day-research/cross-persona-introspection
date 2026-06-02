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
    from core import base_experiment
    from experiments.confidence_entropy import experiment as ce
    from experiments.activation_probing import experiment as ap
    from experiments.patchscope import experiment as ps
    from experiments.persona_self_recognition import self_recognition_experiment as psr
    from experiments.persona_self_recognition import self_recognition_analysis_helpers as psr_analysis
    from experiments.persona_self_recognition import self_recognition_bias_analysis as psr_bias


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
    from experiments.persona_self_recognition.self_recognition_analysis_helpers import summarize_run

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
        # Mirror run.find_config: accept `config.yaml` or a single prefixed
        # `<name>_config.yaml` (e.g. self_recognition_config.yaml).
        cfg = exp_dir / "config.yaml"
        if not cfg.exists():
            matches = sorted(p for p in exp_dir.glob("*_config.yaml") if p.is_file())
            if not matches:
                continue
            cfg = matches[0]
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


# ── Persona self-recognition: response-bias modes ─────────────────────────


def test_self_recognition_parsing_helpers():
    """Free-form parsers for the probability / confidence modes."""
    from experiments.persona_self_recognition.self_recognition_experiment import (
        _parse_probability, _parse_choice_and_confidence, _generated_text_id,
    )

    assert _parse_probability("Probability: 73") == 73
    assert _parse_probability("0") == 0
    assert _parse_probability("100%") == 100
    assert _parse_probability("nope") is None      # no integer
    assert _parse_probability("250") is None       # out of range -> parse failure
    assert _parse_probability("-5") is None         # negative -> parse failure (not 5)
    assert _parse_probability("Probability: -10") is None
    assert _parse_probability(None) is None

    assert _parse_choice_and_confidence("CHOICE: A\nCONFIDENCE: 80") == ("A", 80)
    assert _parse_choice_and_confidence("choice = b confidence=5") == ("B", 5)
    assert _parse_choice_and_confidence("B, not sure") == ("B", None)   # no confidence
    assert _parse_choice_and_confidence("garbage") == (None, None)
    assert _parse_choice_and_confidence("CHOICE: A CONFIDENCE: 200") == ("A", None)  # conf OOR
    assert _parse_choice_and_confidence("CHOICE: B\nCONFIDENCE: -50") == ("B", None)  # neg conf
    assert _parse_choice_and_confidence("CONFIDENCE -50") == (None, None)             # bare neg conf

    assert _generated_text_id("t1", "chemist", "artist") == "t1|chemist|artist"


def test_self_recognition_record_validation():
    """SelfRecognitionRecord.__post_init__ enforces the mode invariants."""
    import pytest as _pytest
    from core.schemas import SelfRecognitionRecord

    base = dict(experiment="persona_self_recognition", phase="paired", model="m",
                task_id="t1", run_id="r1")

    # Valid rows pass (including an all-None generation row).
    SelfRecognitionRecord(experiment="x", phase="generation", model="m", task_id="t", run_id="r")
    SelfRecognitionRecord(**base, parsed_probability=0, parsed_confidence=100,
                          model_answer="A", correct_answer="A", position_of_self="B",
                          chose_self=True)

    # Out-of-range / bad-enum values raise.
    for bad in (
        dict(parsed_probability=150),
        dict(parsed_probability=-1),
        dict(parsed_confidence=101),
        dict(model_answer="MAYBE"),
        dict(parsed_choice="C"),
        dict(correct_answer="X"),
        dict(position_of_self="C"),
    ):
        with _pytest.raises(ValueError):
            SelfRecognitionRecord(**base, **bad)


def test_self_recognition_bias_analysis(tmp_path):
    """All-mode rows round-trip through the logger and the bias analysis funcs."""
    from core.schemas import SelfRecognitionRecord
    from core.results_logger import ResultsLogger, load_results
    from experiments.persona_self_recognition import self_recognition_bias_analysis as B

    # AUROC sanity (matches sklearn's 0.75 on this classic example).
    assert abs(B.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-9
    assert B.auroc([0, 1, 2, 3], [0, 0, 1, 1]) == 1.0

    jsonl = tmp_path / "trials.jsonl"
    log = ResultsLogger(jsonl)
    personas = ["a", "b"]
    for s in personas:
        log.log_trial(SelfRecognitionRecord(
            experiment="persona_self_recognition", phase="generation", model="m",
            task_id="t1", run_id="r1", source_persona=s, target_persona=s,
            generated_text="x", token_length=1,
        ))
    for s in personas:
        for e in personas:
            self_ = (s == e)
            # YES/NO with logprobs
            log.log_trial(SelfRecognitionRecord(
                experiment="persona_self_recognition", phase="individual", model="m",
                task_id="t1", run_id="r1", source_persona=s, target_persona=s,
                evaluator_persona=e, parsed_choice="YES" if self_ else "NO",
                model_answer="YES" if self_ else "NO", is_self=self_,
                prob_yes=0.8 if self_ else 0.2, prob_no=0.2 if self_ else 0.8,
                logprob_yes=-0.2 if self_ else -1.6, logprob_no=-1.6 if self_ else -0.2,
                is_correct=True, has_ground_truth=True,
            ))
            # numeric self-probability
            log.log_trial(SelfRecognitionRecord(
                experiment="persona_self_recognition", phase="individual_probability",
                model="m", task_id="t1", run_id="r1", source_persona=s, target_persona=s,
                evaluator_persona=e, parsed_probability=90 if self_ else 10, is_self=self_,
            ))
    # paired, counterbalanced, with confidence; evaluator 'a' authored one side
    for order, (asrc, bsrc) in (("ab", ("a", "b")), ("ba", ("b", "a"))):
        pos = "A" if asrc == "a" else "B"
        log.log_trial(SelfRecognitionRecord(
            experiment="persona_self_recognition", phase="paired", model="m",
            task_id="t1", run_id="r1", source_persona="a", evaluator_persona="a",
            candidate_a_source=asrc, candidate_b_source=bsrc, pair_order=order,
            parsed_choice=pos, model_answer=pos, prob_a=0.6, prob_b=0.4,
            parsed_confidence=70, correct_answer=pos, position_of_self=pos,
            chose_self=True, is_correct=True, has_ground_truth=True,
        ))

    df = load_results(jsonl)

    # Probability mode: perfect separation -> AUROC 1.0, zero parse failures.
    assert B.auroc_self_vs_nonself(df) == 1.0
    assert B.probability_parse_failure_rate(df)["parse_failure_rate"] == 0.0
    assert not B.mean_probability_by_evaluator(df).empty
    assert not B.calibration_table(df).empty

    # Paired summary: always chose self, perfectly counterbalanced.
    ps = B.paired_summary(df)
    assert ps["accuracy_chose_self"] == 1.0
    assert ps["p_choose_a"] == 0.5

    # YES/NO + logprob margin recorded.
    assert not B.yes_rate_by_evaluator(df).empty
    margin = B.yes_logprob_margin_by_self(df)
    assert margin.loc["self", "mean_logprob_margin"] > margin.loc["nonself", "mean_logprob_margin"]

    # Confidence present.
    assert not B.confidence_correct_vs_incorrect(df).empty

    # bias_summary names all three modes.
    summ = B.bias_summary(df)
    assert {"individual_yes_no", "individual_probability", "paired"} <= set(summ)


def test_self_recognition_bias_analysis_backward_compat():
    """Legacy result rows (missing the new columns) still analyze without error."""
    import pandas as pd
    from experiments.persona_self_recognition import self_recognition_bias_analysis as B

    legacy = pd.DataFrame([
        dict(phase="individual", model="m", task_id="t1", run_id="r1",
             source_persona="a", evaluator_persona="a", parsed_choice="YES",
             is_correct=True, has_ground_truth=True, error=None),
        dict(phase="paired", model="m", task_id="t1", run_id="r1",
             source_persona="a", evaluator_persona="a", candidate_a_source="a",
             candidate_b_source="b", parsed_choice="A", is_correct=True,
             has_ground_truth=True, error=None),
    ])
    # No new columns present, yet these must not raise.
    summ = B.bias_summary(legacy)
    assert "individual_yes_no" in summ and "paired" in summ
    assert not B.yes_rate_by_evaluator(legacy).empty
    assert B.paired_summary(legacy)["accuracy_chose_self"] == 1.0
    # Probability/confidence sections are simply empty (mode absent).
    assert B.mean_probability_by_evaluator(legacy).empty
    assert B.confidence_correct_vs_incorrect(legacy).empty


class _MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class _MockBackend:
    """Content-aware stand-in for HFBackend — no model, no GPU.

    Returns deterministic outputs keyed on the prompt so each recognition mode
    exercises its real parsing/logging path in the experiment.
    """
    def __init__(self):
        self.tokenizer = _MockTokenizer()

    def generate(self, messages, max_new_tokens=256, temperature=0.0, do_sample=False):
        content = messages[-1]["content"]
        if "Probability:" in content:
            return "73"
        if "CONFIDENCE" in content:
            return "CHOICE: A\nCONFIDENCE: 80"
        if "YES or NO" in content:
            return "YES"
        return "A short generated paragraph about the topic."

    def get_choice_probs(self, messages, choices):
        return {choices[0]: 0.7, choices[1]: 0.3}

    def get_choice_probs_and_logprobs(self, messages, choices):
        return {choices[0]: 0.7, choices[1]: 0.3}, {choices[0]: -0.36, choices[1]: -1.20}


def test_self_recognition_modes_generate_rows(tmp_path):
    """Run every recognition mode end-to-end with a mock backend (no model)."""
    from core.schemas import RunConfig, PersonaConfig, TaskItem
    from core.results_logger import load_results
    from core.results_logger import ResultsLogger
    from experiments.persona_self_recognition.self_recognition_experiment import (
        PersonaSelfRecognition,
    )

    persona_names = ["chemist", "artist", "five_year_old"]
    personas = {n: PersonaConfig(name=n, system_prompt=f"You are {n}.") for n in persona_names}
    run_config = RunConfig(
        experiment_name="persona_self_recognition",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        personas=persona_names,
        task_sets=["self_recognition_neutral"],
        output_dir=str(tmp_path),
    )
    # config_name picks up the all-modes-on bias_dev config explicitly.
    exp = PersonaSelfRecognition(run_config, personas, config_name="self_recognition_bias_dev")
    assert exp.individual_probability and exp.paired_confidence and exp.yes_no_logprobs

    # Bypass setup() (which would load a real model + task files): inject a mock
    # backend and a single in-memory task, then run all phases.
    exp.backend = _MockBackend()
    exp.tasks = [TaskItem(task_id="t1", prompt="Write about cats.", task_set="neutral")]
    exp.run_dir.mkdir(parents=True, exist_ok=True)
    exp.results_logger = ResultsLogger(exp.output_path)
    exp._write_manifest_header()
    exp.run()

    df = load_results(exp.output_path)
    phases = set(df["phase"].unique())
    assert {"generation", "individual", "individual_probability", "paired"} <= phases

    # individual_probability rows carry a parsed 0-100 value and a self/nonself label.
    prob = df[df["phase"] == "individual_probability"]
    assert (prob["parsed_probability"] == 73).all()
    assert prob["is_self"].notna().all()

    # individual YES/NO rows carry logprobs (mode on) and prob_yes/prob_no.
    ind = df[df["phase"] == "individual"]
    assert ind["logprob_yes"].notna().all() and ind["prob_yes"].notna().all()

    # paired rows carry confidence + explicit position/choice fields; both A/B
    # orders are present (counterbalanced). In confidence mode the recorded
    # answer comes from the parsed CHOICE line (the mock always says "A"), and
    # the constrained A/B probs are intentionally unset.
    paired = df[df["phase"] == "paired"]
    assert (paired["parsed_confidence"] == 80).all()
    assert (paired["model_answer"] == "A").all()      # from CHOICE: A line, not argmax
    assert paired["prob_a"].isna().all()              # constrained probs unused in confidence mode
    assert set(paired["pair_order"].unique()) == {"ab", "ba"}
    gt = paired[paired["has_ground_truth"] == True]  # noqa: E712
    assert gt["position_of_self"].isin(["A", "B"]).all()
    assert gt["chose_self"].notna().all()

    # The manifest captured example trials for the new phases.
    manifest = (exp.run_dir / "manifest.txt").read_text()
    assert "EXAMPLE TRIALS — INDIVIDUAL_PROBABILITY" in manifest
    assert "individual_probability: True" in manifest
