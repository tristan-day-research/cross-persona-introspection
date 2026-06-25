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
    from experiments.self_recognition import self_recognition_experiment as psr
    from experiments.self_recognition import self_recognition_analysis_helpers as psr_analysis
    from experiments.self_recognition import self_recognition_bias_analysis as psr_bias


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
    from experiments.self_recognition.self_recognition_analysis_helpers import summarize_run

    jsonl_path = tmp_path / "trials.jsonl"
    log = ResultsLogger(jsonl_path)

    personas = ["a", "b"]
    for s in personas:
        log.log_trial(SelfRecognitionRecord(
            experiment="self_recognition", phase="generation",
            model="m", task_id="t1", run_id="r1", source_persona=s,
            generated_text="hello", token_length=1,
        ))
    for s in personas:
        for e in personas:
            log.log_trial(SelfRecognitionRecord(
                experiment="self_recognition", phase="individual",
                model="m", task_id="t1", run_id="r1",
                source_persona=s, evaluator_persona=e,
                parsed_choice="YES" if s == e else "NO",
                choice_probs={"YES": 0.9 if s == e else 0.1, "NO": 0.1 if s == e else 0.9},
                is_correct=True, has_ground_truth=True,
            ))
    log.log_trial(SelfRecognitionRecord(
        experiment="self_recognition", phase="paired",
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
        # Each experiment defines its own persona registry, either inline under
        # `personas:` in the config or in a dedicated `*_personas.yaml` roster
        # file in the same folder (merged by run.load_personas_for).
        has_roster_file = bool(list(exp_dir.glob("*_personas.yaml"))) or \
            (exp_dir / "personas.yaml").exists()
        assert "personas" in data or has_roster_file, \
            f"{exp_dir.name}: no 'personas' in config and no *_personas.yaml roster file"
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
    assert configs["self_recognition_dev"][0] == "self_recognition"

    psr_personas = load_personas_for("self_recognition")
    # Personas now live in the dedicated self_recognition_personas.yaml roster,
    # merged in by load_personas_for; check a current roster member.
    assert "child_five" in psr_personas

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
    from experiments.self_recognition.self_recognition_experiment import (
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

    base = dict(experiment="self_recognition", phase="paired", model="m",
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
    from experiments.self_recognition import self_recognition_bias_analysis as B

    # AUROC sanity (matches sklearn's 0.75 on this classic example).
    assert abs(B.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-9
    assert B.auroc([0, 1, 2, 3], [0, 0, 1, 1]) == 1.0

    jsonl = tmp_path / "trials.jsonl"
    log = ResultsLogger(jsonl)
    personas = ["a", "b"]
    for s in personas:
        log.log_trial(SelfRecognitionRecord(
            experiment="self_recognition", phase="generation", model="m",
            task_id="t1", run_id="r1", source_persona=s, target_persona=s,
            generated_text="x", token_length=1,
        ))
    for s in personas:
        for e in personas:
            self_ = (s == e)
            # YES/NO with logprobs
            log.log_trial(SelfRecognitionRecord(
                experiment="self_recognition", phase="individual", model="m",
                task_id="t1", run_id="r1", source_persona=s, target_persona=s,
                evaluator_persona=e, parsed_choice="YES" if self_ else "NO",
                model_answer="YES" if self_ else "NO", is_self=self_,
                prob_yes=0.8 if self_ else 0.2, prob_no=0.2 if self_ else 0.8,
                logprob_yes=-0.2 if self_ else -1.6, logprob_no=-1.6 if self_ else -0.2,
                is_correct=True, has_ground_truth=True,
            ))
            # numeric self-probability
            log.log_trial(SelfRecognitionRecord(
                experiment="self_recognition", phase="individual_probability",
                model="m", task_id="t1", run_id="r1", source_persona=s, target_persona=s,
                evaluator_persona=e, parsed_probability=90 if self_ else 10, is_self=self_,
            ))
    # paired, counterbalanced, with confidence; evaluator 'a' authored one side
    for order, (asrc, bsrc) in (("ab", ("a", "b")), ("ba", ("b", "a"))):
        pos = "A" if asrc == "a" else "B"
        log.log_trial(SelfRecognitionRecord(
            experiment="self_recognition", phase="paired", model="m",
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
    from experiments.self_recognition import self_recognition_bias_analysis as B

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
        if "confidence:" in content.lower():
            return "choice: A\nconfidence: 80"
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
    from experiments.self_recognition.self_recognition_experiment import (
        SelfRecognition,
    )

    persona_names = ["chemist", "artist", "five_year_old"]
    personas = {n: PersonaConfig(name=n, system_prompt=f"You are {n}.") for n in persona_names}
    run_config = RunConfig(
        experiment_name="self_recognition",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        personas=persona_names,
        task_sets=["self_recognition_neutral"],
        output_dir=str(tmp_path),
    )
    # config_name picks up the all-modes-on bias_dev config explicitly.
    exp = SelfRecognition(run_config, personas, config_name="self_recognition_bias_dev")
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


# ── all_persona_descriptions: multi-class persona-source classification ────


def test_persona_descriptions_parse_helpers():
    """JSON parsing + per-trial candidate shuffle for the descriptions phase."""
    from experiments.self_recognition.evaluate_self_recognition import (
        _parse_description_response, _shuffle_candidates, _labels_for,
        _build_description_user, EvalOptions, _build_eval_options,
    )
    from core.schemas import PersonaConfig, RunConfig

    # Clean JSON summing to 100 → "ok", normalized to sum 1, predicted from most_likely.
    raw = ('{"probabilities": {"A": 35, "B": 10, "C": 45, "D": 10}, '
           '"most_likely": "C", "confidence": 45, "brief_reason": "style"}')
    p = _parse_description_response(raw, ["A", "B", "C", "D"])
    assert p["parse_status"] == "ok"
    assert abs(sum(p["probabilities"].values()) - 1.0) < 1e-9
    assert p["predicted_label"] == "C"
    assert abs(p["probabilities"]["C"] - 0.45) < 1e-9
    assert p["confidence"] == 45.0

    # Sums to 50 → renormalized; predicted = argmax when most_likely absent.
    p2 = _parse_description_response('{"probabilities": {"A": 10, "B": 40}}', ["A", "B"])
    assert p2["parse_status"] == "normalized"
    assert abs(p2["probabilities"]["B"] - 0.8) < 1e-9
    assert p2["predicted_label"] == "B"

    # Code fences + trailing prose still parse.
    fenced = 'Sure!\n```json\n{"probabilities":{"A":50,"B":50},"most_likely":"A"}\n``` done'
    assert _parse_description_response(fenced, ["A", "B"])["parse_status"] == "ok"

    # Unparseable / empty → failed, no probabilities.
    bad = _parse_description_response("no json", ["A", "B"])
    assert bad["parse_status"] == "failed" and bad["probabilities"] is None
    assert _parse_description_response(None, ["A"])["parse_status"] == "failed"

    assert _labels_for(4) == ["A", "B", "C", "D"]

    # Shuffle is deterministic, bijective, and covers every candidate.
    opts = EvalOptions(seed=42)
    kw = dict(dataset="xsum", key="k1", condition="active_persona", evaluator="x", source="y")
    m1 = _shuffle_candidates(["x", "y", "z"], opts, **kw)
    m2 = _shuffle_candidates(["x", "y", "z"], opts, **kw)
    assert m1 == m2 and sorted(m1.values()) == ["x", "y", "z"] and list(m1) == ["A", "B", "C"]

    # full vs short candidate rendering.
    personas = {n: PersonaConfig(name=n, system_prompt=f"You are {n}.") for n in ["x", "y", "z"]}
    full = _build_description_user("text", m1, personas, "full_prompt_anonymous_labels")
    assert "system prompt" in full and f"You are {m1['A']}." in full
    short = _build_description_user("text", m1, personas, "short_label_names")
    assert f"[A] {m1['A']}" in short

    # Option validation.
    rc = RunConfig(experiment_name="self_recognition", model_name="m", personas=["x"], task_sets=[])
    o = _build_eval_options({"measurements": ["all_persona_descriptions"],
                             "candidate_display_mode": "short_label_names"}, rc)
    assert o.measurements == ("all_persona_descriptions",) and o.runs("all_persona_descriptions")
    # Default = all four paper phases.
    assert _build_eval_options({}, rc).measurements == \
        ("pairwise_detection", "pairwise_comparison", "recognition", "scoring")
    # Individual phases are selectable; "individual" alias expands to two phases.
    assert _build_eval_options({"measurements": ["recognition"]}, rc).measurements == ("recognition",)
    assert _build_eval_options({"measurements": ["individual"]}, rc).measurements == \
        ("recognition", "scoring")
    # Mixed alias + legacy-name shorthand, de-duped into canonical order.
    assert _build_eval_options(
        {"measurements": ["all_persona_descriptions", "pairwise", "detection"]}, rc
    ).measurements == ("pairwise_detection", "pairwise_comparison", "all_persona_descriptions")
    with pytest.raises(ValueError):
        _build_eval_options({"measurements": ["bogus"]}, rc)
    with pytest.raises(ValueError):
        _build_eval_options({"candidate_display_mode": "bogus"}, rc)

    # groups: default xsum/cnn; explicit wins; inferred from task_sets in prompt mode.
    assert _build_eval_options({"measurements": ["all_persona_descriptions"]}, rc).groups \
        == ("xsum", "cnn")
    assert _build_eval_options(
        {"measurements": ["all_persona_descriptions"], "groups": ["g1", "g2"]}, rc
    ).groups == ("g1", "g2")
    assert _build_eval_options(
        {"measurements": ["all_persona_descriptions"],
         "task_sets": ["self_recognition_neutral", {"self_recognition_misaligned": ["sr_22"]}]},
        rc,
    ).groups == ("self_recognition_neutral", "self_recognition_misaligned")
    # A classic paper phase on a non-article (prompt-mode) group is rejected.
    with pytest.raises(ValueError):
        _build_eval_options({"measurements": ["recognition"],
                             "groups": ["self_recognition_neutral"]}, rc)
    # ...but descriptions on a prompt-mode group is fine.
    assert _build_eval_options({"measurements": ["all_persona_descriptions"],
                                "groups": ["self_recognition_neutral"]}, rc).groups \
        == ("self_recognition_neutral",)


def test_groups_all_autodetect(tmp_path):
    """`groups: all` scans generations_filepath and uses every group on disk."""
    from core.schemas import RunConfig
    from experiments.self_recognition.evaluate_self_recognition import (
        _build_eval_options, _detect_groups,
    )

    # Lay down two personas, each with two task-set groups (+ one stray dir w/o
    # summaries.json that must be ignored).
    for persona in ("a", "b"):
        for grp in ("set_one", "set_two"):
            f = tmp_path / persona / grp / "summaries.json"
            f.parent.mkdir(parents=True)
            f.write_text(json.dumps({"sr_01": "txt"}))
    (tmp_path / "a" / "empty_dir").mkdir()

    rc = RunConfig(experiment_name="self_recognition", model_name="m",
                   personas=["a", "b"], task_sets=[],
                   generations_filepath=str(tmp_path))
    assert _detect_groups(rc) == ("set_one", "set_two")  # sorted, summaries-only

    o = _build_eval_options({"measurements": ["all_persona_descriptions"],
                             "groups": "all"}, rc)
    assert o.groups == ("set_one", "set_two")
    # List form ["all"] is equivalent.
    o2 = _build_eval_options({"measurements": ["all_persona_descriptions"],
                              "groups": ["all"]}, rc)
    assert o2.groups == ("set_one", "set_two")

    # Empty/wrong path → fail fast.
    rc_empty = RunConfig(experiment_name="self_recognition", model_name="m",
                         personas=["a"], task_sets=[],
                         generations_filepath=str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        _detect_groups(rc_empty)


class _MockDescBackend:
    """Stand-in for HFBackend for the descriptions phase: always emits valid JSON
    favoring label A, regardless of candidate count (missing labels read as 0)."""
    _JSON = ('{"probabilities": {"A": 70, "B": 30}, "most_likely": "A", '
             '"confidence": 70, "brief_reason": "mock"}')

    def generate(self, messages, max_new_tokens=256, temperature=0.0):
        return self._JSON

    def generate_batch(self, messages_list, max_new_tokens=256, temperature=0.0):
        return [self._JSON] * len(messages_list)


def test_persona_descriptions_generate_rows(tmp_path):
    """_run_persona_descriptions writes one analysis-friendly row per evaluation,
    in both active_persona and description_only conditions, with no GPU."""
    from core.schemas import RunConfig, PersonaConfig
    from core.results_logger import ResultsLogger, load_results
    from experiments.self_recognition.evaluate_self_recognition import (
        _run_persona_descriptions, build_units, EvalOptions, NEUTRAL_EVALUATOR,
    )

    names = ["a", "b", "c"]
    personas = {n: PersonaConfig(name=n, system_prompt=f"You are {n}.") for n in names}
    sources = {n: {"k1": f"{n} text one", "k2": f"{n} text two"} for n in names}
    run_config = RunConfig(experiment_name="self_recognition", model_name="m",
                           personas=names, task_sets=[], batch_size=4)
    opts = EvalOptions(measurements=("all_persona_descriptions",), seed=42)
    backend = _MockDescBackend()

    out = tmp_path / "trials.jsonl"
    rl = ResultsLogger(out)
    # Active evaluator "a" + the neutral description-only unit.
    n_active = _run_persona_descriptions(backend, "a", personas, sources, "xsum",
                                         run_config, "r1", rl, opts)
    n_neutral = _run_persona_descriptions(backend, NEUTRAL_EVALUATOR, personas, sources,
                                          "xsum", run_config, "r1", rl, opts)
    assert n_active == 6 and n_neutral == 6  # 3 sources × 2 texts each

    df = load_results(out)
    assert set(df["phase"].unique()) == {"all_persona_descriptions"}
    assert set(df["condition"].unique()) == {"active_persona", "description_only"}

    active = df[df["condition"] == "active_persona"]
    # Mock always predicts label "A"; is_correct iff source maps to A this trial.
    for _, row in active.iterrows():
        assert row["candidate_mapping"][row["correct_label"]] == row["source_persona"]
        assert row["predicted_label"] == "A"
        assert row["is_correct"] == (row["correct_label"] == "A")
        assert row["parse_status"] == "ok"
        assert 0.0 <= row["correct_probability"] <= 1.0
    # is_self set only in the active condition; true exactly when source == evaluator "a".
    assert active[active["source_persona"] == "a"]["is_self"].all()
    assert not active[active["source_persona"] == "b"]["is_self"].any()

    neutral = df[df["condition"] == "description_only"]
    assert neutral["evaluator_persona"].isna().all()
    assert neutral["is_self"].isna().all()

    # build_units adds a neutral unit per dataset when the phase is enabled.
    units = build_units(["a", "b"], ("xsum", "cnn"), opts)
    assert (NEUTRAL_EVALUATOR, "xsum") in units and (NEUTRAL_EVALUATOR, "cnn") in units
    assert len(units) == 6  # 2 evaluators × 2 datasets + 2 neutral


def test_eval_manifest_omits_unused_templates(tmp_path):
    """manifest.txt lists only templates for measurements in this run."""
    from core.schemas import RunConfig, PersonaConfig
    from experiments.self_recognition.evaluate_self_recognition import (
        _write_manifest, EvalOptions,
    )

    personas = {"a": PersonaConfig(name="a", system_prompt="You are a.")}
    rc = RunConfig(experiment_name="self_recognition", model_name="m",
                   personas=["a"], task_sets=[], task="owntasks")
    opts = EvalOptions(measurements=("all_persona_descriptions",), groups=("g1",))
    _write_manifest(tmp_path, rc, ["a"], personas, opts, "r1", examples={})
    manifest = (tmp_path / "manifest.txt").read_text()
    assert "news-article summaries" not in manifest
    assert "ALL_PERSONA_DESCRIPTIONS" in manifest
    assert "pairwise_detection" not in manifest
    assert "EXAMPLE TRIALS" in manifest


def test_eval_manifest_examples_section(tmp_path):
    from core.schemas import RunConfig, PersonaConfig
    from experiments.self_recognition.evaluate_self_recognition import (
        _write_manifest, EvalOptions,
    )

    personas = {"a": PersonaConfig(name="a", system_prompt="You are a.")}
    rc = RunConfig(experiment_name="self_recognition", model_name="m",
                   personas=["a"], task_sets=[], task="owntasks")
    opts = EvalOptions(measurements=("all_persona_descriptions",))
    examples = {
        "all_persona_descriptions": [{
            "dataset": "g1", "key": "k1", "condition": "active_persona",
            "evaluator": "a", "source": "a", "correct_label": "A",
            "candidate_mapping": {"A": "a", "B": "b"},
            "prompt_exact": "<|begin|>system\nYou are a.<|end|>\n<|begin|>user\nText...",
            "completion_exact": '{"probabilities": {"A": 100}}',
            "response_clean": '{"probabilities": {"A": 100}}',
        }],
    }
    _write_manifest(tmp_path, rc, ["a"], personas, opts, "r1", examples=examples)
    manifest = (tmp_path / "manifest.txt").read_text()
    assert "[prompt_exact" in manifest
    assert "<|begin|>system" in manifest or "You are a." in manifest
    assert "completion_exact" in manifest


def _desc_row(source, evaluator, condition, correct_prob, *, correct, group="set1"):
    """Build one PersonaDescriptionEvalRecord for the analysis fixture.

    Fixed mapping {A:a, B:b, C:c}. `correct` decides whether the prediction hits
    the true source (the self cells are correct, others wrong) — so the fixture
    has a clean +1.0 self-advantage to assert against. `group` is the generation
    group (task set), passed as `dataset` so `task_set` mirrors it.
    """
    from core.schemas import PersonaDescriptionEvalRecord

    mapping = {"A": "a", "B": "b", "C": "c"}
    labels = list(mapping)
    correct_label = next(l for l, n in mapping.items() if n == source)
    if correct:
        predicted = correct_label
    else:  # deterministic wrong label (next one cyclically)
        predicted = labels[(labels.index(correct_label) + 1) % len(labels)]
    probs = {l: (correct_prob if l == correct_label else
                 (1 - correct_prob) / (len(labels) - 1)) for l in labels}
    return PersonaDescriptionEvalRecord(
        experiment="self_recognition_eval", phase="all_persona_descriptions",
        model="m", dataset=group, run_id="r1", key="k1", condition=condition,
        source_persona=source, candidate_display_mode="full_prompt_anonymous_labels",
        candidate_mapping=mapping, correct_label=correct_label,
        evaluator_persona=(None if condition == "description_only" else evaluator),
        probabilities=probs, predicted_label=predicted,
        correct_probability=correct_prob, is_correct=(predicted == correct_label),
        is_self=(None if condition == "description_only" else evaluator == source),
        parse_status="ok",
    )


def test_persona_descriptions_analysis(tmp_path):
    """Self-advantage / confusion / active-vs-neutral metrics on a known fixture."""
    from core.results_logger import ResultsLogger
    from experiments.self_recognition import self_recognition_descriptions_analysis as A
    from core.schemas import PersonaDescriptionEvalRecord

    names = ["a", "b", "c"]
    out = tmp_path / "trials.jsonl"
    log = ResultsLogger(out)
    # Two task-set groups, identical structure, so per-group breakdowns are testable.
    for group in ("set1", "set2"):
        # Active: self cells correct w/ high prob; nonself wrong w/ low prob.
        for s in names:
            for e in names:
                self_ = (s == e)
                log.log_trial(_desc_row(s, e, "active_persona",
                                        0.8 if self_ else 0.1, correct=self_, group=group))
            # Neutral baseline: no persona, always wrong (ordinary classification).
            log.log_trial(_desc_row(s, None, "description_only", 0.2, correct=False, group=group))
    # A failed-parse neutral row to exercise parse-failure bookkeeping.
    log.log_trial(PersonaDescriptionEvalRecord(
        experiment="self_recognition_eval", phase="all_persona_descriptions",
        model="m", dataset="set1", run_id="r1", key="k2", condition="description_only",
        source_persona="a", candidate_display_mode="full_prompt_anonymous_labels",
        candidate_mapping={"A": "a", "B": "b", "C": "c"}, correct_label="A",
        parse_status="failed"))

    from core.results_logger import load_results
    df = load_results(out)

    # Two groups: active = 18 rows (6 self correct / 12 nonself wrong) → acc 1/3;
    # neutral = 6 scored (all wrong) + 1 failed parse → 1/7 failure rate, chance 1/3.
    cond = A.condition_summary(df)
    assert abs(cond.loc["active_persona", "top1_accuracy"] - 1 / 3) < 1e-9
    assert cond.loc["description_only", "top1_accuracy"] == 0.0
    assert abs(cond.loc["description_only", "parse_failure_rate"] - 1 / 7) < 1e-9
    assert abs(cond.loc["active_persona", "chance"] - 1 / 3) < 1e-9

    # Headline self-advantage: self all correct (1.0) vs nonself all wrong (0.0).
    adv = A.self_advantage(df)
    assert adv["self_accuracy"] == 1.0 and adv["nonself_accuracy"] == 0.0
    assert adv["self_advantage_acc"] == 1.0
    assert abs(adv["self_advantage_prob"] - 0.7) < 1e-9  # 0.8 - 0.1
    assert adv["n_self"] == 6 and adv["n_nonself"] == 12  # 3 self/6 nonself × 2 groups

    # Per-source breakdown + active-vs-neutral deltas.
    by_src = A.self_advantage_by_source(df)
    assert (by_src["self_advantage_acc"] == 1.0).all()
    avn = A.active_vs_neutral(df)
    assert (avn["active_self_minus_neutral_acc"] == 1.0).all()

    # Task-set tracking: `task_set` mirrors `dataset`; per-group breakdown present.
    assert set(df["task_set"]) == {"set1", "set2"}
    by_group = A.self_advantage_by_group(df)
    assert set(by_group.index) == {"set1", "set2"}
    assert (by_group["self_advantage_acc"] == 1.0).all()
    assert set(A.accuracy_by_group(df).index) == {"set1", "set2"}

    # Confusion matrix: square, row-normalized; diagonal = per-source recall = 1/3.
    cm = A.confusion_matrix(df, condition="active_persona")
    assert list(cm.index) == names and list(cm.columns) == names
    assert abs(cm.loc["a", "a"] - 1 / 3) < 1e-9

    # Generic group table + run-level summary writes the CSVs and markdown.
    by_eval = A.accuracy_by(df, "evaluator_persona")
    assert set(by_eval.index) == set(names)
    metrics = A.summarize_run(out, tmp_path)
    assert metrics["self_advantage"]["self_advantage_acc"] == 1.0
    assert len(metrics["self_advantage_by_task_set"]) == 2
    assert (tmp_path / "descriptions_summary.md").exists()
    assert (tmp_path / "descriptions_self_advantage_by_source.csv").exists()
    assert (tmp_path / "descriptions_self_advantage_by_task_set.csv").exists()
    assert (tmp_path / "descriptions_confusion_active.csv").exists()

    # Guess / probability-bias helpers.
    gc = A.guess_counts(df, condition="active_persona")
    assert int(gc.sum()) == 18
    prob = A.mean_probability_by_persona(df, condition="active_persona")
    assert abs(prob.loc["a", "chance"] - 1 / 3) < 1e-9
    eg = A.evaluator_guess_breakdown(df)
    # Cyclic wrong-guess fixture: each evaluator names itself on 2/3 of its trials.
    assert abs(eg.loc["a", "rate_guess_self"] - 2 / 3) < 1e-9
    assert abs(eg.loc["b", "rate_guess_self"] - 2 / 3) < 1e-9
    assert not A.guess_counts_by_condition(df).empty
    dec = A.guess_self_decomposition(df, "a")
    assert dec["n_self_trials"] == 2 and dec["n_nonself_trials"] == 4
    assert dec["self_hit_rate"] == 1.0
    assert dec["false_self_rate"] == 0.5  # guesses self on source=c wrong trials only
    assert abs(dec["rate_guess_self"] - 2 / 3) < 1e-9
    rates = A.evaluator_classification_rates(df)
    assert set(rates.index) == set(names)
    assert rates.loc["a", "self_hit_rate"] == 1.0
    assert abs(rates.loc["a", "overall_accuracy"] - 1 / 3) < 1e-9


class _MockGenBackend:
    """HFBackend stand-in for generation: echoes a persona-tagged paragraph so
    clean_prompt_text's self-ref stripping is exercised. No model, no GPU."""
    def generate(self, messages, max_new_tokens=256, temperature=0.0):
        sysmsg = next((m["content"] for m in messages if m["role"] == "system"), "x")
        return f"As a {sysmsg.split()[2].rstrip('.')}, here is my answer."

    def generate_batch(self, messages_list, max_new_tokens=256, temperature=0.0):
        return [self.generate(m) for m in messages_list]


def test_prompt_mode_generation(tmp_path):
    """generate_text.py PROMPT mode: tasks/*.json → {persona}/{set}/summaries.json."""
    from core.schemas import RunConfig, PersonaConfig
    import experiments.self_recognition.generate_text as G

    rc = RunConfig(
        experiment_name="self_recognition", model_name="m",
        personas=["chemist", "historian"], task="writing_prompts",
        task_sets=[{"self_recognition_neutral": ["sr_01", "sr_06"]}],
        batch_size=2,
    )
    assert G.prompt_mode(rc) is True
    # Units = persona × task_set; group name = the set name.
    units = G.build_units(rc)
    assert len(units) == 2 and {G._task_set_name(g) for _, g in units} == {"self_recognition_neutral"}

    backend = _MockGenBackend()
    persona = PersonaConfig(name="chemist", system_prompt="You are chemist persona.")
    out_dir = G.generate_prompt_unit(backend, persona, rc.task_sets[0], rc, tmp_path)

    # Folder shape: {out_root}/{persona}/{set}/ with the standard files.
    assert out_dir == tmp_path / "chemist" / "self_recognition_neutral"
    summaries = json.loads((out_dir / "summaries.json").read_text())
    raw = json.loads((out_dir / "summaries_raw.json").read_text())
    assert set(summaries) == {"sr_01", "sr_06"}            # keyed by task_id
    assert summaries["sr_01"] == "here is my answer."      # leading "As a …," stripped
    assert raw["sr_01"].startswith("As a")                 # raw kept verbatim
    assert "prompt mode" in (out_dir / "manifest.txt").read_text()

    # task_root places it under results/text_generations/<task>/<model_slug>/.
    tr = G.task_root("writing_prompts", "meta-llama/Llama-3.1-8B-Instruct")
    assert tr.parent.name == "writing_prompts" and tr.parent.parent.name == "text_generations"


def test_generations_root_and_preflight(tmp_path):
    """generations_filepath resolution + the pre-flight 'missing generations' check."""
    from core.schemas import RunConfig
    import experiments.self_recognition.generate_text as G
    import experiments.self_recognition.evaluate_self_recognition as E

    # task_root: results/text_generations/<task>/<model_slug>/ (flat = derive fallback).
    tr = G.task_root("articles", "meta-llama/Llama-3.1-8B-Instruct")
    assert tr.parent.name == "articles" and tr.name == "Llama-3-1-8B-Instruct"
    assert tr.parent.parent.name == "text_generations"
    # With a run_id, each run nests in its own timestamped folder under that.
    stamped = G.task_root("articles", "meta-llama/Llama-3.1-8B-Instruct", "20260616_120000")
    assert stamped.name == "20260616_120000" and stamped.parent == tr

    # Explicit generations_filepath wins; otherwise derive from task + model.
    rc_explicit = RunConfig(experiment_name="self_recognition", model_name="m",
                            personas=["a"], task_sets=[], generations_filepath=str(tmp_path))
    assert E._generations_root(rc_explicit) == tmp_path
    rc_derived = RunConfig(experiment_name="self_recognition",
                           model_name="meta-llama/Llama-3.1-8B-Instruct",
                           personas=["a"], task_sets=[], task="articles")
    assert E._generations_root(rc_derived) == G.task_root("articles", rc_derived.model_name)

    # Legacy `generations_dir:` config key still resolves via run.py's alias.
    from run import build_run_config
    rc_legacy = build_run_config("self_recognition", {
        "model_name": "m", "personas": ["a"], "task_sets": [],
        "mode": "evaluate_self_recognition", "generations_dir": str(tmp_path)})
    assert E._generations_root(rc_legacy) == tmp_path

    # Pre-flight: missing → FileNotFoundError naming the gaps; present → passes.
    rc = RunConfig(experiment_name="self_recognition", model_name="m",
                   personas=["a", "b"], task_sets=[], generations_filepath=str(tmp_path))
    with pytest.raises(FileNotFoundError) as ei:
        E.check_generations_present(rc)
    assert "a/xsum" in str(ei.value) and "b/cnn" in str(ei.value)
    for p in ("a", "b"):
        for ds in ("xsum", "cnn"):
            f = tmp_path / p / ds / "summaries.json"
            f.parent.mkdir(parents=True)
            f.write_text(json.dumps({"k": "t"}))
    E.check_generations_present(rc)  # no raise once all present


# ── Binary 12-case self-recognition eval (case1…case12) ─────────────────────


def test_binary_trial_enumeration_counts_and_balance():
    """Each case enumerates the spec's exact trial count, perfectly A/B-balanced,
    with unique ids; expand_conditions fans over the case's eval-SP states; case5
    excludes the persona of interest from both texts; case12 mirrors a base case."""
    from math import comb
    from experiments.self_recognition.evaluation_cases import (
        CASE_REGISTRY, CASE_TYPES, _enumerate_case, build_case_trials,
        expand_conditions, resolve_spec,
    )
    P = [f"p{i}" for i in range(5)]
    K = [f"t{i}" for i in range(4)]
    p, n = len(P), len(K)
    base_n = 4 * n * p * (p - 1)                 # every self-vs-other case
    expected = {c: base_n for c in CASE_REGISTRY}
    expected["case5"] = 4 * n * p * comb(p - 1, 2)
    for case_id, exp_n in expected.items():
        trials = _enumerate_case(CASE_REGISTRY[case_id], "xsum", sorted(K), sorted(P), seed=1)
        assert len(trials) == exp_n, f"{case_id}: {len(trials)} != {exp_n}"
        # exact 50/50 correct-answer balance by construction
        nA = sum(t.correct_answer == "A" for t in trials)
        assert nA * 2 == len(trials), f"{case_id} A/B imbalance {nA}/{len(trials)}"
        # unique base_trial_ids; case_type tagged
        assert len({t.base_trial_id for t in trials}) == len(trials)
        assert all(t.case_type == CASE_TYPES[case_id] for t in trials)
    # case5: the persona of interest authored neither text and is not a candidate
    c5 = _enumerate_case(CASE_REGISTRY["case5"], "xsum", sorted(K), sorted(P), seed=1)
    assert all(t.evaluator_persona not in (t.text1_source_persona, t.text2_source_persona)
               for t in c5)
    assert all(t.evaluator_persona not in (t.other_persona_1, t.other_persona_2) for t in c5)

    # Sampling caps the case and flags it; base_trial_ids reused across eval-SP states.
    spec5 = CASE_REGISTRY["case5"]               # eval_sp = (True, False)
    sample = build_case_trials(spec5, "xsum", K, P, sampling_seed=3, max_trials=20)
    assert len(sample) == 20 and all(t.sampled_from_full_case for t in sample)
    expanded = expand_conditions(sample, spec5)
    assert len(expanded) == 20 * 2               # two eval-SP states
    assert {t.base_trial_id for t in expanded} == {t.base_trial_id for t in sample}
    assert len({t.trial_id for t in expanded}) == len(expanded)
    assert {t.eval_system_prompt_enabled for t in expanded} == {True, False}
    # case7 is active-only and describes nobody → one eval-SP state, style n/a.
    spec7 = CASE_REGISTRY["case7"]
    b7 = build_case_trials(spec7, "xsum", K[:2], P[:3], sampling_seed=3, max_trials=10 ** 9)
    e7 = expand_conditions(b7, spec7)
    assert len(e7) == len(b7)                    # eval_sp = (True,)
    assert {t.eval_system_prompt_enabled for t in e7} == {True}
    assert {t.other_description_style for t in e7} == {"not_applicable"}
    # case12 mirrors its base case's structure but is retagged calibration.
    spec12 = resolve_spec("case12", calibration_base_case="case7")
    c12 = _enumerate_case(spec12, "xsum", sorted(K), sorted(P), seed=1)
    assert len(c12) == base_n and all(t.case_id == "case12" for t in c12)
    assert all(t.case_type == "calibration" for t in c12)
    # deterministic sampling
    a = build_case_trials(spec5, "xsum", K, P, sampling_seed=9, max_trials=15)
    b = build_case_trials(spec5, "xsum", K, P, sampling_seed=9, max_trials=15)
    assert [t.base_trial_id for t in a] == [t.base_trial_id for t in b]


def test_binary_correct_answer_logic():
    """correct_answer tracks the actual source/text-order under each counterbalance."""
    from experiments.self_recognition.evaluation_cases import CASE_REGISTRY, _enumerate_case
    # case3: the correct letter must point at the persona-of-interest's text.
    for t in _enumerate_case(CASE_REGISTRY["case3"], "xsum", ["t0"], ["e", "o"], seed=1):
        e_slot = "text1" if t.text1_source_persona == t.evaluator_persona else "text2"
        assert t.answer_mapping[t.correct_answer] == e_slot
    # case1: correct option is "current" iff the single text is the evaluator's own
    # (the letter itself is counterbalanced, so check the semantic key).
    for t in _enumerate_case(CASE_REGISTRY["case1"], "xsum", ["t0"], ["e", "o"], seed=1):
        truth = "current" if t.source_persona == t.evaluator_persona else "other"
        assert t.answer_mapping[t.correct_answer] == truth


class _MockLogprobBackend:
    """Stand-in whose constrained read always favors A (prob 0.7) — lets the
    binary runner be exercised with no GPU by monkeypatching choice_logprobs_batch."""


def test_binary_runner_records_and_resume(tmp_path, monkeypatch):
    """_run_binary_trials writes one PersonaBinaryEvalRecord per trial with valid
    prob_A/prob_B/logprob fields, correct is_correct, and resumes (skips done ids)."""
    import experiments.self_recognition.evaluate_self_recognition as E
    from core.schemas import PersonaConfig, RunConfig
    from core.results_logger import ResultsLogger, load_results
    from experiments.self_recognition.evaluation_cases import (
        CASE_REGISTRY, build_case_trials, expand_conditions,
    )
    from experiments.self_recognition.evaluate_self_recognition_helpers import EvalOptions

    names = ["artist", "chemist", "historian"]
    personas = {n: PersonaConfig(name=n, system_prompt=f"You are a {n}.") for n in names}
    text_map = {"xsum": {n: {"t0": f"{n} wrote this"} for n in names}}

    # Mock the single forward-pass seam: always (probs favoring A, plausible logprobs).
    def fake_logprobs(backend, messages_list, choices, batch_size=1, **kw):
        return [({"A": 0.7, "B": 0.3}, {"A": -0.5, "B": -1.5}) for _ in messages_list]
    monkeypatch.setattr(E, "choice_logprobs_batch", fake_logprobs)

    spec = CASE_REGISTRY["case2"]
    base = build_case_trials(spec, "xsum", ["t0"], names, sampling_seed=1, max_trials=10 ** 9)
    trials = expand_conditions(base, spec)
    rc = RunConfig(experiment_name="self_recognition", model_name="m", personas=names,
                   task_sets=[], batch_size=4)
    opts = EvalOptions(cases_to_run=("case2",), save_logprobs=True)

    out = tmp_path / "trials.jsonl"
    rl = ResultsLogger(out)
    n = E._run_binary_trials(_MockLogprobBackend(), trials, personas, text_map, rc,
                             "run1", "gen1", rl, opts)
    assert n == len(trials)
    df = load_results(out)
    assert (df["phase"] == "binary_recognition").all()
    assert (abs(df["prob_A"] + df["prob_B"] - 1.0) < 1e-9).all()
    assert (df["predicted_answer"] == "A").all()           # mock favors A
    assert (df["logprob_A"].notna()).all() and (df["logprob_A"] <= 0).all()
    # is_correct must equal (predicted == correct_answer)
    assert (df["is_correct"] == (df["predicted_answer"] == df["correct_answer"])).all()
    # system prompt recorded only when enabled
    assert df[df["eval_system_prompt_enabled"]]["system_prompt_text"].notna().all()
    assert df[~df["eval_system_prompt_enabled"]]["system_prompt_text"].isna().all()

    # Resume: re-running with all trial_ids marked done writes nothing new.
    done = set(df["trial_id"])
    n2 = E._run_binary_trials(_MockLogprobBackend(), trials, personas, text_map, rc,
                              "run1", "gen1", rl, opts, done_ids=done)
    assert n2 == 0
    assert len(load_results(out)) == len(trials)


def test_build_eval_options_binary_path():
    """cases_to_run switches EvalOptions to the binary path; bad inputs are caught."""
    from core.schemas import RunConfig
    from experiments.self_recognition.evaluate_self_recognition import _build_eval_options

    rc = RunConfig(experiment_name="self_recognition", model_name="m", personas=["a"],
                   task_sets=[], seed=7)
    opts = _build_eval_options({
        "cases_to_run": ["case1", "case9", "case12"],
        "description_style": "redacted",
        "calibration_base_case": "case3",
        "calibration_personas": ["five_year_old"],
        "max_trials_per_case": 100, "groups": ["xsum"],
    }, rc)
    assert opts.runs_binary() and opts.cases_to_run == ("case1", "case9", "case12")
    assert opts.description_style == "redacted"
    assert opts.calibration_base_case == "case3"
    assert opts.calibration_personas == ("five_year_old",)
    assert opts.max_trials_per_case == 100
    assert opts.sampling_seed == 7  # defaults to run seed
    assert not opts.needs_articles()  # binary never needs source articles

    # The legacy key max_trials_per_case_condition is still accepted as an alias.
    legacy = _build_eval_options(
        {"cases_to_run": ["case1"], "max_trials_per_case_condition": 55}, rc)
    assert legacy.max_trials_per_case == 55

    with pytest.raises(ValueError):
        _build_eval_options({"cases_to_run": ["case99"]}, rc)                       # unknown case
    with pytest.raises(ValueError):
        _build_eval_options({"cases_to_run": ["case1"], "description_style": "bogus"}, rc)
    with pytest.raises(ValueError):  # case12 cannot mirror itself
        _build_eval_options({"cases_to_run": ["case12"], "calibration_base_case": "case12"}, rc)


def test_generation_conceal_and_strip_helpers():
    """conceal_persona appends the instruction to the user turn; strip_self_refs
    gates the leading "As a <role>," strip in BOTH article and prompt modes."""
    import experiments.self_recognition.generate_text as G
    from core.schemas import RunConfig

    on = RunConfig(experiment_name="self_recognition", model_name="m", personas=["chemist"],
                   task_sets=[], conceal_persona=True, strip_self_refs=True)
    off = RunConfig(experiment_name="self_recognition", model_name="m", personas=["chemist"],
                    task_sets=[], conceal_persona=False, strip_self_refs=False)
    assert G.CONCEAL_INSTRUCTION in G._compose_user("Summarize.", on)
    assert G._compose_user("Summarize.", off) == "Summarize."

    xs = "As a chemist, the reaction released gas."
    assert G.clean_summary(xs, "xsum", strip_self_refs=True) == "the reaction released gas."
    assert G.clean_summary(xs, "xsum", strip_self_refs=False) == xs
    assert G.clean_prompt_text("As a historian, X.", strip_self_refs=True) == "X."
    assert G.clean_prompt_text("As a historian, X.", strip_self_refs=False) == "As a historian, X."


class _EchoUserBackend:
    """Generation stand-in that records the user turns it sees and returns a
    persona-tagged line, so concealment-in-prompt and self-ref stripping are testable."""
    def __init__(self):
        self.seen_users = []

    def generate(self, messages, max_new_tokens=256, temperature=0.0):
        self.seen_users.append(next(m["content"] for m in messages if m["role"] == "user"))
        return "As a chemist, here is the summary."

    def generate_batch(self, messages_list, max_new_tokens=256, temperature=0.0):
        return [self.generate(m) for m in messages_list]


def test_article_mode_generation_conceal(tmp_path):
    """generate_text.py ARTICLE mode with conceal_persona: the concealment text
    reaches the user turn and the stored summary has the giveaway stripped."""
    from core.schemas import RunConfig, PersonaConfig
    import experiments.self_recognition.generate_text as G

    rc = RunConfig(experiment_name="self_recognition", model_name="m", personas=["chemist"],
                   task_sets=[], sample_size=2, batch_size=2,
                   conceal_persona=True, strip_self_refs=True)
    backend = _EchoUserBackend()
    persona = PersonaConfig(name="chemist", system_prompt="You are a chemist.")
    out_dir = G.generate_unit(backend, persona, "xsum", rc, tmp_path, start_index=0)

    # Concealment instruction made it into the user turn the model saw.
    assert backend.seen_users and all(G.CONCEAL_INSTRUCTION in u for u in backend.seen_users)
    summaries = json.loads((out_dir / "summaries.json").read_text())
    raw = json.loads((out_dir / "summaries_raw.json").read_text())
    assert len(summaries) == 2
    # Stored text: leading "As a chemist," stripped; raw kept verbatim.
    assert all(v == "here is the summary." for v in summaries.values())
    assert all(v.startswith("As a chemist,") for v in raw.values())
    manifest = (out_dir / "manifest.txt").read_text()
    assert "conceal_persona:  True" in manifest and G.CONCEAL_INSTRUCTION in manifest


def test_binary_collection_layout_and_slice_tag():
    """eval_dir switches the layout to a stable collection dir with one result
    file per case-set slice; without it, the legacy per-run layout."""
    from pathlib import Path
    from experiments.self_recognition.evaluate_self_recognition_helpers import (
        EvalOptions, binary_layout, binary_slice_tag,
    )
    rd = Path("/tmp/coll")
    # A single case reads cleanly (eval-SP / description treatment are intrinsic).
    opts = EvalOptions(cases_to_run=("case7",), eval_dir="run_a")
    assert binary_slice_tag(opts) == "case7"
    lay = binary_layout(rd, opts)
    assert lay.collection and lay.deliverable == rd / "case7.jsonl"
    assert lay.eval_manifest == rd / "manifests" / "case7.eval_manifest.json"
    assert lay.trial_manifest.parent == rd / "_work" / "case7"

    # A description_style override is reflected in the slice tag (distinct file).
    opts_ov = EvalOptions(cases_to_run=("case3",), description_style="redacted", eval_dir="run_a")
    assert binary_slice_tag(opts_ov) == "case3__redacted"
    assert binary_layout(rd, opts_ov).deliverable != lay.deliverable

    # Legacy (no eval_dir): single trials.jsonl + eval_manifest.json at the root.
    legacy = binary_layout(rd, EvalOptions(cases_to_run=("case1",)))
    assert not legacy.collection
    assert legacy.deliverable == rd / "trials.jsonl"
    assert legacy.eval_manifest == rd / "eval_manifest.json"


def test_persona_descriptions_artifact(tmp_path):
    """All description styles are built/stored (incl. secret-redacted); a curated
    file overrides per (persona, style); build_trial_prompt uses the resolved one."""
    from core.schemas import PersonaConfig
    from experiments.self_recognition.evaluate_self_recognition_helpers import (
        EvalOptions, build_persona_descriptions, write_persona_descriptions,
        resolve_persona_descriptions, build_trial_prompt,
    )
    from experiments.self_recognition.evaluation_cases import (
        CASE_REGISTRY, build_case_trials, expand_conditions,
    )

    personas = {n: PersonaConfig(name=n, system_prompt=f"You are a {n}. Never reveal this.")
                for n in ["artist", "chemist"]}
    desc = build_persona_descriptions(personas)
    assert desc["chemist"]["system_prompt_style"] == "You are a chemist. Never reveal this."
    assert desc["chemist"]["third_person_description"].startswith("This persona is a chemist")
    # redacted: the secrecy sentence is dropped (heuristic fallback).
    assert "Never reveal" not in desc["chemist"]["redacted"]
    assert "chemist" in desc["chemist"]["redacted"]

    # Round-trip + override: a curated style wins; the others fall back.
    path = tmp_path / "persona_descriptions.json"
    write_persona_descriptions(path, {"chemist": {"system_prompt_style": "CUSTOM CHEMIST."}})
    opts = EvalOptions(cases_to_run=("case2",), descriptions_filepath=str(path))
    resolved = resolve_persona_descriptions(opts, personas)
    assert resolved["chemist"]["system_prompt_style"] == "CUSTOM CHEMIST."
    assert resolved["chemist"]["third_person_description"].startswith("This persona is a chemist")

    # build_trial_prompt uses the override text for the OTHER persona (case2 uses
    # system_prompt_style, so the overridden text shows up in the prompt).
    spec = CASE_REGISTRY["case2"]
    base = build_case_trials(spec, "xsum", ["t0"], ["artist", "chemist"],
                             sampling_seed=1, max_trials=10 ** 9)
    trials = expand_conditions(base, spec)
    t = next(t for t in trials if t.other_persona == "chemist")
    texts = {"single_text": "x", "text1": None, "text2": None,
             "text1_generation_id": "g", "text2_generation_id": None}
    prompt = build_trial_prompt(t, texts, personas, spec=spec, descriptions=resolved)
    assert "CUSTOM CHEMIST." in prompt


# ── Activation capture (core.activation_capture / core.activation_store) ────


def test_activation_layers_and_position_extraction():
    """resolve_layers depth-scaling, char→token span mapping, and the
    mean/last10/final/token position reductions (+ None for empty spans)."""
    import torch
    from core.activation_capture import (
        DEFAULT_LAYERS, resolve_layers, char_span_to_token_span, extract_positions,
    )
    # All defaults fit a deep model → unchanged; a shallow model → scaled & clamped.
    assert resolve_layers(40) == tuple(sorted(set(DEFAULT_LAYERS)))
    scaled = resolve_layers(22)
    assert max(scaled) <= 21 and scaled == tuple(sorted(set(scaled)))
    assert resolve_layers(24)  # Dolphin-ish: no crash, in range
    assert all(0 <= l < 24 for l in resolve_layers(24))

    # one "token" per character → char spans map 1:1 to token spans
    offsets = [(i, i + 1) for i in range(10)]
    assert char_span_to_token_span(offsets, 2, 5) == (2, 5)
    assert char_span_to_token_span(offsets, 100, 200) is None

    L, seq, H = 3, 8, 4
    states = torch.arange(L * seq * H, dtype=torch.float16).reshape(L, seq, H)
    feats = extract_positions(states, {
        "m": ("mean", 2, 5),
        "l10": ("last10_mean", 0, seq),   # ≤10 tokens → mean over all
        "f": ("final", 2, 5),
        "tok": ("token", -1),
        "empty": ("mean", 4, 4),
        "skip": None,
    })
    assert feats["m"].shape == (L, H)
    assert torch.allclose(feats["m"], states[:, 2:5, :].mean(1))
    assert torch.allclose(feats["l10"], states.mean(1))
    assert torch.allclose(feats["f"], states[:, 4, :])
    assert torch.allclose(feats["tok"], states[:, -1, :])
    assert feats["empty"] is None and feats["skip"] is None


def test_activation_store_roundtrip_and_resume(tmp_path):
    """Sharded safetensors + parquet store: None-skip, sharding, metadata, resume."""
    import torch
    from safetensors.torch import load_file
    import pandas as pd
    from core.activation_store import ActivationStore

    L, H = 3, 5

    def feats(i):
        return {"text1_mean": torch.randn(L, H), "text2_mean": None}

    s = ActivationStore(tmp_path, "evaluation", layers=list(range(L)), hidden_dim=H,
                        model="m", shard_size=2)
    for i in range(3):
        s.add(f"t{i}", feats(i), {"case": "case3", "correct": bool(i % 2)})
    assert s.has("t0") and not s.has("t9")
    s.close()

    shards = sorted((tmp_path / "evaluation").glob("acts_*.safetensors"))
    assert len(shards) == 2  # 3 ids, shard_size 2 → [2, 1]
    keys = load_file(shards[0])
    assert all("text2_mean" not in k for k in keys)  # None captures skipped
    meta = pd.read_parquet(tmp_path / "evaluation" / "metadata.parquet")
    assert len(meta) == 3 and {"id", "case", "correct"} <= set(meta.columns)

    # Resume: a fresh store over the same dir sees prior ids as done.
    s2 = ActivationStore(tmp_path, "evaluation", layers=list(range(L)), hidden_dim=H,
                         model="m", shard_size=2)
    assert all(s2.has(f"t{i}") for i in range(3))


def test_eval_and_generation_capture_spans():
    """The named-position span builders place text/description/system spans and the
    final-token / generated-text positions correctly (char-per-token offsets)."""
    from experiments.self_recognition.binary_activations import eval_capture_spans
    from experiments.self_recognition.generation_activations import generation_capture_spans

    # EVAL: full prompt = "SYS" + user; user contains TEXT1 and a description.
    user = "...TEXT1ABC...DESCXYZ...(((((("   # primer "(" tokens at the tail
    full = "SYS" + user + " My answer is ("
    offsets = [(i, i + 1) for i in range(len(full))]
    seg = {"text1": (user.index("TEXT1ABC"), user.index("TEXT1ABC") + 8),
           "other_description": (user.index("DESCXYZ"), user.index("DESCXYZ") + 7)}
    spans = eval_capture_spans(full, user, seg, offsets=offsets, seq_len=len(full),
                               system_prompt_text="SYS")
    t1s = 3 + user.index("TEXT1ABC")
    assert spans["text1_mean"] == ("mean", t1s, t1s + 8)
    assert spans["pre_text_token"] == ("token", t1s - 1)
    assert spans["active_system_prompt_final"][0] == "final"
    assert spans["final_prompt_token_before_answer"] == ("token", len(full) - 1)
    assert "other_description_mean" in spans

    # GENERATION: rendered prompt then generated text appended.
    rendered = "SYSPROMPT>>>"
    full_g = rendered + "GENERATED_TEXT"
    offs = [(i, i + 1) for i in range(len(full_g))]
    gspans = generation_capture_spans(full_g, rendered, offsets=offs, seq_len=len(full_g),
                                      system_prompt_text="SYSPROMPT")
    assert gspans["generation_prompt_final"] == ("token", len(rendered) - 1)
    assert gspans["generated_text_mean"] == ("mean", len(rendered), len(full_g))
    assert gspans["persona_prompt_final"][0] == "final"
