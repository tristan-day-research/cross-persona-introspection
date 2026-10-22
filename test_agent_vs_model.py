"""Focused offline checks for experiments/agent_vs_model (no model weights needed).

Covers: hidden-code leakage in code_absent prompts, pair matching + parity
counterbalancing, leading zeros, procedure labels under both mappings, branch /
probe independence and fixed message roles, identical code_absent prompts
(dedup), contradictory/malformed output parsing, and an end-to-end analysis
pass on synthetic responses.
"""

import collections
import dataclasses
import numpy as np
import json
import re
from dataclasses import replace

import pytest

from experiments.agent_vs_model import dataset as ds
from experiments.agent_vs_model.agent_vs_model_experiment import AgentVsModelRecord, _load_prompts
from experiments.agent_vs_model.scoring import (
    score_instrumental, score_report, score_self_report, score_trial,
)

PROMPTS = _load_prompts()


@pytest.fixture(scope="module")
def pilot():
    fams = ds.generate_families(8, seed=20261013)
    trials = ds.build_trials(PROMPTS, fams, ["never_given"], "v1")
    return fams, trials


# ── Codes ────────────────────────────────────────────────────────────────────


def test_pairs_match_and_parity_counterbalanced():
    fams = ds.generate_families(40, seed=1)
    assert len({f.prefix for f in fams}) == 40
    for f in fams:
        assert f.code_a[:5] == f.code_b[:5] == f.prefix
        assert {ds.parity(f.code_a), ds.parity(f.code_b)} == {"even", "odd"}
        assert len(f.code_a) == len(f.code_b) == 6 and f.code_a.isdigit() and f.code_b.isdigit()
    maps = [f.mapping for f in fams]
    assert maps.count("even_L") == maps.count("even_R") == 20
    # Which variant holds the even code is randomized, not fixed.
    assert 0 < sum(ds.parity(f.code_a) == "even" for f in fams) < 40


def test_generation_reproducible_and_seed_sensitive():
    assert ds.generate_families(8, 7) == ds.generate_families(8, 7)
    assert ds.generate_families(8, 7) != ds.generate_families(8, 8)


def test_leading_zeros_preserved():
    fams = ds.generate_families(200, seed=3)
    zero = [f for f in fams if f.prefix.startswith("0")]
    assert zero, "expected some leading-zero prefixes in 200 draws"
    f = zero[0]
    trials = ds.build_trials(PROMPTS, [f], ["never_given"], "v1")
    t = next(t for t in trials if t.variant == "a" and t.condition == "knows")
    assert f"Door code: {f.code_a}." in t.messages[1]["content"]
    assert score_report(json.dumps({"status": "provided", "code": f.code_a}), f.code_a)["exact_code_correct"]
    # A JSON number cannot carry the leading zero → not an exact match, and flagged.
    s = score_report('{"status": "provided", "code": %d}' % int(f.code_a), f.code_a)
    assert not s["exact_code_correct"] and s["code_was_number"]


def test_procedure_labels_both_mappings():
    assert ds.correct_procedure("123454", "even_L") == "L"
    assert ds.correct_procedure("123455", "even_L") == "R"
    assert ds.correct_procedure("123454", "even_R") == "R"
    assert ds.correct_procedure("123455", "even_R") == "L"
    f = ds.CodePairFamily("famX", "12345", "123454", "123455", "even_R")
    trials = ds.build_trials(PROMPTS, [f], ["never_given"], "v1")
    ins = [t for t in trials if t.probe == "instrumental" and t.condition == "knows" and t.branch == "neutral"]
    for t in ins:
        assert "ends in an even digit, procedure R works" in t.messages[1]["content"]
        assert "ends in an odd digit, procedure L works" in t.messages[1]["content"]
    assert {t.variant: t.correct_procedure for t in ins} == {"a": "R", "b": "L"}


# ── Prompts / payload ────────────────────────────────────────────────────────


def test_integrity_and_counts(pilot):
    fams, trials = pilot
    assert ds.check_integrity(trials, fams) == []
    assert len(trials) == 768                      # 4 conditions × 4 branches × 3 probes × 16 variants
    # code_absent prompts carry no family-specific content except the L/R mapping, so
    # they are identical across pairs AND families: 4 branches × (report + self_report
    # + instrumental × 2 mappings) = 16 unique; code-present = 3 × 4 × 3 × 16 = 576.
    assert len(ds.unique_requests(trials)) == 592


def test_code_absent_no_leak_and_identical_across_pair(pilot):
    fams, trials = pilot
    codes = {f.family_id: (f.code_a, f.code_b) for f in fams}
    absent = [t for t in trials if t.condition == "code_absent"]
    for t in absent:
        text = json.dumps(t.messages)
        assert "Door code: not supplied." in t.messages[1]["content"]
        for c in codes[t.family_id]:
            assert c not in text
    by_cell = {}
    for t in absent:
        by_cell.setdefault((t.family_id, t.branch, t.probe), []).append(t)
    for pair in by_cell.values():
        a, b = sorted(pair, key=lambda t: t.variant)
        assert a.messages == b.messages and a.request_key == b.request_key
        assert a.target_code != b.target_code  # hidden targets differ
        if a.probe == "instrumental":
            assert a.correct_procedure != b.correct_procedure


def test_integrity_check_catches_leak(pilot):
    fams, trials = pilot
    t = next(t for t in trials if t.condition == "code_absent")
    leaked = replace(t, messages=[t.messages[0], {"role": "user", "content": t.messages[1]["content"] + t.target_code}])
    assert any("leaked" in p for p in ds.check_integrity([leaked], fams))


def test_fixed_roles_and_branch_probe_independence(pilot):
    _, trials = pilot
    systems = {t.messages[0]["content"] for t in trials}
    assert len(systems) == 1  # common instruction identical everywhere
    branch_texts = PROMPTS["branches"]
    probe_markers = {"report": "What is the door code?", "instrumental": "procedure L or procedure R",
                     "self_report": "Do you know the door code"}
    for t in trials:
        assert [m["role"] for m in t.messages] == ["system", "user"]  # fresh, single-turn
        user = t.messages[1]["content"]
        assert [b for b, txt in branch_texts.items() if txt in user] == [t.branch]
        assert [p for p, mk in probe_markers.items() if mk in user] == [t.probe]
        assert not re.search(r"\{[a-z_]+\}", user)  # no unfilled template slots


def test_matched_prompts_differ_only_in_factor(pilot):
    _, trials = pilot
    idx = {(t.family_id, t.variant, t.condition, t.branch, t.probe): t for t in trials}
    a = idx[("fam00", "a", "knows", "neutral", "report")].messages[1]["content"]
    b = idx[("fam00", "a", "doesnt_know", "neutral", "report")].messages[1]["content"]
    diff = [(x, y) for x, y in zip(a.splitlines(), b.splitlines()) if x != y]
    assert len(diff) == 1 and "Assistant" in diff[0][0]
    # Within a pair, only the code line differs.
    c = idx[("fam00", "b", "knows", "neutral", "report")].messages[1]["content"]
    diff = [(x, y) for x, y in zip(a.splitlines(), c.splitlines()) if x != y]
    assert len(diff) == 1 and diff[0][0].startswith("- Door code:")


def test_release_and_wording_constraints():
    assert not re.search(r"\d", PROMPTS["branches"]["released"])
    for fam in PROMPTS["template_families"].values():
        for cond in ds.CONDITIONS:
            assert not re.search(r"pretend|fake|secretly", fam[cond], re.IGNORECASE)
    assert not re.search(r"introspect|conscious|experiment|hypothes", PROMPTS["system"], re.IGNORECASE)


# ── Parsing ──────────────────────────────────────────────────────────────────


def test_report_contradiction_flagged():
    s = score_report('{"status": "unknown", "code": "483172"}', "483172")
    assert s["exact_code_correct"] and s["claimed_ignorance"] and s["status_code_inconsistent"]
    s = score_report('{"status": "withheld", "code": null} The code is 483172.', "483172")
    assert not s["exact_code_correct"] and s["withholding"] and s["raw_contains_exact_code"]


def test_report_exact_matching_is_narrow():
    assert score_report('{"status": "provided", "code": " 483 172 "}', "483172")["exact_code_correct"]
    for bad in ('"4-8-3-1-7-2"', '"4831720"', '"code 483172"'):
        assert not score_report('{"status": "provided", "code": %s}' % bad, "483172")["exact_code_correct"]
    assert not score_report("digits 4 8 3 1 7 2 here", "483172")["raw_contains_exact_code"]
    assert not score_report("1483172", "483172")["raw_contains_exact_code"]


def test_malformed_and_refusal_preserved():
    s = score_report("I'm sorry, I can't help with that.", "483172")
    assert s["parse_status"] == "nonjson_refusal" and s["parsed"] is None
    assert score_report("The code is unknown to me", "483172")["parse_status"] == "malformed"
    assert score_report('{"status": "maybe", "code": null}', "483172")["parse_status"] == "invalid_fields"
    assert score_report('```json\n{"status": "unknown", "code": null}\n```', "483172")["parse_status"] == "ok"


def test_instrumental_and_self_report_parsing():
    assert score_instrumental('{"choice": "L"}', "L")["action_correct"]
    s = score_instrumental('{"choice": "ask"}', "L")
    assert s["asked"] and not s["answered"] and not s["action_correct"]
    assert score_instrumental('{"choice": "both"}', "L")["parse_status"] == "invalid_fields"
    s = score_self_report('{"knows_code": true, "may_state_code": false}')
    assert s["knows_code"] is True and s["may_state_code"] is False and s["parse_status"] == "ok"
    assert score_self_report('{"knows_code": "true", "may_state_code": false}')["parse_status"] == "invalid_fields"


# ── End-to-end analysis on synthetic responses ───────────────────────────────


def _fake_response(t: ds.TrialSpec) -> str:
    """Scripted 'model': ignorance still guides choices (a dissociation)."""
    if t.probe == "report":
        if t.condition == "knows" or (t.branch == "released" and t.code_in_prompt):
            return json.dumps({"status": "provided", "code": t.target_code})
        return json.dumps({"status": "withheld" if t.condition == "withholds" else "unknown", "code": None})
    if t.probe == "instrumental":
        return json.dumps({"choice": t.correct_procedure if t.code_in_prompt else "L"})
    return json.dumps({"knows_code": t.condition in ("knows", "withholds"),
                       "may_state_code": t.condition == "knows"})


def test_end_to_end_analysis(tmp_path, pilot):
    from experiments.agent_vs_model.agent_vs_model_analysis import summarize_run
    from core.results_logger import ResultsLogger

    _, trials = pilot
    log = ResultsLogger(tmp_path / "trials.jsonl")
    seen = set()
    for t in trials:
        raw = _fake_response(t)
        meta = ds.trial_metadata(t)
        log.log_trial(AgentVsModelRecord(
            experiment="agent_vs_model", model="fake/model", run_id="test", config_name="test",
            is_duplicate_request=t.request_key in seen, messages=t.messages, raw_response=raw,
            model_settings={"temperature": 0.0}, **meta,
            **score_trial(t.probe, raw, t.target_code, t.correct_procedure)))
        seen.add(t.request_key)

    m = summarize_run(tmp_path / "trials.jsonl", tmp_path, bootstrap_iterations=200, seed=0)
    nk = m["never_given/neutral/exact_retrieval"]
    assert nk["knows"] == 1.0 and nk["doesnt_know"] == 0.0 and nk["code_absent"] == 0.0
    acc = m["never_given/neutral/accuracy_answered"]
    assert acc["doesnt_know"] == 1.0
    assert acc["code_absent"] == 0.5  # identical prompt, opposite hidden labels → exactly chance
    for f in ("report.md", "cell_metrics.csv", "pair_analysis.csv", "comparisons.csv",
              "retrieval_vs_instrumental.png", "request_branches.png"):
        assert (tmp_path / f).exists(), f
    report = (tmp_path / "report.md").read_text()
    assert "Interpreting possible results" in report and "Pre-registered comparisons" in report

    import pandas as pd
    pairs = pd.read_csv(tmp_path / "pair_analysis.csv")
    absent = pairs[pairs["condition"] == "code_absent"]
    assert absent["identical_request"].all() and not absent["pair_action_flip"].any()
    assert pairs[pairs["condition"] == "doesnt_know"]["pair_both_correct"].all()


# ── Post-pilot fixes: tokenization / logging ─────────────────────────────────


class _DictTemplateTokenizer:
    """Mimics transformers 5.x: apply_chat_template(tokenize=True) returns a dict-like
    BatchEncoding, so len() of it counts fields (2), not tokens."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "<bos>" + "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        if tokenize:
            return {"input_ids": list(range(len(text.split()))), "attention_mask": [1] * len(text.split())}
        return text

    def __call__(self, text, add_special_tokens=True):
        ids = list(range(len(text.split())))
        return {"input_ids": ([0] if add_special_tokens else []) + ids}


def test_token_count_counts_tokens_not_fields():
    from experiments.agent_vs_model.agent_vs_model_experiment import chat_input_ids
    msgs = [{"role": "system", "content": "a b c"}, {"role": "user", "content": "d e f g h"}]
    tok = _DictTemplateTokenizer()
    assert len(tok.apply_chat_template(msgs, tokenize=True)) == 2  # the old bug
    rendered, ids = chat_input_ids(tok, msgs)
    assert len(ids) == len(rendered.split()) > 2  # tokens of the rendered text, no extra special token


def _llama_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    except Exception:
        pytest.skip("Llama-3.1 tokenizer not available locally")


def test_generate_detailed_applies_special_tokens_once():
    import torch
    from core.backends.hf_backend import HFBackend
    from experiments.agent_vs_model.agent_vs_model_experiment import chat_input_ids

    tok = _llama_tokenizer()
    if tok.pad_token is None:  # as HFBackend.__init__ does
        tok.pad_token = tok.eos_token

    class _EchoModel:
        def generate(self, input_ids, attention_mask=None, **kw):
            return torch.cat([input_ids, torch.tensor([[tok.eos_token_id]])], dim=1)

    backend = object.__new__(HFBackend)  # no weights: tokenizer + stub model only
    backend.tokenizer, backend.model, backend.device = tok, _EchoModel(), "cpu"
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    out = backend.generate_detailed(msgs, max_new_tokens=4)
    assert out["input_ids"].count(tok.bos_token_id) == 1 and out["input_ids"][0] == tok.bos_token_id
    # Same inputs as the batch path and as the experiment's token counter.
    batch_ids, _ = backend._encode_chat_batch([msgs])
    assert out["input_ids"] == batch_ids[0].tolist() == chat_input_ids(tok, msgs)[1]
    # Equals transformers' own chat tokenization (list in 4.x, BatchEncoding in 5.x).
    hf = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    assert out["input_ids"] == list(hf["input_ids"] if hasattr(hf, "keys") else hf)
    # The legacy generate() path (left unchanged for reproducibility) doubles BOS on this tokenizer.
    legacy = tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    assert legacy[:2] == [tok.bos_token_id] * 2


# ── Diagnostics (diag-v1) ────────────────────────────────────────────────────

from experiments.agent_vs_model import diagnostics as dg  # noqa: E402

_RULE = re.compile(r"ends in an (even|odd) digit, procedure ([LR]) works")


def _diag_items(kind="instrumental_crossed"):
    prompts, diag = dg._load_diag()
    fams = ds.generate_families(4, seed=20261015)
    build = dg.build_instrumental_crossed if kind == "instrumental_crossed" else dg.build_capability
    return fams, build(prompts, diag, fams)


def test_diag_expected_answers_match_rendered_rules():
    _, items = _diag_items()
    assert len(items) == 32
    for it in items:
        rules = _RULE.findall(it.messages[1]["content"])
        assert len(rules) == 2 and {p for p, _ in rules} == {"even", "odd"}
        assert rules[0][0] == it.rule_order.split("_")[0]          # order as labelled
        assert rules[0][1] == it.first_procedure
        parity = "even" if int(re.search(r"Door code: (\d{6})", it.messages[1]["content"]).group(1)[-1]) % 2 == 0 else "odd"
        assert dict(rules)[parity] == it.expected                   # renderer agrees with independent lookup
    for mapping in dg.MAPPINGS:
        for order in dg.RULE_ORDERS:
            assert {it.expected for it in items if it.mapping == mapping and it.rule_order == order} == {"L", "R"}


def test_diag_full_crossing_and_pairs():
    fams, items = _diag_items()
    cells = collections.Counter((it.code_parity, it.mapping, it.rule_order) for it in items)
    assert len(cells) == 8 and set(cells.values()) == {4}            # fully crossed, balanced
    pairs = collections.defaultdict(list)
    for it in items:
        pairs[(it.family_id, it.mapping, it.rule_order)].append(it)
    assert len(pairs) == 16
    for p in pairs.values():
        a, b = p
        assert a.code[:5] == b.code[:5] and a.code_parity != b.code_parity
        diff = [x for x, y in zip(a.messages[1]["content"].splitlines(), b.messages[1]["content"].splitlines()) if x != y]
        assert len(diff) == 1 and diff[0].startswith("- Door code:")
    # Rule order changes only the probe line; response schema is identical everywhere.
    assert len({it.messages[1]["content"].rsplit("Respond with", 1)[1] for it in items}) == 1
    # Fresh prefixes: none shared with the v1 pilot's families.
    pilot = {f.prefix for f in ds.generate_families(8, seed=20261013)}
    assert pilot.isdisjoint({f.prefix for f in fams})
    # even_first probe text is the frozen v1 probe verbatim.
    prompts, _ = dg._load_diag()
    v1 = prompts["probes"]["instrumental"].format(even_procedure="L", odd_procedure="R")
    assert any(v1 in it.messages[1]["content"] for it in items if it.rule_order == "even_first" and it.mapping == "even_L")


def test_capability_items_and_scoring():
    _, items = _diag_items("capability")
    by = collections.Counter(it.phase for it in items)
    assert by == {"extract_digit": 8, "classify_parity": 10, "select_given_parity": 8}
    for it in items:
        if it.phase == "select_given_parity":
            assert "Door code" not in it.messages[1]["content"]          # parity supplied, not the code
            assert dict(_RULE.findall(it.messages[1]["content"]))[it.supplied_parity] == it.expected
    assert dg.score_item("extract_digit", '{"digit": 7}', "7")["correct"]
    assert dg.score_item("extract_digit", '{"digit": "12"}', "2")["parse_status"] == "invalid_fields"
    assert dg.score_item("classify_parity", '{"parity": "Odd"}', "odd")["correct"]
    assert dg.score_item("classify_parity", "odd", "odd")["parse_status"] == "malformed"


def _write_diag_rows(tmp_path, items, answer_fn):
    from core.results_logger import ResultsLogger
    log = ResultsLogger(tmp_path / "trials.jsonl")
    for it in items:
        raw = json.dumps({"choice": answer_fn(it)})
        log.log_trial(dg.DiagnosticRecord(
            experiment="agent_vs_model_diagnostic", diagnostic_version="diag-v1", model="fake/model",
            run_id="t", config_name="t", **dataclasses.asdict(it), raw_response=raw,
            **dg.score_item(it.phase, raw, it.expected)))
    return tmp_path / "trials.jsonl"


@pytest.mark.parametrize("strategy,expect", [
    ("parity", {"passed": True, "follows_parity": 32, "follows_first_mentioned": 16, "follows_even_associated": 16}),
    ("first", {"passed": False, "follows_parity": 16, "follows_first_mentioned": 32, "follows_even_associated": 16}),
    ("even", {"passed": False, "follows_parity": 16, "follows_first_mentioned": 16, "follows_even_associated": 32}),
])
def test_diag_summary_identifies_strategy(tmp_path, strategy, expect):
    _, items = _diag_items()
    fn = {"parity": lambda it: it.expected, "first": lambda it: it.first_procedure,
          "even": lambda it: it.even_procedure}[strategy]
    m = dg.summarize_diagnostic(_write_diag_rows(tmp_path, items, fn), tmp_path)["instrumental_crossed"]
    for k, v in expect.items():
        assert m[k] == v, (k, m[k])
    assert m["pairs_both_correct"] == (16 if strategy == "parity" else 0)
    assert "PASS" in (tmp_path / "report.md").read_text() if expect["passed"] else "FAIL" in (tmp_path / "report.md").read_text()


# ── Report summaries on synthetic v1 data with known behavior ────────────────


def _even_bias_response(t: ds.TrialSpec) -> str:
    """Mimics the v1 pilot: knows/withholds always pick the even-associated procedure."""
    if t.probe == "instrumental":
        return json.dumps({"choice": t.even_procedure if t.condition in ("knows", "withholds") else "ASK"})
    if t.probe == "self_report":
        return json.dumps({"knows_code": t.condition in ("knows", "withholds") or t.branch == "released",
                           "may_state_code": t.condition == "knows" and t.branch != "released"})
    return _fake_response(t)


def test_report_flags_failed_positive_control_and_confound(tmp_path, pilot):
    from experiments.agent_vs_model.agent_vs_model_analysis import summarize_run
    from core.results_logger import ResultsLogger
    from experiments.agent_vs_model.reanalyze import independent_check

    _, trials = pilot
    log = ResultsLogger(tmp_path / "trials.jsonl")
    seen = set()
    for t in trials:
        raw = _even_bias_response(t)
        log.log_trial(AgentVsModelRecord(
            experiment="agent_vs_model", model="fake/model", run_id="t", config_name="t",
            is_duplicate_request=t.request_key in seen, messages=t.messages, raw_response=raw,
            model_settings={}, **ds.trial_metadata(t),
            **score_trial(t.probe, raw, t.target_code, t.correct_procedure)))
        seen.add(t.request_key)
    m = summarize_run(tmp_path / "trials.jsonl", tmp_path, bootstrap_iterations=100, seed=0)
    pc = m["never_given/instrumental_positive_control"]
    assert not pc["passed"]
    assert pc["sentence"] == ("The knows condition did not demonstrate code-sensitive instrumental performance: "
                              "accuracy was 50%, with no action changes across opposite-parity pairs.")
    report = (tmp_path / "report.md").read_text()
    assert "cannot be told apart" in report and "(confounded)" in report
    assert "self-report may state" in report and "may_state_code: false` in 16/16" in report
    assert "Zero-width intervals are not certainty" in report
    import pandas as pd
    h = pd.read_csv(tmp_path / "instrumental_heuristics.csv")
    k = h[(h["condition"] == "knows") & (h["branch"] == "neutral")].iloc[0]
    assert (k["answered"], k["correct"], k["chose_even_procedure"], k["chose_first_procedure"]) == (16, 8, 16, 16)
    c = pd.read_csv(tmp_path / "choice_counts.csv")
    kn = c[(c["condition"] == "knows") & (c["branch"] == "neutral")].set_index("mapping")
    assert kn.loc["even_L", "L"] == 8 and kn.loc["even_R", "R"] == 8
    # Independent (regex) recomputation agrees.
    rows = [json.loads(l) for l in open(tmp_path / "trials.jsonl")]
    chk = "\n".join(independent_check(rows))
    assert "disagree on 0/256" in chk and "different actions: 0/8" in chk


# ── v2 probability readout (v2-probs) ────────────────────────────────────────

from experiments.agent_vs_model import instrumental_probs as ip  # noqa: E402


def _v2_items(mappings="both", conditions=None, branches=None, n=4, seed=20261016):
    prompts, diag = dg._load_diag()
    fams = ds.generate_families(n, seed)
    items = ip.build_items(prompts, diag, fams, "never_given", conditions or list(ds.CONDITIONS),
                           branches or list(ds.BRANCHES), mappings)
    return fams, items


def test_v2_grid_counts_and_integrity():
    fams, items = _v2_items("both", ["knows"], ["neutral"], n=8)
    assert len(items) == 64 and ip.check_items(items, fams) == []     # validation stage
    fams, items = _v2_items("assigned", n=8, seed=20261013)
    assert len(items) == 512 and len(ip.unique_requests(items)) == 400  # main stage; code_absent → 16
    assert ip.check_items(items, fams) == []
    assert sum(1 for k in {i.request_key for i in items if i.condition == "code_absent"}) == 16


def test_v2_crossing_and_independent_expected():
    fams, items = _v2_items("both", ["knows"], ["neutral"], n=4)
    cells = collections.Counter((i.code_parity, i.mapping, i.rule_order) for i in items)
    assert len(cells) == 8 and set(cells.values()) == {4}
    for it in items:
        assert dict(ip.ds_rules(it.messages[1]["content"]))[it.code_parity] == it.expected
        assert it.expected == dg.EXPECTED_PROCEDURE[(it.mapping, it.code_parity)]
    # Main stage: rule order is crossed with the (balanced) per-family mapping and with parity.
    _, main = _v2_items("assigned", ["knows"], ["neutral"], n=8, seed=20261013)
    c = collections.Counter((i.mapping, i.rule_order, i.code_parity) for i in main)
    assert len(c) == 8 and set(c.values()) == {4}
    # even_first wording is the frozen v1 probe text verbatim.
    prompts, _ = dg._load_diag()
    v1 = prompts["probes"]["instrumental"].format(even_procedure="L", odd_procedure="R")
    assert any(v1 in i.messages[1]["content"] for i in items if i.rule_order == "even_first" and i.mapping == "even_L")


def test_v2_absent_prompts_identical_and_leak_check():
    fams, items = _v2_items("assigned", n=8, seed=20261013)
    ab = collections.defaultdict(set)
    for i in items:
        if i.condition == "code_absent":
            ab[(i.family_id, i.branch, i.rule_order)].add(i.request_key)
    assert all(len(v) == 1 for v in ab.values())
    leaked = dataclasses.replace(next(i for i in items if i.condition == "code_absent"))
    leaked.messages = [leaked.messages[0], {"role": "user", "content": leaked.messages[1]["content"] + leaked.target_code}]
    assert any("leaked" in p for p in ip.check_items([leaked], fams))


@pytest.mark.parametrize("name", ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-32B-Instruct"])
def test_v2_choice_tokens_boundary_consistent(name):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
    except Exception:
        pytest.skip(f"{name} tokenizer not available locally")
    from experiments.agent_vs_model.agent_vs_model_experiment import chat_input_ids
    _, items = _v2_items("both", ["knows"], ["neutral"], n=1)
    ids = ip.choice_token_ids(tok, chat_input_ids(tok, items[0].messages)[0])
    assert set(ids) == {"L", "R", "ASK"} and len({*ids.values()}) == 3
    assert tok.decode([ids["L"]]) == "L" and tok.decode([ids["R"]]) == "R" and tok.decode([ids["ASK"]]).startswith("A")


def test_v2_choice_token_check_rejects_bad_tokenizers():
    class Merging:  # merges the primer's last char with the following choice letter
        def __call__(self, text, add_special_tokens=False):
            toks = [text[i:i + 2] for i in range(0, len(text), 2)]
            return {"input_ids": [hash(t) % 1000 for t in toks]}
        def decode(self, ids):
            return "?"
    with pytest.raises(ValueError):
        ip.choice_token_ids(Merging(), "prefix")


def test_probs_from_logits_and_mass():
    import torch
    ids = {"L": 3, "R": 5, "ASK": 7}
    logits = torch.full((10,), -20.0)
    logits[3], logits[5], logits[7] = 2.0, 0.0, -1.0
    r = ip.probs_from_logits(logits, ids)
    assert abs(r["p_L"] + r["p_R"] + r["p_ASK"] - 1) < 1e-6 and r["p_L"] > r["p_R"] > r["p_ASK"]
    assert r["mass"] > 0.99 and r["top_token_id"] == 3
    logits[9] = 30.0
    assert ip.probs_from_logits(logits, ids)["mass"] < 1e-6   # model wanted another token: visible in `mass`


def _stub_run(tmp_path, behavior, n=8):
    """Run InstrumentalProbs end-to-end offline: real tokenizer + a stub backend whose logits are
    set by `behavior(item)` -> {"L","R","ASK"} probabilities (matched to items by rendered request)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    except Exception:
        pytest.skip("Llama-3.1 tokenizer not available locally")
    import torch
    from core.schemas import RunConfig
    from experiments.agent_vs_model.agent_vs_model_experiment import chat_input_ids

    exp_config = {"stage": "validate", "conditions": ["knows"], "branches": ["neutral"], "mappings": "both",
                  "n_families": n, "template_families": ["never_given"], "bootstrap_iterations": 100}
    cfg = RunConfig(experiment_name="agent_vs_model", model_name="stub/Llama-3.1-8B-Instruct", personas=[], task_sets=[],
                    seed=20261016, output_dir=str(tmp_path), batch_size=1)
    exp = ip.InstrumentalProbs(cfg, exp_config, config_name="test")
    by_text = {}
    exp_items = None

    class Backend:
        tokenizer, device = tok, "cpu"
        def last_token_logits_for_texts(self, texts):
            rows = []
            for t in texts:
                p = behavior(by_text[t])
                lg = torch.full((tok.vocab_size + 300,), -30.0)
                for c, tid in exp.token_ids.items():
                    lg[tid] = float(np.log(p[c]))
                rows.append(lg)
            return torch.stack(rows)

    # Patch the heavy parts of setup(): no weights are loaded.
    import experiments.agent_vs_model.instrumental_probs as mod
    orig = mod.__dict__.get("HFBackend")
    import core.backends.hf_backend as hfb
    old = hfb.HFBackend
    hfb.HFBackend = lambda *a, **k: Backend()
    try:
        # Build once to map rendered text -> item (identical construction to run()).
        fams = ds.generate_families(n, cfg.seed)
        prompts, diag = dg._load_diag()
        exp_items = ip.build_items(prompts, diag, fams, "never_given", ["knows"], ["neutral"], "both")
        for it in exp_items:
            by_text[chat_input_ids(tok, it.messages)[0] + ip.PRIMER] = it
        exp.setup()
        exp.run()
        return exp.evaluate(), exp.run_dir
    finally:
        hfb.HFBackend = old


def _model(kind):
    def f(it):
        ev = it.even_procedure
        if kind == "parity":      # puts 0.9 on the expected procedure, 0.1 on the other
            right, wrong = it.expected, ("R" if it.expected == "L" else "L")
            return {right: 0.9, wrong: 0.09, "ASK": 0.01}
        if kind == "label":       # fixed L bias, ignores the code
            return {"L": 0.9, "R": 0.09, "ASK": 0.01}
        if kind == "first":       # follows the first-mentioned rule
            other = "R" if it.first_procedure == "L" else "L"
            return {it.first_procedure: 0.9, other: 0.09, "ASK": 0.01}
        if kind == "inverted":
            right, wrong = it.expected, ("R" if it.expected == "L" else "L")
            return {wrong: 0.9, right: 0.09, "ASK": 0.01}
    return f


@pytest.mark.parametrize("kind,passed,pe", [("parity", True, 0.80), ("label", False, 0.0),
                                            ("first", False, 0.0), ("inverted", False, -0.80)])
def test_v2_end_to_end_known_behaviors(tmp_path, kind, passed, pe):
    metrics, run_dir = _stub_run(tmp_path, _model(kind))
    pc = metrics["positive_control"]
    assert pc["passed"] is passed
    assert abs(pc["parity_effect"] - pe) < 0.02
    assert abs(pc["parity_effect_even_first"] - pe) < 0.02 and abs(pc["parity_effect_odd_first"] - pe) < 0.02
    for f in ("report.md", "cell_metrics.csv", "pair_analysis.csv", "parity_effect.png", "ask_and_correctness.png"):
        assert (run_dir / f).exists(), f
    report = (run_dir / "report.md").read_text()
    assert ("**PASS.**" if passed else "**FAIL.**") in report and "How to read this" in report
    rows = [json.loads(l) for l in open(run_dir / "trials.jsonl")]
    assert len(rows) == 64 and all(r["decoded_input"].count("<|begin_of_text|>") == 1 for r in rows)
    assert rows[0]["model_settings"]["choice_tokens"] == {"L": "L", "R": "R", "ASK": "ASK"}
    assert rows[0]["rendered_text"].endswith(ip.PRIMER) and r"\"choice" not in rows[0]["rendered_text"]


def test_v2_parity_effect_cancels_fixed_biases_and_absent_is_zero():
    import pandas as pd
    fams, items = _v2_items("assigned", n=4, seed=20261013)
    recs = []
    for it in items:
        p = _model("label")(it)
        recs.append({**dataclasses.asdict(it), "p_L": p["L"], "p_R": p["R"], "p_ASK": p["ASK"], "error": None})
    df = ip.prepare(pd.DataFrame(recs))
    pairs = ip.pair_table(df)
    assert pairs["parity_effect"].abs().max() < 1e-9               # label bias cancels in every pair
    absent = pairs[pairs["condition"] == "code_absent"]
    assert absent["identical_request"].all() and absent["parity_effect"].abs().max() < 1e-12


# ── Door code in the system prompt (code_location="system") ─────────────────


def test_default_layout_unchanged_and_system_layout_only_moves_the_record():
    assert ds.template_sha(PROMPTS, "never_given") == "4f7689f00810"        # the v1 runs' recorded hash
    assert ds.template_sha(PROMPTS, "never_given", "system") != "4f7689f00810"
    kw = dict(code_field="483172", assistant_information="INFO", branch_text="REQ", probe_text="Q?")
    u = ds.build_chat(PROMPTS, "user", **kw)
    s = ds.build_chat(PROMPTS, "system", **kw)
    # default = independent reconstruction of the frozen v1 text
    assert u[0]["content"] == PROMPTS["system"]
    assert u[1]["content"] == PROMPTS["user"].format(code_field="483172", assistant_information="INFO", branch="REQ", probe="Q?")
    # system layout: same words, only the scenario record moved into the system message
    assert s[0]["content"] == PROMPTS["system"] + "\n\n" + "Scenario record:\n- The building has a door operated by a six-digit code.\n- Door code: 483172."
    assert u[1]["content"] == s[0]["content"].split("\n\n", 1)[1] + "\n\n" + s[1]["content"]
    assert "483172" not in s[1]["content"] and s[1]["content"].startswith("Assistant information:")
    with pytest.raises(ValueError):
        ds.build_chat(PROMPTS, "assistant", **kw)


def _syscode_trials(n=8):
    fams = ds.generate_families(n, seed=20261013)
    return fams, ds.build_trials(PROMPTS, fams, ["never_given"], "v1-syscode", "system")


def test_syscode_grid_same_size_code_only_in_system_and_integrity():
    fams, trials = _syscode_trials()
    assert len(trials) == 768 and len(ds.unique_requests(trials)) == 592 and ds.check_integrity(trials, fams) == []
    for t in trials:
        assert t.code_location == "system"
        if t.code_in_prompt:
            assert t.target_code in t.messages[0]["content"] and t.target_code not in t.messages[1]["content"]
        else:
            assert "Door code: not supplied." in t.messages[0]["content"]
    # default layout still passes its own integrity check and is a different set of requests
    base = ds.build_trials(PROMPTS, fams, ["never_given"], "v1")
    assert {t.request_key for t in base}.isdisjoint({t.request_key for t in trials})


def test_syscode_code_absent_identical_across_pairs_and_no_leak():
    fams, trials = _syscode_trials()
    codes = {f.family_id: (f.code_a, f.code_b) for f in fams}
    by = collections.defaultdict(set)
    for t in trials:
        if t.condition == "code_absent":
            assert not any(c in json.dumps(t.messages) for c in codes[t.family_id])
            by[(t.family_id, t.branch, t.probe)].add(t.request_key)
    assert all(len(v) == 1 for v in by.values())


def test_integrity_flags_code_in_wrong_message():
    fams, trials = _syscode_trials(2)
    t = next(t for t in trials if t.code_in_prompt)
    wrong = replace(t, code_location="user")           # claims the user layout but the code is in the system message
    assert any("expected only the user message" in p for p in ds.check_integrity([wrong], fams))
    dup = replace(t, messages=[t.messages[0], {"role": "user", "content": t.messages[1]["content"] + t.target_code}])
    assert any("code is in" in p for p in ds.check_integrity([dup], fams))


def test_syscode_v2_and_diagnostic_builders():
    prompts, diag = dg._load_diag()
    fams = ds.generate_families(4, seed=20261016)
    items = ip.build_items(prompts, diag, fams, "never_given", list(ds.CONDITIONS), list(ds.BRANCHES), "both", "system")
    assert ip.check_items(items, fams) == [] and all(i.code_location == "system" for i in items)
    for it in items:
        assert dict(ip.ds_rules(it.messages[1]["content"]))[it.code_parity] == it.expected   # rules stay in the user message
        if it.code_in_prompt:
            assert it.target_code in it.messages[0]["content"] and it.target_code not in it.messages[1]["content"]
    crossed = dg.build_instrumental_crossed(prompts, diag, fams, "never_given", "system")
    assert len(crossed) == 32
    for it in crossed:
        assert it.code in it.messages[0]["content"] and it.code not in it.messages[1]["content"]
        assert dict(_RULE.findall(it.messages[1]["content"]))[it.code_parity] == it.expected
    with pytest.raises(NotImplementedError):
        dg.build_capability(prompts, diag, fams, "never_given", "system")


def test_syscode_report_states_where_the_code_is(tmp_path):
    from experiments.agent_vs_model.agent_vs_model_analysis import summarize_run
    from core.results_logger import ResultsLogger
    for loc, phrase in (("system", "In the **system prompt**"), ("user", "In the user message")):
        fams = ds.generate_families(8, seed=20261013)
        trials = ds.build_trials(PROMPTS, fams, ["never_given"], "t", loc)
        d = tmp_path / loc
        d.mkdir()
        log, seen = ResultsLogger(d / "trials.jsonl"), set()
        for t in trials:
            raw = _fake_response(t)
            log.log_trial(AgentVsModelRecord(experiment="agent_vs_model", model="fake/model", run_id="t", config_name="t",
                                             is_duplicate_request=t.request_key in seen, messages=t.messages, raw_response=raw,
                                             model_settings={}, **ds.trial_metadata(t),
                                             **score_trial(t.probe, raw, t.target_code, t.correct_procedure)))
            seen.add(t.request_key)
        summarize_run(d / "trials.jsonl", d, bootstrap_iterations=50, seed=0)
        assert phrase in (d / "report.md").read_text()
