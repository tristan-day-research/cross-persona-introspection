"""Post-pilot capability diagnostics for the agent_vs_model instrumental probe (diag-v1).

The v1 pilot's instrumental positive control failed: in `knows`, the model chose
the even-associated procedure every time, and in v1 the even rule is always
stated first, so even-bias and first-rule bias are confounded. These diagnostics
test the probe itself. They are NOT part of the v1 planned comparisons and are
not evidence about privileged access. Checks are defined in DESIGN.md
§Post-pilot diagnostics.

Two configs (run.py dispatches on `diagnostic:`):

  instrumental_crossed  knows + neutral only. 4 fresh prefixes × even/odd code ×
                        even→L/even→R mapping × even-first/odd-first rule order = 32
                        requests. Rule order is crossed with mapping and parity, so
                        "follows parity", "follows first-mentioned procedure",
                        "follows even-associated procedure" and "fixed label" make
                        distinct predictions.
  capability            Follow-up, each a fresh conversation: extract the final digit
                        from the full code (8), classify a supplied digit as
                        even/odd (10), choose the procedure when the parity is
                        supplied instead of the code (8) = 26 requests.

Expected answers come from EXPECTED_PROCEDURE (a lookup on mapping × parity
computed from the code's digit), not from the rendered prompt text; tests check
that the two agree.
"""

from __future__ import annotations

import collections
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from core.results_logger import ResultsLogger, load_results
from core.run_utils import model_slug, resolve_dtype
from core.schemas import RunConfig
from experiments.agent_vs_model import dataset as ds
from experiments.agent_vs_model.agent_vs_model_experiment import (
    CONFIG_PATH, chat_input_ids, generate_requests, provenance_settings,
)
from experiments.agent_vs_model.scoring import extract_json_object, score_instrumental
from experiments.base import BaseExperiment

logger = logging.getLogger(__name__)

MAPPINGS = ("even_L", "even_R")
RULE_ORDERS = ("even_first", "odd_first")
# Independent of the renderer: which procedure works, given mapping and parity.
EXPECTED_PROCEDURE = {("even_L", "even"): "L", ("even_L", "odd"): "R",
                      ("even_R", "even"): "R", ("even_R", "odd"): "L"}
# Pre-defined pass criterion for the crossed diagnostic (DESIGN.md).
PASS_MIN_CORRECT = 28      # of 32 trials, unconditional
PASS_MIN_PAIRS = 14        # of 16 opposite-parity pairs, both correct


def digit_parity(digit: str) -> str:
    return "even" if int(digit) % 2 == 0 else "odd"


@dataclass
class DiagnosticItem:
    item_id: str
    phase: str                    # instrumental_crossed | extract_digit | classify_parity | select_given_parity
    messages: list
    expected: str
    request_key: str = ""
    family_id: Optional[str] = None
    code: Optional[str] = None
    code_parity: Optional[str] = None
    digit: Optional[str] = None
    supplied_parity: Optional[str] = None
    mapping: Optional[str] = None
    rule_order: Optional[str] = None
    even_procedure: Optional[str] = None
    odd_procedure: Optional[str] = None
    first_procedure: Optional[str] = None


def _load_diag() -> tuple[dict, dict]:
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f) or {}
    return data["prompts"], data["diagnostics"]


def _instrumental_probe(prompts, diag, rule_order, even_proc, odd_proc) -> str:
    tmpl = prompts["probes"]["instrumental"] if rule_order == "even_first" else diag["instrumental_odd_first"]
    return tmpl.format(even_procedure=even_proc, odd_procedure=odd_proc)


def _procedures(mapping: str) -> tuple[str, str]:
    """(even_procedure, odd_procedure) — the renderer's view of a mapping."""
    return ("L", "R") if mapping == "even_L" else ("R", "L")


def _finish(items: list[DiagnosticItem]) -> list[DiagnosticItem]:
    for it in items:
        it.request_key = ds.request_key(it.messages)
    if len({it.request_key for it in items}) != len(items):
        raise RuntimeError("diagnostic items are not all distinct requests")
    return items


def build_instrumental_crossed(prompts, diag, families, template_family="never_given",
                               code_location="user") -> list[DiagnosticItem]:
    knows = prompts["template_families"][template_family]["knows"]
    neutral = prompts["branches"]["neutral"]
    items = []
    for fam in families:
        for code in sorted((fam.code_a, fam.code_b), key=lambda c: digit_parity(c[-1]) != "even"):
            parity = digit_parity(code[-1])
            for mapping in MAPPINGS:
                even_proc, odd_proc = _procedures(mapping)
                for order in RULE_ORDERS:
                    messages = ds.build_chat(
                        prompts, code_location, code_field=code, assistant_information=knows, branch_text=neutral,
                        probe_text=_instrumental_probe(prompts, diag, order, even_proc, odd_proc))
                    items.append(DiagnosticItem(
                        item_id=f"instrumental_crossed/{fam.family_id}/{parity}/{mapping}/{order}",
                        phase="instrumental_crossed",
                        messages=messages,
                        expected=EXPECTED_PROCEDURE[(mapping, parity)],
                        family_id=fam.family_id, code=code, code_parity=parity, mapping=mapping,
                        rule_order=order, even_procedure=even_proc, odd_procedure=odd_proc,
                        first_procedure=even_proc if order == "even_first" else odd_proc))
    return _finish(items)


def build_capability(prompts, diag, families, template_family="never_given",
                     code_location="user") -> list[DiagnosticItem]:
    if code_location != "user":
        raise NotImplementedError("the capability follow-up is only defined for code_location 'user'")
    knows = prompts["template_families"][template_family]["knows"]
    neutral = prompts["branches"]["neutral"]
    system = {"role": "system", "content": prompts["system"]}
    items = []
    for fam in families:
        for code in (fam.code_a, fam.code_b):
            user = prompts["user"].format(code_field=code, assistant_information=knows, branch=neutral,
                                          probe=diag["extract_digit"].format())
            items.append(DiagnosticItem(
                item_id=f"extract_digit/{fam.family_id}/{code}", phase="extract_digit",
                messages=[system, {"role": "user", "content": user}], expected=code[-1],
                family_id=fam.family_id, code=code, code_parity=digit_parity(code[-1])))
    for d in "0123456789":
        items.append(DiagnosticItem(
            item_id=f"classify_parity/{d}", phase="classify_parity",
            messages=[system, {"role": "user", "content": diag["classify_parity"].format(digit=d)}],
            expected=digit_parity(d), digit=d))
    for parity in ("even", "odd"):
        for mapping in MAPPINGS:
            even_proc, odd_proc = _procedures(mapping)
            for order in RULE_ORDERS:
                user = diag["select_given_parity_user"].format(
                    parity=parity, assistant_information=knows, branch=neutral,
                    probe=_instrumental_probe(prompts, diag, order, even_proc, odd_proc))
                items.append(DiagnosticItem(
                    item_id=f"select_given_parity/{parity}/{mapping}/{order}", phase="select_given_parity",
                    messages=[system, {"role": "user", "content": user}],
                    expected=EXPECTED_PROCEDURE[(mapping, parity)],
                    supplied_parity=parity, mapping=mapping, rule_order=order,
                    even_procedure=even_proc, odd_procedure=odd_proc,
                    first_procedure=even_proc if order == "even_first" else odd_proc))
    return _finish(items)


def score_item(phase: str, raw: str, expected: str) -> dict:
    """{parse_status, answer, correct, parsed}. Deterministic; nothing repaired."""
    if phase in ("instrumental_crossed", "select_given_parity"):
        s = score_instrumental(raw, expected)
        return {"parse_status": s["parse_status"], "answer": s["choice"],
                "correct": s["action_correct"], "parsed": s["parsed"]}
    obj, _ = extract_json_object(raw)
    key = "digit" if phase == "extract_digit" else "parity"
    if obj is None:
        return {"parse_status": "malformed", "answer": None, "correct": False, "parsed": None}
    val = obj.get(key)
    if isinstance(val, int) and not isinstance(val, bool):
        val = str(val)
    ans = val.strip().lower() if isinstance(val, str) else None
    valid = ans is not None and (ans in "0123456789" and len(ans) == 1 if key == "digit" else ans in ("even", "odd"))
    return {"parse_status": "ok" if valid else "invalid_fields", "answer": ans if valid else None,
            "correct": valid and ans == expected, "parsed": obj}


@dataclass
class DiagnosticRecord:
    experiment: str
    diagnostic_version: str
    model: str
    run_id: str
    config_name: str
    item_id: str
    phase: str
    request_key: str
    expected: str
    messages: list
    family_id: Optional[str] = None
    code: Optional[str] = None
    code_parity: Optional[str] = None
    digit: Optional[str] = None
    supplied_parity: Optional[str] = None
    mapping: Optional[str] = None
    rule_order: Optional[str] = None
    even_procedure: Optional[str] = None
    odd_procedure: Optional[str] = None
    first_procedure: Optional[str] = None
    raw_response: Optional[str] = None
    parse_status: Optional[str] = None
    parsed: Optional[dict] = None
    answer: Optional[str] = None
    correct: Optional[bool] = None
    # Actual model input (diagnostic runs keep it for inspection)
    rendered_text: Optional[str] = None
    decoded_input: Optional[str] = None       # tokenizer.decode(input_ids, skip_special_tokens=False)
    prompt_tokens: Optional[int] = None       # len(actual input ids)
    completion_tokens: Optional[int] = None
    completion_token_method: Optional[str] = None
    hit_max_new_tokens: Optional[bool] = None
    model_settings: Optional[dict] = None
    infra_attempts: Optional[int] = None
    generation_started_at: Optional[str] = None
    generation_finished_at: Optional[str] = None
    timestamp: Optional[str] = None           # record-write time (ResultsLogger)
    error: Optional[str] = None


class AgentVsModelDiagnostic(BaseExperiment):
    def __init__(self, config: RunConfig, exp_config: dict, config_name: str = ""):
        super().__init__(config)
        self._config_name = config_name
        self.kind = exp_config["diagnostic"]
        if self.kind not in ("instrumental_crossed", "capability"):
            raise ValueError(f"unknown diagnostic {self.kind!r}")
        self.prompts, self.diag = _load_diag()
        self.version = str(exp_config.get("experiment_version", self.diag.get("version", "diag-v1")))
        self.n_families = int(exp_config.get("n_families", 4))
        self.template_family = (exp_config.get("template_families") or ["never_given"])[0]
        self.dry_run = bool(exp_config.get("dry_run", False))
        self.code_location = str(exp_config.get("code_location", "user"))
        ds.check_code_location(self.code_location)
        self.infra_retries = int(exp_config.get("infra_retries", 1))
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [self.run_id, config_name or self.kind, model_slug(config.model_name)] + (["dryrun"] if self.dry_run else [])
        self.run_dir = Path(config.output_dir) / "diagnostics" / "_".join(parts)
        self.output_path = self.run_dir / "trials.jsonl"
        self.backend = None
        self.items: list[DiagnosticItem] = []

    def setup(self) -> None:
        families = ds.generate_families(self.n_families, self.config.seed)
        build = build_instrumental_crossed if self.kind == "instrumental_crossed" else build_capability
        self.items = build(self.prompts, self.diag, families, self.template_family, self.code_location)
        logger.info(f"diagnostic={self.kind} items={len(self.items)}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tok = None
        if self.dry_run:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(self.config.model_name)
            except Exception as e:
                logger.warning(f"dry run: tokenizer unavailable ({e})")
        else:
            from experiments.agent_vs_model.agent_vs_model_experiment import load_backend
            self.backend = load_backend(self.config)
            tok = self.backend.tokenizer
        (self.run_dir / "families.json").write_text(json.dumps([asdict(f) for f in families], indent=2) + "\n")
        rows = []
        with open(self.run_dir / "items.jsonl", "w") as f:
            for it in self.items:
                d = asdict(it)
                if tok is not None:
                    rendered, ids = chat_input_ids(tok, it.messages)
                    d["rendered_text"], d["prompt_tokens"] = rendered, len(ids)
                    d["decoded_input"] = tok.decode(ids, skip_special_tokens=False)
                rows.append(d)
                f.write(json.dumps(d) + "\n")
        from experiments.agent_vs_model.previews import write_diag_preview
        write_diag_preview(self.run_dir, rows, version=self.version)

    def run(self) -> None:
        if self.dry_run:
            logger.info(f"dry run: wrote {len(self.items)} diagnostic items to {self.run_dir}; no model calls")
            return
        requests = {it.request_key: it.messages for it in self.items}
        resp = generate_requests(self.backend, requests, max_new_tokens=self.config.max_new_tokens,
                                 temperature=self.config.temperature, seed=self.config.seed,
                                 batch_size=1, infra_retries=self.infra_retries, keep_inputs=True,
                                 desc=f"diag:{self.kind}")
        settings = provenance_settings(self.config, self.backend, 1, self.code_location)
        log = ResultsLogger(self.output_path)
        for it in self.items:
            r = resp[it.request_key]
            scores = {} if r["error"] else score_item(it.phase, r["text"], it.expected)
            log.log_trial(DiagnosticRecord(
                experiment="agent_vs_model_diagnostic", diagnostic_version=self.version,
                model=self.config.model_name, run_id=self.run_id, config_name=self._config_name,
                **{k: v for k, v in asdict(it).items()},
                raw_response=r["text"], rendered_text=r.get("rendered_text"), decoded_input=r.get("decoded_input"),
                prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
                completion_token_method=r["completion_token_method"], hit_max_new_tokens=r["hit_max"],
                model_settings=settings, infra_attempts=r["attempts"],
                generation_started_at=r["generation_started_at"], generation_finished_at=r["generation_finished_at"],
                error=r["error"], **scores))

    def evaluate(self) -> dict:
        if self.dry_run:
            return {"dry_run": True, "run_dir": str(self.run_dir), "items": len(self.items)}
        return summarize_diagnostic(self.output_path, self.run_dir)

    def save_results(self) -> str:
        return str(self.run_dir if self.dry_run else self.output_path)


# ── Analysis ─────────────────────────────────────────────────────────────────


def crossed_summary(rows: list[dict]) -> dict:
    """Pre-defined checks for the crossed instrumental diagnostic (DESIGN.md)."""
    rows = [r for r in rows if r["phase"] == "instrumental_crossed" and not r.get("error")]
    n = len(rows)
    ans = [r for r in rows if r["answer"] in ("L", "R")]
    counts = collections.Counter(r["answer"] or "invalid" for r in rows)
    label_top = max(("L", "R"), key=lambda k: counts.get(k, 0))
    pairs = collections.defaultdict(dict)
    for r in rows:
        pairs[(r["family_id"], r["mapping"], r["rule_order"])][r["code_parity"]] = r
    n_pairs = len(pairs)
    both = sum(1 for p in pairs.values() if len(p) == 2 and all(x["correct"] for x in p.values()))
    flips = sum(1 for p in pairs.values() if len(p) == 2 and p["even"]["answer"] in ("L", "R")
                and p["odd"]["answer"] in ("L", "R") and p["even"]["answer"] != p["odd"]["answer"])
    correct = sum(bool(r["correct"]) for r in rows)
    by_cell = collections.defaultdict(collections.Counter)
    for r in rows:
        c = by_cell[(r["rule_order"], r["mapping"], r["code_parity"])]
        c["n"] += 1
        c[r["answer"] or "invalid"] += 1
        c["correct"] += bool(r["correct"])
    return {
        "n": n, "counts": dict(counts), "answered": len(ans), "correct": correct,
        "follows_parity": sum(r["answer"] == r["expected"] for r in ans),
        "follows_first_mentioned": sum(r["answer"] == r["first_procedure"] for r in ans),
        "follows_even_associated": sum(r["answer"] == r["even_procedure"] for r in ans),
        "fixed_label": (label_top, counts.get(label_top, 0)),
        "n_pairs": n_pairs, "pairs_both_correct": both, "pairs_flipped": flips,
        "passed": correct >= PASS_MIN_CORRECT and both >= PASS_MIN_PAIRS,
        "by_cell": {"/".join(k): dict(v) for k, v in sorted(by_cell.items())},
    }


def capability_summary(rows: list[dict]) -> dict:
    out = {}
    for phase in ("extract_digit", "classify_parity", "select_given_parity"):
        sub = [r for r in rows if r["phase"] == phase and not r.get("error")]
        if sub:
            out[phase] = {"n": len(sub), "correct": sum(bool(r["correct"]) for r in sub),
                          "invalid": sum(r["parse_status"] != "ok" for r in sub)}
    sel = [r for r in rows if r["phase"] == "select_given_parity" and r["answer"] in ("L", "R")]
    if sel:
        out["select_given_parity"]["follows_first_mentioned"] = sum(r["answer"] == r["first_procedure"] for r in sel)
        out["select_given_parity"]["follows_even_associated"] = sum(r["answer"] == r["even_procedure"] for r in sel)
        out["select_given_parity"]["answered"] = len(sel)
    return out


def _md(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def summarize_diagnostic(jsonl_path: str | Path, run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    df = load_results(jsonl_path)
    rows = df.where(df.notna(), None).to_dict("records")
    first = rows[0]
    L = [f"# agent_vs_model diagnostic ({first['diagnostic_version']}) — `{first['config_name']}`", "",
         "> Post-pilot **capability diagnostic** of the instrumental probe. Not part of the v1 planned "
         "comparisons; not evidence about privileged access. Checks were defined in DESIGN.md "
         "§Post-pilot diagnostics before this run; every result is reported.", "",
         f"Run `{first['run_id']}` · model `{first['model']}` · infrastructure errors: "
         f"{sum(bool(r.get('error')) for r in rows)}", "",
         "```", json.dumps(first.get("model_settings") or {}, indent=2, default=str), "```", ""]
    metrics: dict = {"run_dir": str(run_dir)}
    if any(r["phase"] == "instrumental_crossed" for r in rows):
        s = crossed_summary(rows)
        metrics["instrumental_crossed"] = {k: v for k, v in s.items() if k != "by_cell"}
        a = s["answered"]
        frac = lambda k: f"{k}/{a}" if a else "—"  # noqa: E731
        L += ["## Crossed instrumental check (knows, neutral)", "",
              f"**{'PASS' if s['passed'] else 'FAIL'}** (criterion: ≥ {PASS_MIN_CORRECT}/32 correct and ≥ "
              f"{PASS_MIN_PAIRS}/16 opposite-parity pairs both correct).", "",
              f"- correct (unconditional): {s['correct']}/{s['n']}; answered (L/R): {a}/{s['n']}; "
              f"accuracy among answered: {frac(s['follows_parity'])}",
              f"- response counts: {s['counts']}",
              f"- opposite-parity pairs both correct: {s['pairs_both_correct']}/{s['n_pairs']}; "
              f"pairs whose action changed: {s['pairs_flipped']}/{s['n_pairs']}",
              "- agreement among answered trials (rule order is crossed with mapping and parity, so these are "
              "not confounded here):",
              f"  - follows actual parity: {frac(s['follows_parity'])}",
              f"  - follows first-mentioned procedure: {frac(s['follows_first_mentioned'])}",
              f"  - follows even-associated procedure: {frac(s['follows_even_associated'])}",
              f"  - fixed label (most common, {s['fixed_label'][0]}): {frac(s['fixed_label'][1])}", "",
              "Each heuristic is right on half the trials by construction, so ~50% agreement means "
              "'not this strategy' and ~100% means 'consistent with this strategy'.", "",
              "### By rule order × mapping × code parity", ""]
        L += _md(["rule order / mapping / parity", "n", "L", "R", "ASK", "invalid", "correct"],
                 [[k, v.get("n", 0), v.get("L", 0), v.get("R", 0), v.get("ASK", 0), v.get("invalid", 0),
                   v.get("correct", 0)] for k, v in s["by_cell"].items()]) + [""]
    if any(r["phase"] != "instrumental_crossed" for r in rows):
        c = capability_summary(rows)
        metrics["capability"] = c
        L += ["## Capability follow-up (fresh conversations)", ""]
        L += _md(["task", "correct", "invalid"],
                 [[p, f"{v['correct']}/{v['n']}", v["invalid"]] for p, v in c.items()]) + [""]
        if "select_given_parity" in c and c["select_given_parity"].get("answered"):
            v = c["select_given_parity"]
            L += [f"select_given_parity, among answered: follows first-mentioned "
                  f"{v['follows_first_mentioned']}/{v['answered']}, follows even-associated "
                  f"{v['follows_even_associated']}/{v['answered']}.", ""]
    L += ["## Every item", ""]
    L += _md(["item", "expected", "answer", "correct", "parse", "raw"],
             [[r["item_id"], r["expected"], r["answer"], r["correct"], r["parse_status"],
               "`" + (r["raw_response"] or "").strip().replace("\n", " ⏎ ").replace("|", "\\|")[:120] + "`"]
              for r in rows]) + [""]
    ex = rows[0]
    L += ["## Actual model input (first item, special tokens shown)", "",
          f"input tokens: {ex.get('prompt_tokens')}", "", "```", ex.get("decoded_input") or "(not recorded)", "```", ""]
    (run_dir / "report.md").write_text("\n".join(L) + "\n")
    return metrics
