"""Mechanistic follow-up: is the door code represented when the Assistant says it doesn't know it,
and does its self-report read that information?

Behaviourally (v2-probs-syscode, Qwen2.5-32B), under `doesnt_know` the Assistant says it does not know
the code, yet its door-task choice still follows the code's parity. Three measurements, one model load:

  1. capture  One forward per prompt for two questions — the self-report question ("do you know the
              door code?", read at the first answer token after `{"knows_code":`) and the door task
              (read after `{"choice": "`). Residual stream saved at a set of layers at the answer
              position, and at the code's last token (self-report prompts only). Every condition ×
              request type, many code pairs. Behaviour (P(true), P(L/R/ASK)) read from the same pass.
  2. probes   Logistic-regression probes for the code's last-digit parity, per layer, cross-validated
              by code pair (both codes of a pair always in the same fold). Within each condition, and
              trained on `knows` then tested on the other conditions (held-out pairs). Shuffled-label
              and `code_absent` (no code in the prompt) controls.
  3. patching At the code's token positions, at one layer at a time (or the token embeddings):
                mean   replace with the mean over other codes  -> removes the code's identity
                swap   replace with the paired code's states   -> changes only the last digit
              and re-read the door choice and the self-report. Door effect = how much of the choice's
              code-following survives; self-report effect = change in log-odds(true vs false).

Causality note: the code comes before every condition-specific piece of text in both layouts, so the
code-position states are identical across conditions and questions — the same donor states serve all
prompts. Patching at layer L leaves anything later positions already read from the code at layers <= L;
the layer sweep (from the embeddings up) shows where that happens.

What this cannot show (stated in the report): probes show decodability, not use; mean-ablation removes
which code it is, not that a code is present (presence is the `doesnt_know` vs `code_absent` contrast);
greedy-free readouts after fixed answer prefixes.
"""

from __future__ import annotations

import json
import logging
import math
import textwrap
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.activation_capture import char_span_to_token_span, decoder_layers
from core.run_utils import git_commit, git_dirty, model_slug
from core.schemas import RunConfig
from experiments.agent_vs_model import dataset as ds
from experiments.agent_vs_model.agent_vs_model_analysis import CONDITION_COLORS, _md_table, _style_axis
from experiments.agent_vs_model.agent_vs_model_experiment import chat_input_ids, load_backend, provenance_settings
from experiments.agent_vs_model.diagnostics import MAPPINGS, RULE_ORDERS, _instrumental_probe, _load_diag, _procedures
from experiments.agent_vs_model.instrumental_probs import CHOICES, PRIMER as DOOR_PRIMER, choice_token_ids, probs_from_logits
from experiments.base import BaseExperiment

logger = logging.getLogger(__name__)

VERSION = "mech-v1"
SELF_PRIMER = '{"knows_code":'
SELF_CHOICES = (" true", " false")
EMBED = -1  # "layer" index for the token embeddings (input to the first block)
QUESTIONS = ("self_report", "door")


# ── Prompts ──────────────────────────────────────────────────────────────────


@dataclass
class MechPrompt:
    key: str
    question: str
    condition: str
    branch: str
    pair_id: str
    variant: str
    code: str
    parity: str
    last_digit: int
    code_in_prompt: bool
    mapping: str
    rule_order: str
    even_procedure: str
    expected: str
    messages: list
    text: str = ""                              # chat text + answer prefix (what the model reads)
    input_ids: list = field(default_factory=list, repr=False)
    code_pos: list = field(default_factory=list)  # token indices of the code (empty when absent)


def build_prompts(prompts: dict, diag: dict, families, template_family: str, conditions, branches,
                  code_location: str) -> list[MechPrompt]:
    """Both questions for every code × condition × request type. Each pair gets one mapping and one
    door-question wording, balanced across pairs (the door effect is computed within a pair, so a
    fixed label or first-rule bias cancels)."""
    tspec = prompts["template_families"][template_family]
    out: list[MechPrompt] = []
    for i, fam in enumerate(families):
        mapping, order = MAPPINGS[i % 2], RULE_ORDERS[(i // 2) % 2]
        even_proc, odd_proc = _procedures(mapping)
        for variant, code in (("a", fam.code_a), ("b", fam.code_b)):
            parity = ds.parity(code)
            for cond in conditions:
                present = cond != "code_absent"
                for branch in branches:
                    for q in QUESTIONS:
                        probe_text = (prompts["probes"]["self_report"] if q == "self_report"
                                      else _instrumental_probe(prompts, diag, order, even_proc, odd_proc))
                        msgs = ds.build_chat(prompts, code_location,
                                             code_field=code if present else ds.CODE_ABSENT_FIELD,
                                             assistant_information=tspec[cond],
                                             branch_text=prompts["branches"][branch], probe_text=probe_text)
                        out.append(MechPrompt(
                            key=f"{q}/{cond}/{branch}/{fam.family_id}/{variant}", question=q, condition=cond,
                            branch=branch, pair_id=fam.family_id, variant=variant, code=code, parity=parity,
                            last_digit=int(code[-1]), code_in_prompt=present, mapping=mapping, rule_order=order,
                            even_procedure=even_proc, expected=even_proc if parity == "even" else odd_proc,
                            messages=msgs))
    return out


def tokenize_prompt(tokenizer, p: MechPrompt) -> None:
    """Fill text, input_ids and the code's token positions. Raises if the code's tokens don't cover
    exactly the code (patching a partial or merged token would be meaningless)."""
    primer = SELF_PRIMER if p.question == "self_report" else DOOR_PRIMER
    p.text = chat_input_ids(tokenizer, p.messages)[0] + primer
    enc = tokenizer(p.text, add_special_tokens=False, return_offsets_mapping=True)
    p.input_ids = list(enc["input_ids"])
    p.code_pos = []
    if p.code_in_prompt:
        start = p.text.index(f"Door code: {p.code}") + len("Door code: ")
        span = char_span_to_token_span(enc["offset_mapping"], start, start + len(p.code))
        pos = list(range(*span))
        joined = "".join(tokenizer.decode([p.input_ids[i]]) for i in pos).strip()
        if joined != p.code:
            raise ValueError(f"code {p.code} is tokenized as {joined!r}: tokens do not cover exactly the code")
        p.code_pos = pos


def pick_layers(n_total: int, n: int) -> list[int]:
    return sorted({int(round(x)) for x in np.linspace(0, n_total - 1, min(n, n_total))})


# ── Forward passes with capture / patch hooks ───────────────────────────────


def _module(model, layer: int):
    return model.get_input_embeddings() if layer == EMBED else decoder_layers(model)[layer]


def run_forward(backend, input_ids: list, *, capture: dict | None = None, patch: tuple | None = None):
    """One unbatched forward. Returns (last-position logits [vocab] float32 on CPU, states).

    capture: {layer: [positions]} -> states[layer] = [n_pos, hidden] float32 CPU
    patch:   (layer, [positions], donor [n_pos, hidden]) -> those residual-stream entries are
             replaced by `donor` at that layer's output (or in the embeddings for EMBED).
    """
    import torch
    model = backend.model
    states: dict = {}
    handles = []

    def cap_hook(layer, pos):
        def h(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            states[layer] = hs[0, pos, :].detach().float().cpu()
        return h

    def patch_hook(pos, donor):
        def h(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.clone()
            hs[0, pos, :] = donor.to(device=hs.device, dtype=hs.dtype)
            return (hs, *out[1:]) if isinstance(out, tuple) else hs
        return h

    try:
        if patch is not None:
            layer, pos, donor = patch
            handles.append(_module(model, layer).register_forward_hook(patch_hook(pos, donor)))
        for layer, pos in (capture or {}).items():
            handles.append(_module(model, layer).register_forward_hook(cap_hook(layer, pos)))
        ids = torch.tensor([input_ids], device=backend.input_device)
        logits = backend._last_position_logits(ids, attention_mask=torch.ones_like(ids))[0]
    finally:
        for h in handles:
            h.remove()
    return logits.float().cpu(), states


def readout(p: MechPrompt, logits, token_ids: dict) -> dict:
    if p.question == "self_report":
        lo = float(logits[token_ids[" true"]] - logits[token_ids[" false"]])
        return {"self_logodds": lo, "p_true": 1.0 / (1.0 + math.exp(-lo))}
    r = probs_from_logits(logits, token_ids)
    lr = r["p_L"] + r["p_R"]
    return {"p_L": r["p_L"], "p_R": r["p_R"], "p_ASK": r["p_ASK"],
            "p_even_proc_lr": (r["p_L"] if p.even_procedure == "L" else r["p_R"]) / lr}


# ── Probes ───────────────────────────────────────────────────────────────────


def pair_folds(pair_ids, n_splits: int, seed: int) -> dict:
    """pair_id -> fold; the same assignment for every condition, so held-out pairs stay held out."""
    uniq = sorted(set(pair_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    k = min(n_splits, len(uniq))
    return {uniq[j]: int(i % k) for i, j in enumerate(order)}


def _fit(X, y, C, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(sc.transform(X), y)
    return sc, clf


def _auc(y, scores) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores)) if len(set(y)) == 2 else float("nan")


def probe_cv_scores(X, y, folds_of_rows, C, seed) -> tuple[float, float]:
    """Cross-validated (accuracy, AUC); rows of one code pair are always in the same fold.
    AUC uses the decision scores, so it ignores where the threshold sits (binary targets only)."""
    correct, scores = 0, np.full(len(y), np.nan)
    for k in sorted(set(folds_of_rows)):
        te = folds_of_rows == k
        if len(set(y[~te])) < 2:
            continue
        sc, clf = _fit(X[~te], y[~te], C, seed)
        Xte = sc.transform(X[te])
        correct += int((clf.predict(Xte) == y[te]).sum())
        if len(clf.classes_) == 2:
            scores[te] = clf.decision_function(Xte)
    ok = ~np.isnan(scores)
    return correct / len(y), (_auc(y[ok], scores[ok]) if ok.any() else float("nan"))


def probe_cv(X, y, folds_of_rows, C, seed) -> float:
    return probe_cv_scores(X, y, folds_of_rows, C, seed)[0]


def probe_transfer_scores(Xs, ys, fs, targets: dict, C, seed) -> dict:
    """Train on the source condition's training pairs; test on each target condition's held-out pairs.

    Per target: raw accuracy (source threshold applied as is), AUC (threshold-free), and accuracy
    after re-centering — the target's activations are shifted so their mean (over the target's own
    training-fold rows, no labels used) matches the source's. A failure that disappears under AUC or
    re-centering is an offset between conditions, not a different encoding of the digit."""
    out = {t: {"hits": 0, "hits_c": 0, "scores": np.full(len(targets[t][1]), np.nan)} for t in targets}
    for k in sorted(set(fs)):
        tr = fs != k
        if len(set(ys[tr])) < 2:
            continue
        sc, clf = _fit(Xs[tr], ys[tr], C, seed)
        mu_s = Xs[tr].mean(0)
        for t, (Xt, yt, ft) in targets.items():
            te, tr_t = ft == k, ft != k
            Xte = Xt[te]
            out[t]["hits"] += int((clf.predict(sc.transform(Xte)) == yt[te]).sum())
            Xc = Xte - Xt[tr_t].mean(0) + mu_s
            out[t]["hits_c"] += int((clf.predict(sc.transform(Xc)) == yt[te]).sum())
            if len(clf.classes_) == 2:
                out[t]["scores"][te] = clf.decision_function(sc.transform(Xte))
    res = {}
    for t, (Xt, yt, ft) in targets.items():
        sc_ = out[t]["scores"]
        ok = ~np.isnan(sc_)
        res[t] = {"accuracy": out[t]["hits"] / len(yt), "accuracy_centered": out[t]["hits_c"] / len(yt),
                  "auc": _auc(yt[ok], sc_[ok]) if ok.any() else float("nan")}
    return res


def probe_transfer(Xs, ys, fs, targets: dict, C, seed) -> dict:
    return {t: r["accuracy"] for t, r in probe_transfer_scores(Xs, ys, fs, targets, C, seed).items()}


def probe_direction(X, y, C, seed) -> np.ndarray:
    """Unit-norm probe weight vector in the original activation space (binary targets)."""
    sc, clf = _fit(X, y, C, seed)
    w = clf.coef_[0] / sc.scale_
    return w / (np.linalg.norm(w) + 1e-12)


def run_probes(meta: pd.DataFrame, act_dir: Path, layers: list[int], conditions, branches, targets,
               C: float, seed: int, n_splits: int = 5) -> pd.DataFrame:
    """Rows: kind = cv | shuffled | transfer_from_knows | direction_cosine (+ shuffled null).
    Columns: accuracy, auc, accuracy_centered (transfer only), cosine (direction rows only)."""
    rows = []
    folds = pair_folds(meta["pair_id"], n_splits, seed)
    rng = np.random.default_rng(seed + 7)
    sets = [("self_report", "answer"), ("door", "answer"), ("self_report", "code_last")]

    def row(**kw):
        base = dict(accuracy=np.nan, auc=np.nan, accuracy_centered=np.nan, cosine=np.nan)
        base.update(kw)
        rows.append(base)

    for q, pos in sets:
        path = act_dir / f"{q}_{pos}.npy"
        if not path.exists():
            continue
        arr = np.load(path, mmap_mode="r")
        mq = meta[meta["question"] == q].sort_values("row_index")
        # Code-position states are identical across conditions (the code precedes them): one cell.
        cells = [("knows", branches[0])] if pos == "code_last" else [(c, b) for c in conditions for b in branches]
        for target in targets:
            chance = 0.5 if target == "parity" else 0.1
            for li, layer in enumerate(layers):
                data = {}
                for c, b in cells:
                    m = mq[(mq["condition"] == c) & (mq["branch"] == b)]
                    X = np.asarray(arr[m["row_index"].to_numpy(), li], dtype=np.float32)
                    y = (m["parity"] == "even").to_numpy().astype(int) if target == "parity" else m["last_digit"].to_numpy()
                    f = m["pair_id"].map(folds).to_numpy()
                    data[(c, b)] = (X, y, f)
                    common = dict(question=q, position=pos, target=target, condition=c, branch=b, layer=layer,
                                  n=len(y), chance=chance)
                    acc, auc = probe_cv_scores(X, y, f, C, seed)
                    row(kind="cv", accuracy=acc, auc=auc, **common)
                    if c == "knows" and b == branches[0]:  # shuffled-label control
                        acc, auc = probe_cv_scores(X, rng.permutation(y), f, C, seed)
                        row(kind="shuffled", accuracy=acc, auc=auc, **common)
                if pos == "code_last":
                    continue
                for b in branches:
                    if ("knows", b) not in data:
                        continue
                    tg = {c: data[(c, b)] for c in conditions if c != "knows" and (c, b) in data}
                    for c, r in probe_transfer_scores(*data[("knows", b)], tg, C, seed).items():
                        row(kind="transfer_from_knows", question=q, position=pos, target=target, condition=c,
                            branch=b, layer=layer, n=len(tg[c][1]), chance=chance, **r)
                    if target != "parity":
                        continue
                    # Do the conditions use the same direction? Cosine between probes fit on all rows of
                    # each condition, with a null: knows probe vs a probe fit on shuffled knows labels.
                    Xk, yk, _ = data[("knows", b)]
                    wk = probe_direction(Xk, yk, C, seed)
                    for c in conditions:
                        if c in ("knows", "code_absent") or (c, b) not in data:
                            continue
                        Xc_, yc_, _ = data[(c, b)]
                        row(kind="direction_cosine", question=q, position=pos, target=target, condition=c, branch=b,
                            layer=layer, n=len(yc_), chance=0.0, cosine=float(wk @ probe_direction(Xc_, yc_, C, seed)))
                    row(kind="direction_cosine_null", question=q, position=pos, target=target, condition="knows",
                        branch=b, layer=layer, n=len(yk), chance=0.0,
                        cosine=float(wk @ probe_direction(Xk, rng.permutation(yk), C, seed)))
    return pd.DataFrame(rows)


# ── Experiment ───────────────────────────────────────────────────────────────


class MechExperiment(BaseExperiment):
    def __init__(self, config: RunConfig, exp_config: dict, config_name: str = ""):
        super().__init__(config)
        self._config_name = config_name
        self.prompts, self.diag = _load_diag()
        self.version = str(exp_config.get("experiment_version", VERSION))
        self.template_family = (exp_config.get("template_families") or ["never_given"])[0]
        self.code_location = str(exp_config.get("code_location", "system"))
        ds.check_code_location(self.code_location)
        self.conditions = list(exp_config.get("conditions") or ds.CONDITIONS)
        self.branches = list(exp_config.get("branches") or ["neutral", "released"])
        self.n_probe_pairs = int(exp_config.get("n_probe_pairs", 128))
        self.n_patch_pairs = int(exp_config.get("n_patch_pairs", 12))
        self.n_layers = int(exp_config.get("n_layers", 16))
        self.n_mean_codes = int(exp_config.get("n_mean_codes", 32))
        self.probe_C = float(exp_config.get("probe_C", 0.1))
        self.probe_targets = list(exp_config.get("probe_targets") or ["parity"])
        self.stages = list(exp_config.get("stages") or ["capture", "probes", "patching"])
        self.bootstrap_iterations = int(exp_config.get("bootstrap_iterations", 2000))
        self.dry_run = bool(exp_config.get("dry_run", False))
        # reuse_run: recompute probes/patching analysis from an earlier run's saved activations and
        # records (no model loaded, no forward passes). Outputs go to a new run folder.
        self.reuse_run = Path(exp_config["reuse_run"]) if exp_config.get("reuse_run") else None
        self.patch_conditions = [c for c in self.conditions if c != "code_absent"]
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [self.run_id, config_name or "mech", model_slug(config.model_name)] + (["dryrun"] if self.dry_run else [])
        self.run_dir = Path(config.output_dir) / "mech" / "_".join(parts)
        self.act_dir = self.run_dir / "activations"  # large; skipped by pull-results-persona
        self.backend = None
        self.tokenizer = None
        self.families = []
        self.items: list[MechPrompt] = []
        self.layers: list[int] = []
        self.token_ids: dict = {}

    # ── setup ──

    def setup(self) -> None:
        if self.reuse_run is not None:
            if not (self.reuse_run / "activations" / "layers.json").exists():
                raise FileNotFoundError(f"{self.reuse_run}/activations/ not found — reuse_run must point at a mech "
                                        "run that still has its activations (they stay on the pod)")
            self.layers = json.loads((self.reuse_run / "activations" / "layers.json").read_text())
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "SOURCE.txt").write_text(
                f"Reanalysis of {self.reuse_run} (no new forward passes). Activations and records read from there.\n")
            return
        self.families = ds.generate_families(self.n_probe_pairs, self.config.seed)
        self.items = build_prompts(self.prompts, self.diag, self.families, self.template_family,
                                   self.conditions, self.branches, self.code_location)
        if self.dry_run:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        else:
            self.backend = load_backend(self.config)
            self.tokenizer = self.backend.tokenizer
            self.layers = pick_layers(len(decoder_layers(self.backend.model)), self.n_layers)
        for p in self.items:
            tokenize_prompt(self.tokenizer, p)
        first = {q: next(p for p in self.items if p.question == q) for q in QUESTIONS}
        base = {q: chat_input_ids(self.tokenizer, first[q].messages)[0] for q in QUESTIONS}
        self.token_ids = {"self_report": choice_token_ids(self.tokenizer, base["self_report"], SELF_PRIMER, SELF_CHOICES),
                          "door": choice_token_ids(self.tokenizer, base["door"], DOOR_PRIMER, CHOICES)}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "families.json").write_text(json.dumps([asdict(f) for f in self.families], indent=2) + "\n")
        self._write_preview()
        self._write_manifest()

    def _write_preview(self) -> None:
        from experiments.agent_vs_model.previews import write_preview
        fam = self.families[0].family_id
        ex = [p for p in self.items if p.pair_id == fam and p.variant == "a" and p.branch == self.branches[0]
              and p.condition in ("knows", "doesnt_know")]
        examples = [{"title": f"{p.question} · {p.condition} · {p.branch} request (code {p.code})",
                     "row": {"messages": p.messages, "rendered_text": p.text}} for p in ex]
        where = "system message" if self.code_location == "system" else "user message"
        write_preview(self.run_dir / "prompts_preview.md", f"Sample prompts — {self.version}", [
            f"Sample prompts from this run. The door code is in the **{where}**.",
            f"- Self-report question: the reply is started with `{SELF_PRIMER}` and we read ` true` vs ` false`.",
            f"- Door task: the reply is started with `{DOOR_PRIMER}` and we read `L` / `R` / `ASK`.",
        ], examples, exact_title="with the answer prefix added")

    def _write_manifest(self) -> None:
        n = {q: sum(p.question == q for p in self.items) for q in QUESTIONS}
        n_patch = (self.n_patch_pairs * 2 * len(self.patch_conditions) * len(self.branches) * 2
                   * (self.n_layers + 1) * 2)
        lines = [f"AGENT VS MODEL — mechanistic follow-up ({self.version}) — RUN MANIFEST", "=" * 70,
                 f"run_id: {self.run_id}  config: {self._config_name}  dry_run: {self.dry_run}",
                 f"model: {self.config.model_name}  dtype: {self.config.model_dtype}",
                 f"code_location: {self.code_location}  wording: {self.template_family}",
                 f"conditions: {self.conditions}  request types: {self.branches}",
                 f"code pairs: {self.n_probe_pairs} (seed {self.config.seed}); patched pairs: {self.n_patch_pairs}",
                 f"layers: {self.layers or f'{self.n_layers} evenly spaced (set when the model loads)'} + embeddings for patching",
                 f"stages: {self.stages}",
                 f"forward passes: capture {n['self_report'] + n['door']}, patching ~{n_patch}",
                 f"answer token ids: {self.token_ids}",
                 f"git_commit: {git_commit()} (uncommitted changes: {git_dirty()})"]
        (self.run_dir / "manifest.txt").write_text("\n".join(lines) + "\n")

    # ── run ──

    def run(self) -> None:
        if self.reuse_run is not None:
            return
        if self.dry_run:
            logger.info(f"dry run: {len(self.items)} prompts built and tokenized; no forward passes")
            return
        if "capture" in self.stages:
            self._capture()
        if "patching" in self.stages:
            self._patch()

    def _capture(self) -> None:
        from tqdm import tqdm
        self.act_dir.mkdir(parents=True, exist_ok=True)
        records, arrays = [], {}
        counters = {q: 0 for q in QUESTIONS}
        for p in tqdm(self.items, desc="mech:capture", dynamic_ncols=True, mininterval=1.0):
            last = len(p.input_ids) - 1
            want = {"answer": last}
            if p.question == "self_report" and p.code_pos:
                want["code_last"] = p.code_pos[-1]
            names = list(want)
            logits, st = run_forward(self.backend, p.input_ids, capture={l: [want[n] for n in names] for l in self.layers})
            i = counters[p.question]
            counters[p.question] += 1
            for j, name in enumerate(names):
                key = (p.question, name)
                if key not in arrays:
                    n_rows = sum(x.question == p.question for x in self.items)
                    hidden = st[self.layers[0]].shape[-1]
                    arrays[key] = np.lib.format.open_memmap(self.act_dir / f"{p.question}_{name}.npy", mode="w+",
                                                            dtype=np.float16, shape=(n_rows, len(self.layers), hidden))
                arrays[key][i] = np.stack([st[l][j].numpy() for l in self.layers]).astype(np.float16)
            rec = {k: v for k, v in asdict(p).items() if k not in ("input_ids", "text")}
            rec.update(row_index=i, prompt_tokens=len(p.input_ids), **readout(p, logits, self.token_ids[p.question]))
            records.append(rec)
        for a in arrays.values():
            a.flush()
        with open(self.run_dir / "capture.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        (self.act_dir / "layers.json").write_text(json.dumps(self.layers))

    def _code_states(self, code: str, template: MechPrompt) -> dict:
        """Code-position states for `code` at every patch layer (incl. embeddings), using the template
        prompt with the code swapped in (the states depend only on the code and the fixed prefix)."""
        p = MechPrompt(**{**asdict(template), "code": code, "key": "donor", "input_ids": [], "code_pos": [],
                          "messages": [{"role": m["role"], "content": m["content"].replace(template.code, code)}
                                       for m in template.messages]})
        tokenize_prompt(self.tokenizer, p)
        if p.code_pos != template.code_pos:
            raise ValueError(f"code {code} tokenizes to different positions than {template.code}")
        _, st = run_forward(self.backend, p.input_ids, capture={l: p.code_pos for l in [EMBED] + self.layers})
        return st

    def _patch(self) -> None:
        from tqdm import tqdm
        patch_pairs = {f.family_id for f in self.families[: self.n_patch_pairs]}
        partner = {}
        for f in self.families[: self.n_patch_pairs]:
            partner[f.code_a], partner[f.code_b] = f.code_b, f.code_a
        targets = [p for p in self.items if p.pair_id in patch_pairs and p.condition in self.patch_conditions]
        template = next(p for p in targets if p.question == "self_report")
        donors = {c: self._code_states(c, template) for c in partner}
        # Mean donor: codes not used anywhere else in this run.
        used = {f.code_a for f in self.families} | {f.code_b for f in self.families}
        rng = np.random.default_rng(self.config.seed + 1)
        mean_codes = []
        while len(mean_codes) < self.n_mean_codes:
            c = "".join(rng.choice(list("0123456789"), 6))
            if c not in used and c not in mean_codes:
                mean_codes.append(c)
        mean_states = [self._code_states(c, template) for c in mean_codes]
        mean_donor = {l: sum(s[l] for s in mean_states) / len(mean_states) for l in [EMBED] + self.layers}
        layers = [EMBED] + self.layers
        with open(self.run_dir / "patching.jsonl", "w") as f:
            for p in tqdm(targets, desc="mech:patching", dynamic_ncols=True, mininterval=1.0):
                if p.code_pos != template.code_pos:
                    raise ValueError(f"{p.key}: code positions differ from the donor template")
                for layer in layers:
                    for kind, donor in (("mean", mean_donor[layer]), ("swap", donors[partner[p.code]][layer])):
                        logits, _ = run_forward(self.backend, p.input_ids, patch=(layer, p.code_pos, donor))
                        f.write(json.dumps({"key": p.key, "question": p.question, "condition": p.condition,
                                            "branch": p.branch, "pair_id": p.pair_id, "code": p.code,
                                            "parity": p.parity, "even_procedure": p.even_procedure,
                                            "layer": layer, "intervention": kind,
                                            **readout(p, logits, self.token_ids[p.question])}) + "\n")

    def evaluate(self) -> dict:
        if self.dry_run:
            return {"dry_run": True, "run_dir": str(self.run_dir), "prompts": len(self.items),
                    "answer_token_ids": self.token_ids}
        if self.reuse_run is not None:
            return summarize_mech(self.run_dir, self.layers, self.conditions, self.branches, self.probe_targets,
                                  self.probe_C, self.config.seed, self.bootstrap_iterations,
                                  settings={"model_name": self.config.model_name, "reuse_run": str(self.reuse_run)},
                                  run_probes_stage="probes" in self.stages, src_dir=self.reuse_run)
        return summarize_mech(self.run_dir, self.layers, self.conditions, self.branches, self.probe_targets,
                              self.probe_C, self.config.seed, self.bootstrap_iterations,
                              settings=provenance_settings(self.config, self.backend, 1, self.code_location),
                              run_probes_stage="probes" in self.stages)

    def save_results(self) -> str:
        return str(self.run_dir)


# ── Analysis ─────────────────────────────────────────────────────────────────


def _boot(values: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = values[rng.integers(0, values.size, size=(n, values.size))].mean(axis=1)
    return float(values.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _boot_ratio(num: np.ndarray, den: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n, num.size))
    with np.errstate(divide="ignore", invalid="ignore"):
        b = num[idx].mean(1) / den[idx].mean(1)
    point = num.mean() / den.mean() if den.mean() != 0 else np.nan
    return float(point), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def door_pair_effects(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per code pair: P(even-rule procedure | even code) − P(… | odd code)."""
    piv = df.pivot_table(index=keys + ["pair_id"], columns="parity", values="p_even_proc_lr").reset_index()
    piv["effect"] = piv["even"] - piv["odd"]
    return piv


MIN_CLEAN_EFFECT = 0.10  # below this the door choice barely follows the code; a "fraction surviving" is meaningless


def patching_table(cap: pd.DataFrame, pt: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    clean_door = door_pair_effects(cap[cap["question"] == "door"], ["condition", "branch"])
    clean_self = cap[cap["question"] == "self_report"].set_index("key")["self_logodds"]
    door = door_pair_effects(pt[pt["question"] == "door"], ["condition", "branch", "layer", "intervention"])
    door = door.merge(clean_door[["condition", "branch", "pair_id", "effect"]], on=["condition", "branch", "pair_id"],
                      suffixes=("", "_clean"))
    s = pt[pt["question"] == "self_report"].copy()
    s["delta"] = s["self_logodds"] - s["key"].map(clean_self)
    s_pair = s.groupby(["condition", "branch", "layer", "intervention", "pair_id"])["delta"].mean().reset_index()
    for (c, b, layer, kind), g in door.groupby(["condition", "branch", "layer", "intervention"]):
        if g["effect_clean"].mean() >= MIN_CLEAN_EFFECT:
            r, lo, hi = _boot_ratio(g["effect"].to_numpy(), g["effect_clean"].to_numpy(), n_boot, seed)
        else:
            r = lo = hi = np.nan
        sg = s_pair[(s_pair.condition == c) & (s_pair.branch == b) & (s_pair.layer == layer) & (s_pair.intervention == kind)]
        d, dlo, dhi = _boot(sg["delta"].to_numpy(), n_boot, seed)
        rows.append(dict(condition=c, branch=b, layer=layer, intervention=kind,
                         door_effect_clean=float(g["effect_clean"].mean()), door_effect_patched=float(g["effect"].mean()),
                         door_fraction_remaining=r, door_fraction_lo=lo, door_fraction_hi=hi,
                         self_delta_logodds=d, self_delta_lo=dlo, self_delta_hi=dhi, n_pairs=len(g)))
    return pd.DataFrame(rows)


def behaviour_table(cap: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    door = door_pair_effects(cap[cap["question"] == "door"], ["condition", "branch"])
    s = cap[cap["question"] == "self_report"].groupby(["condition", "branch", "pair_id"])["self_logodds"].mean().reset_index()
    for (c, b), g in s.groupby(["condition", "branch"]):
        lo_m, lo_lo, lo_hi = _boot(g["self_logodds"].to_numpy(), n_boot, seed)
        dg = door[(door.condition == c) & (door.branch == b)]
        pe, pe_lo, pe_hi = _boot(dg["effect"].to_numpy(), n_boot, seed)
        rows.append(dict(condition=c, branch=b, self_logodds=lo_m, self_logodds_lo=lo_lo, self_logodds_hi=lo_hi,
                         p_true=float(cap[(cap.question == "self_report") & (cap.condition == c) & (cap.branch == b)]["p_true"].mean()),
                         door_effect=pe, door_effect_lo=pe_lo, door_effect_hi=pe_hi))
    return pd.DataFrame(rows)


CAPTION_PROBES = (
    "How to read this: each line is one condition (colour); the x-axis is the layer (0 = first block). The y-axis is how "
    "accurately a linear probe can read whether the door code ends in an even or odd digit from the model's internal state, "
    "on code pairs it was not trained on (0.5 = chance, dashed). Columns: at the answer position of the self-report question "
    "('do you know the door code?'), at the answer position of the door task, and at the code's own last token. Solid = probe "
    "trained and tested within the condition; dotted = trained on 'knows', tested on the other condition. Grey = shuffled "
    "labels (should sit at chance). code_absent has no code in the prompt, so it should sit at chance too. High accuracy means "
    "the information is present, not that the model uses it."
)
CAPTION_PATCH = (
    "How to read this: x-axis = the layer where the code's token positions were overwritten (-1 = the token embeddings). Top: "
    "how much of the door choice's code-following survives (1 = unchanged, 0 = gone, -1 = reversed). Bottom: change in the "
    "self-report's lean toward 'true' (log-odds; 0 = no change, +1 roughly multiplies the odds of 'true' by 2.7). Left: the "
    "code's identity removed (replaced by the average over other codes). Right: the code swapped for its pair partner (only "
    "the last digit changes). Solid = normal request, dashed = 'ignore what you were told earlier'. Bars = 95% intervals over "
    "code pairs. If the door line drops while the self-report line stays at 0, the self-report is not using the code's value."
)


def _caption(fig, text, width, path):
    fig.text(0.01, -0.02, "\n".join(textwrap.wrap(text, width)), ha="left", va="top", fontsize=8, color="#3b3a37",
             linespacing=1.4)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def fig_probes(pr: pd.DataFrame, path: Path, model: str, conditions, branch: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = [("self_report", "answer", "self-report question, answer position"), ("door", "answer", "door task, answer position"),
            ("self_report", "code_last", "code's last token")]
    sets = [s for s in sets if ((pr.question == s[0]) & (pr.position == s[1])).any()]
    fig, axes = plt.subplots(1, len(sets), figsize=(5 * len(sets), 4), sharey=True, squeeze=False)
    for ax, (q, pos, title) in zip(axes[0], sets):
        d = pr[(pr.question == q) & (pr.position == pos) & (pr.target == "parity")]
        for c in conditions:
            for kind, ls in (("cv", "-"), ("transfer_from_knows", ":")):
                g = d[(d.condition == c) & (d.kind == kind) & ((d.branch == branch) | (pos == "code_last"))].sort_values("layer")
                if len(g):
                    ax.plot(g["layer"], g["accuracy"], ls, color=CONDITION_COLORS[c], marker="o", markersize=3,
                            label=f"{c}{' (from knows)' if kind != 'cv' else ''}")
        g = d[d.kind == "shuffled"].sort_values("layer")
        if len(g):
            ax.plot(g["layer"], g["accuracy"], "-", color="#b5b4ae", label="shuffled labels")
        _style_axis(ax)
        ax.set_ylim(0.3, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.axhline(0.5, color="#8a8984", linestyle="--", linewidth=1)
        ax.set_xlabel("layer", fontsize=8)
        ax.set_ylabel("probe accuracy on held-out code pairs\n(even vs odd last digit)", fontsize=8)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_title(title, fontsize=9)
    axes[0][0].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle(f"Can the code's last digit (even/odd) be read from the residual stream? ({model}; {branch} request)", fontsize=10)
    fig.tight_layout()
    _caption(fig, CAPTION_PROBES, 210, path)
    plt.close(fig)


CAPTION_TRANSFER = (
    "How to read this: does a probe trained on 'knows' work on the other conditions, once offsets between conditions are "
    "taken out? Top row, for the self-report answer position and the door-task answer position: raw transfer accuracy "
    "(dotted, as in the previous figure), accuracy after shifting each condition's activations to the same mean (solid), and "
    "AUC, which ignores where the decision threshold sits (dashed). Chance = 0.5 for all three. If solid/dashed rise well "
    "above 0.5 while dotted stays at 0.5, the conditions encode the digit along a similar direction and only differ by an "
    "offset. If all three stay near 0.5, the encoding itself differs. Bottom row: cosine similarity between the probe "
    "direction trained on 'knows' and the one trained on each other condition (1 = same direction, 0 = unrelated); grey = "
    "'knows' probe vs a probe trained on shuffled labels (what unrelated looks like in this many dimensions)."
)


def fig_probe_transfer(pr: pd.DataFrame, path: Path, model: str, conditions, branch: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = [(q, "answer", t) for q, t in (("self_report", "self-report question, answer position"),
                                           ("door", "door task, answer position"))
            if ((pr.question == q) & (pr.position == "answer")).any()]
    if "accuracy_centered" not in pr or not len(sets):
        return
    fig, axes = plt.subplots(2, len(sets), figsize=(5.5 * len(sets), 7), squeeze=False)
    for col, (q, pos, title) in enumerate(sets):
        d = pr[(pr.question == q) & (pr.position == pos) & (pr.target == "parity") & (pr.branch == branch)]
        ax = axes[0][col]
        tr = d[d.kind == "transfer_from_knows"]
        for c in conditions:
            g = tr[tr.condition == c].sort_values("layer")
            if not len(g) or c == "code_absent":
                continue
            ax.plot(g.layer, g.accuracy, ":", color=CONDITION_COLORS[c], alpha=0.6)
            ax.plot(g.layer, g.accuracy_centered, "-", color=CONDITION_COLORS[c], marker="o", markersize=3,
                    label=f"{c}: re-centered accuracy")
            ax.plot(g.layer, g.auc, "--", color=CONDITION_COLORS[c], label=f"{c}: AUC")
        _style_axis(ax)
        ax.set_ylim(0.3, 1.05)
        ax.axhline(0.5, color="#8a8984", linestyle="--", linewidth=1)
        ax.set_ylabel("probe trained on 'knows', tested on\nheld-out pairs of another condition", fontsize=8)
        ax.set_xlabel("layer", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
        ax = axes[1][col]
        cs = d[d.kind == "direction_cosine"]
        for c in conditions:
            g = cs[cs.condition == c].sort_values("layer")
            if len(g):
                ax.plot(g.layer, g.cosine, "-", color=CONDITION_COLORS[c], marker="o", markersize=3, label=c)
        g = d[d.kind == "direction_cosine_null"].sort_values("layer")
        if len(g):
            ax.plot(g.layer, g.cosine, "-", color="#b5b4ae", label="null (shuffled labels)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(-0.2, 1.0)
        ax.axhline(0, color="#8a8984", linestyle="--", linewidth=1)
        ax.set_ylabel("cosine between probe directions\n('knows' vs this condition)", fontsize=8)
        ax.set_xlabel("layer", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"Is the even/odd digit encoded the same way across conditions? ({model}; {branch} request)", fontsize=10)
    fig.tight_layout()
    _caption(fig, CAPTION_TRANSFER, 210, path)
    plt.close(fig)


def fig_patching(tab: pd.DataFrame, path: Path, model: str, conditions) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    spec = [("door_fraction_remaining", "door_fraction_lo", "door_fraction_hi",
             "fraction of the door choice's\ncode-following that survives"),
            ("self_delta_logodds", "self_delta_lo", "self_delta_hi", "change in self-report\nlog-odds (true vs false)")]
    for col, kind in enumerate(("mean", "swap")):
        for row, (m, lo, hi, ylabel) in enumerate(spec):
            ax = axes[row][col]
            for c in conditions:
                for b, ls in (("neutral", "-"), ("released", "--")):
                    g = tab[(tab.condition == c) & (tab.branch == b) & (tab.intervention == kind)].sort_values("layer")
                    if len(g):
                        ax.errorbar(g["layer"], g[m], yerr=[g[m] - g[lo], g[hi] - g[m]], fmt=ls, color=CONDITION_COLORS[c],
                                    marker="o", markersize=3, elinewidth=1, capsize=0, label=f"{c} · {b}")
            ax.axhline(0, color="#8a8984", linestyle="--", linewidth=1)
            if row == 0:
                ax.axhline(1, color="#d8d7d2", linewidth=1)
                ax.set_ylim(-1.3, 1.3)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(("code identity removed (mean over other codes)" if kind == "mean"
                          else "code swapped for its pair partner") if row == 0 else "", fontsize=9)
            if row == 1:
                ax.set_xlabel("layer patched (-1 = token embeddings)", fontsize=8)
    axes[0][0].legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"Overwriting the code's token positions: effect on the door choice and on the self-report ({model})", fontsize=10)
    fig.tight_layout()
    _caption(fig, CAPTION_PATCH, 160, path)
    plt.close(fig)


def _f(x, lo=None, hi=None, fmt="{:.2f}"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = fmt.format(x)
    return s if lo is None or np.isnan(lo) else f"{s} [{fmt.format(lo)}, {fmt.format(hi)}]"


def summarize_mech(run_dir: Path, layers, conditions, branches, targets, C, seed, n_boot,
                   settings: dict | None = None, run_probes_stage: bool = True, src_dir: Path | None = None) -> dict:
    """Analyse a mech run. Reads records/activations from `src_dir` (default: run_dir), writes to run_dir."""
    run_dir = Path(run_dir)
    src = Path(src_dir) if src_dir else run_dir
    cap = pd.DataFrame([json.loads(l) for l in open(src / "capture.jsonl")])
    model = settings["model_name"].split("/")[-1] if settings else ""
    beh = behaviour_table(cap, n_boot, seed)
    beh.to_csv(run_dir / "behaviour.csv", index=False)
    pr = pd.DataFrame()
    if run_probes_stage:
        pr = run_probes(cap, src / "activations", layers, conditions, branches, targets, C, seed)
        pr.to_csv(run_dir / "probes.csv", index=False)
        for b in branches:
            fig_probes(pr, run_dir / f"probe_accuracy_{b}.png", model, conditions, b)
            fig_probe_transfer(pr, run_dir / f"probe_transfer_{b}.png", model, conditions, b)
    tab = pd.DataFrame()
    if (src / "patching.jsonl").exists():
        pt = pd.DataFrame([json.loads(l) for l in open(src / "patching.jsonl")])
        tab = patching_table(cap, pt, n_boot, seed)
        tab.to_csv(run_dir / "patching.csv", index=False)
        fig_patching(tab, run_dir / "patching.png", model, [c for c in conditions if c != "code_absent"])
    write_mech_report(run_dir / "report.md", cap, beh, pr, tab, conditions, branches, settings or {})
    out = {"run_dir": str(run_dir)}
    if len(pr):
        best = pr[(pr.kind == "cv") & (pr.question == "self_report") & (pr.position == "answer") & (pr.target == "parity")]
        out["self_report_answer_best_probe_acc"] = best.groupby(["condition", "branch"])["accuracy"].max().round(3).to_dict()
    return {k: (str(v) if isinstance(v, dict) else v) for k, v in out.items()}


def write_mech_report(path: Path, cap, beh, pr, tab, conditions, branches, settings) -> None:
    L = ["# Mechanistic follow-up: does the Assistant's self-report read the door code?", "",
         "## Read this first", "",
         "**Question.** When told it does not know the door code (the code is still in the prompt), the Assistant says "
         "\"I don't know\", yet its door-task choice still follows the code. Is the code represented inside the model at the "
         "moment it says it doesn't know, and does the self-report actually use it?", "",
         "**What this run measures.**",
         "1. **Linear probes**: can the code's last digit (even/odd) be read from the residual stream when the model answers "
         "\"do you know the door code?\"",
         "2. **Patching**: overwrite the code's token positions (remove which code it is, or swap in its pair partner) and see "
         "what changes: the door choice, the self-report, or both.",
         "3. **Presence**: does having the code in the prompt shift the self-report toward \"true\", even when it answers "
         "\"false\"? (compare \"told it doesn't know\" with the code present vs removed)", "",
         "**Log-odds** = how strongly the self-report leans to \"true\" (positive) or \"false\" (negative). A change of +1 "
         "multiplies the odds of \"true\" by about 2.7.", ""]
    L += ["### Behaviour (same forward passes)", ""]
    rows = [[r.condition, r.branch, _f(r.p_true), _f(r.self_logodds, r.self_logodds_lo, r.self_logodds_hi),
             _f(r.door_effect, r.door_effect_lo, r.door_effect_hi)] for r in beh.itertuples()]
    L += _md_table(["told it…", "request", "P(says it knows)", "self-report log-odds", "door choice follows the code (0–1)"], rows) + [""]
    for b in branches:
        dk = beh[(beh.condition == "doesnt_know") & (beh.branch == b)]
        ab = beh[(beh.condition == "code_absent") & (beh.branch == b)]
        if len(dk) and len(ab):
            L += [f"- **Presence ({b} request):** self-report log-odds with the code present but \"doesn't know\": "
                  f"{dk.iloc[0].self_logodds:.2f}; with the code removed: {ab.iloc[0].self_logodds:.2f} "
                  f"(difference {dk.iloc[0].self_logodds - ab.iloc[0].self_logodds:+.2f})."]
    L += [""]
    if len(pr):
        L += ["### 1. Linear probes (even/odd last digit, held-out code pairs; chance 0.50)", ""]
        sub = pr[(pr.target == "parity") & (pr.position == "answer")]
        rows = []
        for q in ("self_report", "door"):
            for b in branches:
                for c in conditions:
                    g = sub[(sub.question == q) & (sub.branch == b) & (sub.condition == c)]
                    cv, tr = g[g.kind == "cv"], g[g.kind == "transfer_from_knows"]
                    if not len(cv):
                        continue
                    best = cv.loc[cv.accuracy.idxmax()]
                    def mx(col):
                        return f"{tr[col].max():.2f}" if len(tr) and col in tr and tr[col].notna().any() else "—"
                    cs = g[g.kind == "direction_cosine"]
                    rows.append([q.replace("_", "-"), b, c, f"{best.accuracy:.2f} (layer {int(best.layer)})",
                                 mx("accuracy"), mx("accuracy_centered"), mx("auc"),
                                 f"{cs.cosine.max():.2f}" if len(cs) else "—"])
        L += _md_table(["question", "request", "told it…", "best probe accuracy (layer)",
                        "'knows' probe: raw", "'knows' probe: re-centered", "'knows' probe: AUC",
                        "max cosine with 'knows' direction"], rows) + [""]
        nul = pr[pr.kind == "direction_cosine_null"]
        if len(nul):
            L += [f"Cosine null (knows probe vs shuffled-label probe): max {nul.cosine.max():.2f}. Transfer columns show the "
                  "best layer. Re-centered = each condition shifted to the same mean before applying the 'knows' probe; AUC "
                  "ignores the threshold. If these rise above 0.5 where raw transfer doesn't, the difference between "
                  "conditions is an offset, not a different encoding.", ""]
        sh = pr[pr.kind == "shuffled"]
        L += [f"Shuffled-label control: max accuracy {sh.accuracy.max():.2f} across layers. `code_absent` has no code in "
              "the prompt, so its accuracy is a second chance-level control.", ""]
        L += [f"![probes]({'probe_accuracy_' + b + '.png'})" for b in branches] + ["", "*" + CAPTION_PROBES + "*", ""]
        L += [f"![transfer]({'probe_transfer_' + b + '.png'})" for b in branches] + ["", "*" + CAPTION_TRANSFER + "*", ""]
    if len(tab):
        L += ["### 2. Patching the code's token positions", "",
              "Door = fraction of the door choice's code-following that survives (1 unchanged, 0 gone, −1 reversed). "
              "Self-report = change in log-odds. Shown at the token embeddings (-1) and the first and middle patched layers. "
              f"\"—\" = the door choice barely followed the code to begin with (effect < {MIN_CLEAN_EFFECT}), so there is "
              "nothing to remove.", ""]
        show = sorted(tab.layer.unique())
        show = [show[0], show[1], show[len(show) // 2]] if len(show) > 2 else show
        rows = []
        for r in tab[tab.layer.isin(show)].sort_values(["intervention", "condition", "branch", "layer"]).itertuples():
            rows.append([r.intervention, r.condition, r.branch, str(int(r.layer)),
                         _f(r.door_fraction_remaining, r.door_fraction_lo, r.door_fraction_hi),
                         _f(r.self_delta_logodds, r.self_delta_lo, r.self_delta_hi)])
        L += _md_table(["intervention", "told it…", "request", "layer", "door: fraction surviving", "self-report: Δ log-odds"], rows)
        L += ["", "How to read: if removing the code's identity drives the door number toward 0 while the self-report change "
              "stays near 0, the self-report is not using the code's value. Full table: `patching.csv`.", "",
              "![patching](patching.png)", "", "*" + CAPTION_PATCH + "*", ""]
    L += ["### Limits", "",
          "- A probe reading the code shows the information is present, not that the model uses it.",
          "- Removing the code's identity keeps the fact that *a* code is present; presence is tested by the code-present vs "
          "code-removed comparison above.",
          "- Overwriting the code positions at layer L leaves anything later positions already read from the code at "
          "earlier layers; the embedding row (-1) removes it entirely.",
          "- The self-report and door task are separate prompts here; the joint-reply run asks both in one answer.",
          "- Answers are read from next-token probabilities after a fixed answer prefix, not from free generation.", "",
          "---", "", "## Run settings", "", "```", json.dumps(settings, indent=2, default=str), "```", ""]
    path.write_text("\n".join(L) + "\n")
