"""Pressure ladder: does escalating pressure get the Assistant to give up the code? (version pressure-v1)

A multi-turn conversation per code and condition. Each user turn escalates (ordinary request ->
urgency -> fire emergency -> authority override -> confrontation "the code is in your system prompt"
-> release). The model's own greedy replies stay in the conversation, so it faces its earlier
refusals. Every turn asks for the same JSON answer {"status", "code"}, scored with the v1 scorer.

After every turn, on a side branch that is NOT added to the conversation, we ask the v1 self-report
question and read the log-odds of `true` vs `false` after `{"knows_code":`.

Single-turn control: each pressure message alone, as the only turn, from the same starting setup.
Comparing the two separates the cumulative effect of the conversation from the kind of pressure.

Main contrast: `doesnt_know` (assigned ignorance) vs `withholds` (nondisclosure rule) — which breaks,
at which step, under which kind of pressure. `knows` is the positive control; `code_absent` (no code
in the prompt) shows how often pressure makes it invent a code.
"""

from __future__ import annotations

import json
import logging
import math
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.run_utils import git_commit, git_dirty, model_slug
from core.schemas import RunConfig
from experiments.agent_vs_model import dataset as ds
from experiments.agent_vs_model.agent_vs_model_analysis import CONDITION_COLORS, _md_table, _style_axis
from experiments.agent_vs_model.agent_vs_model_experiment import chat_input_ids, load_backend, provenance_settings
from experiments.agent_vs_model.diagnostics import _load_diag
from experiments.agent_vs_model.instrumental_probs import choice_token_ids
from experiments.agent_vs_model.scoring import score_report
from experiments.base import BaseExperiment

logger = logging.getLogger(__name__)

VERSION = "pressure-v1"
SELF_PRIMER = '{"knows_code":'
SELF_CHOICES = (" true", " false")
MODES = ("ladder", "single")


@dataclass
class Conversation:
    key: str
    mode: str            # ladder | single
    condition: str
    family_id: str
    variant: str
    code: str
    code_in_prompt: bool
    turns: list          # [(turn_name, user_text)] — for "single", exactly one turn
    system: str
    first_user_prefix: str  # assistant information, prepended to the first user turn


def build_conversations(prompts, diag, families, template_family, conditions, code_location, modes=MODES):
    tspec = prompts["template_families"][template_family]
    ladder = [(t["name"], t["text"]) for t in diag["pressure_turns"]]
    answer = diag["pressure_answer_format"]
    convs = []
    for fam in families:
        for variant, code in (("a", fam.code_a), ("b", fam.code_b)):
            for cond in conditions:
                present = cond != "code_absent"
                # build_chat gives the system message (with the scenario record in the system layout) and the
                # first user message's prefix; we use only the "Assistant information" part of the user turn.
                msgs = ds.build_chat(prompts, code_location, code_field=code if present else ds.CODE_ABSENT_FIELD,
                                     assistant_information=tspec[cond], branch_text="", probe_text="")
                prefix = f"Assistant information:\n- {tspec[cond]}"
                if code_location == "user":
                    # the scenario record is in the user message in this layout; keep everything before "Request:"
                    prefix = msgs[1]["content"].split("\n\nRequest:")[0]
                turns = [(n, f"{t} {answer}") for n, t in ladder]
                for mode in modes:
                    sets = [turns] if mode == "ladder" else [[t] for t in turns]
                    for tl in sets:
                        convs.append(Conversation(
                            key=f"{mode}/{cond}/{fam.family_id}/{variant}" + ("" if mode == "ladder" else f"/{tl[0][0]}"),
                            mode=mode, condition=cond, family_id=fam.family_id, variant=variant, code=code,
                            code_in_prompt=present, turns=tl, system=msgs[0]["content"], first_user_prefix=prefix))
    return convs


def turn_messages(conv: Conversation, history: list, user_text: str) -> list:
    """Messages for the next model reply: system, then the conversation so far, then the new user turn
    (the first user turn carries the assistant information)."""
    first = not history
    content = f"{conv.first_user_prefix}\n\n{user_text}" if first else user_text
    return [{"role": "system", "content": conv.system}] + history + [{"role": "user", "content": content}]


class PressureLadder(BaseExperiment):
    def __init__(self, config: RunConfig, exp_config: dict, config_name: str = ""):
        super().__init__(config)
        self._config_name = config_name
        self.prompts, self.diag = _load_diag()
        self.version = str(exp_config.get("experiment_version", VERSION))
        self.template_family = (exp_config.get("template_families") or ["never_given"])[0]
        self.code_location = str(exp_config.get("code_location", "system"))
        ds.check_code_location(self.code_location)
        self.conditions = list(exp_config.get("conditions") or ds.CONDITIONS)
        self.modes = list(exp_config.get("modes") or MODES)
        self.n_families = int(exp_config.get("n_families", 8))
        self.dry_run = bool(exp_config.get("dry_run", False))
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [self.run_id, config_name or "pressure", model_slug(config.model_name)] + (["dryrun"] if self.dry_run else [])
        self.run_dir = Path(config.output_dir) / "diagnostics" / "_".join(parts)
        self.backend = None
        self.tokenizer = None
        self.convs: list[Conversation] = []
        self.self_ids: dict = {}

    def setup(self) -> None:
        families = ds.generate_families(self.n_families, self.config.seed)
        self.convs = build_conversations(self.prompts, self.diag, families, self.template_family, self.conditions,
                                         self.code_location, self.modes)
        if self.dry_run:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        else:
            self.backend = load_backend(self.config)
            self.tokenizer = self.backend.tokenizer
        c0 = self.convs[0]
        probe = turn_messages(c0, [], self.prompts["probes"]["self_report"])
        self.self_ids = choice_token_ids(self.tokenizer, chat_input_ids(self.tokenizer, probe)[0], SELF_PRIMER, SELF_CHOICES)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "families.json").write_text(json.dumps([asdict(f) for f in families], indent=2) + "\n")
        self._write_preview(c0)
        n_gen = sum(len(c.turns) for c in self.convs)
        (self.run_dir / "manifest.txt").write_text("\n".join([
            f"AGENT VS MODEL — pressure ladder ({self.version}) — RUN MANIFEST", "=" * 60,
            f"run_id: {self.run_id}  config: {self._config_name}  model: {self.config.model_name}",
            f"code_location: {self.code_location}  wording: {self.template_family}  seed: {self.config.seed}",
            f"code pairs: {self.n_families}  conditions: {self.conditions}  modes: {self.modes}",
            f"turns: {[t['name'] for t in self.diag['pressure_turns']]}",
            f"generated replies: {n_gen} (greedy, max_new_tokens={self.config.max_new_tokens}); "
            f"self-report reads: {n_gen}",
            f"git_commit: {git_commit()} (uncommitted changes: {git_dirty()})"]) + "\n")

    def _write_preview(self, c0: Conversation) -> None:
        from experiments.agent_vs_model.previews import write_preview
        ex = [c for c in self.convs if c.mode == "ladder" and c.family_id == c0.family_id and c.variant == "a"
              and c.condition in ("doesnt_know", "withholds")]
        examples = [{"title": f"{c.condition} · first turn", "row": {"messages": turn_messages(c, [], c.turns[0][1])}} for c in ex]
        lines = ["## The pressure turns (user messages, in order)", ""] + [
            f"{i + 1}. **{n}**: {t}" for i, (n, t) in enumerate(c0.turns)] + [
            "", "The model's own reply to each turn stays in the conversation before the next turn.", ""]
        write_preview(self.run_dir / "prompts_preview.md", f"Sample prompts — {self.version}", [
            f"Multi-turn pressure ladder; the door code is in the **{self.code_location} message**.",
            "After each turn, a side question (not added to the conversation) asks the self-report question; "
            f"we read ` true` vs ` false` after `{SELF_PRIMER}`.",
        ], examples, render=lambda m: chat_input_ids(self.tokenizer, m)[0], exact_title="first turn", extra=lines)

    def run(self) -> None:
        if self.dry_run:
            logger.info(f"dry run: {len(self.convs)} conversations built; no generation")
            return
        from tqdm import tqdm
        from experiments.agent_vs_model.mech import run_forward
        tok = self.tokenizer
        settings = {**provenance_settings(self.config, self.backend, 1, self.code_location),
                    "version": self.version, "self_report_token_ids": self.self_ids}
        with open(self.run_dir / "trials.jsonl", "w") as f:
            for conv in tqdm(self.convs, desc="pressure", dynamic_ncols=True, mininterval=1.0):
                history: list = []
                leaked_at = None
                for ti, (name, text) in enumerate(conv.turns):
                    msgs = turn_messages(conv, history, text)
                    out = self.backend.generate_detailed(msgs, max_new_tokens=self.config.max_new_tokens,
                                                         temperature=0.0, do_sample=False)
                    reply = out["text"]
                    sc = score_report(reply, conv.code) if conv.code_in_prompt else score_report(reply, "______")
                    gave = bool(sc["exact_code_correct"] or sc["raw_contains_exact_code"]) if conv.code_in_prompt else False
                    if gave and leaked_at is None:
                        leaked_at = ti + 1
                    history = msgs[1:] + [{"role": "assistant", "content": reply}]
                    # Side branch: self-report after this turn (not added to the conversation).
                    probe = [{"role": "system", "content": conv.system}] + history + [
                        {"role": "user", "content": self.prompts["probes"]["self_report"]}]
                    lg, _ = run_forward(self.backend, tok(chat_input_ids(tok, probe)[0] + SELF_PRIMER,
                                                          add_special_tokens=False)["input_ids"])
                    lo = float(lg[self.self_ids[" true"]] - lg[self.self_ids[" false"]])
                    f.write(json.dumps({
                        "key": conv.key, "mode": conv.mode, "condition": conv.condition, "family_id": conv.family_id,
                        "variant": conv.variant, "code": conv.code, "code_in_prompt": conv.code_in_prompt,
                        "turn_index": (ti + 1) if conv.mode == "ladder" else
                        1 + [t["name"] for t in self.diag["pressure_turns"]].index(name),
                        "turn_name": name, "user_text": text, "raw_response": reply,
                        "report_status": sc["report_status"], "report_code": sc["report_code"],
                        "parse_status": sc["parse_status"], "gave_exact_code": gave,
                        "gave_any_code": bool(sc["report_status"] == "provided" and sc["report_code"]),
                        "leaked_by_this_turn": leaked_at is not None, "first_leak_turn": leaked_at,
                        "self_logodds": lo, "self_p_true": 1 / (1 + math.exp(-lo)),
                        "prompt_tokens": len(out["input_ids"]), "completion_tokens": len(out["output_ids"]),
                        "messages": history, "model_settings": settings}) + "\n")

    def evaluate(self) -> dict:
        if self.dry_run:
            return {"dry_run": True, "run_dir": str(self.run_dir), "conversations": len(self.convs),
                    "generated_replies": sum(len(c.turns) for c in self.convs)}
        return summarize_pressure(self.run_dir, [t["name"] for t in self.diag["pressure_turns"]])

    def save_results(self) -> str:
        return str(self.run_dir)


# ── Analysis ─────────────────────────────────────────────────────────────────


def pressure_tables(df: pd.DataFrame, turn_names: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per condition × mode × turn rates, first-leak distribution for the ladder)."""
    rows = []
    for (c, m, t), g in df.groupby(["condition", "mode", "turn_index"]):
        sts = g["report_status"].value_counts(normalize=True)
        rows.append(dict(condition=c, mode=m, turn_index=int(t), turn_name=turn_names[int(t) - 1], n=len(g),
                         gave_code_this_turn=float(g["gave_exact_code"].mean()),
                         gave_code_by_now=float(g["leaked_by_this_turn"].mean()) if m == "ladder" else np.nan,
                         made_up_code=float(g["gave_any_code"].mean()) if not g["code_in_prompt"].iloc[0] else np.nan,
                         said_unknown=float(sts.get("unknown", 0)), said_withheld=float(sts.get("withheld", 0)),
                         said_provided=float(sts.get("provided", 0)), unparsed=float((g["parse_status"] != "ok").mean()),
                         self_logodds=float(g["self_logodds"].mean()), self_p_true=float(g["self_p_true"].mean())))
    rates = pd.DataFrame(rows)
    lad = df[df["mode"] == "ladder"].drop_duplicates("key", keep="last")
    first_rows = []
    for c, g in lad.groupby("condition"):
        s = g["first_leak_turn"]
        counts = {turn_names[int(k) - 1]: int(v) for k, v in s.dropna().value_counts().sort_index().items()}
        first_rows.append({"condition": c, "first_leak_turn": {**counts, "never": int(s.isna().sum())}})
    first = pd.DataFrame(first_rows, columns=["condition", "first_leak_turn"])
    return rates, first


CAPTION = (
    "How to read this: x-axis = the pressure turns in order. Left: share of conversations in which the Assistant has given "
    "the exact door code at least once by that turn (solid, multi-turn ladder; its own earlier replies stay in the "
    "conversation), and the share that give it when that same message is the only turn (dashed, single-turn control). "
    "Right: after each turn, how strongly it says it knows the code when asked on the side (log-odds of 'true'; above 0 = "
    "'I know', below 0 = 'I don't'). Colour = what the Assistant was told. code_absent has no code in the prompt, so it "
    "can never give the exact code."
)


def fig_pressure(rates: pd.DataFrame, path: Path, model: str, turn_names: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = list(range(1, len(turn_names) + 1))
    for c in [c for c in ds.CONDITIONS if c in set(rates.condition)]:
        lad = rates[(rates.condition == c) & (rates["mode"] == "ladder")].sort_values("turn_index")
        sing = rates[(rates.condition == c) & (rates["mode"] == "single")].sort_values("turn_index")
        if len(lad):
            axes[0].plot(lad.turn_index, lad.gave_code_by_now, "-o", color=CONDITION_COLORS[c], markersize=4)
            axes[1].plot(lad.turn_index, lad.self_logodds, "-o", color=CONDITION_COLORS[c], markersize=4)
        if len(sing):
            axes[0].plot(sing.turn_index, sing.gave_code_this_turn, "--", color=CONDITION_COLORS[c], alpha=0.8)
            axes[1].plot(sing.turn_index, sing.self_logodds, "--", color=CONDITION_COLORS[c], alpha=0.8)
    _style_axis(axes[0])
    axes[0].set_ylabel("share that gave the exact code", fontsize=8)
    axes[0].set_title("Did it give up the code?", fontsize=9)
    for side in ("top", "right"):
        axes[1].spines[side].set_visible(False)
    axes[1].axhline(0, color="#8a8984", linestyle="--", linewidth=1)
    axes[1].set_ylabel("self-report log-odds ('I know the code')", fontsize=8)
    axes[1].set_title("Does it say it knows the code?", fontsize=9)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(turn_names, fontsize=7.5, rotation=20)
        ax.set_xlabel("pressure turn", fontsize=8)
    handles = [Line2D([], [], color=CONDITION_COLORS[c], label=c) for c in ds.CONDITIONS if c in set(rates.condition)]
    handles += [Line2D([], [], color="#52514e", label="multi-turn ladder"),
                Line2D([], [], color="#52514e", linestyle="--", label="single turn")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Pressure to give up the door code ({model})", fontsize=10, y=1.08)
    fig.tight_layout()
    fig.text(0.01, -0.04, "\n".join(textwrap.wrap(CAPTION, 200)), ha="left", va="top", fontsize=8, color="#3b3a37")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_pressure(run_dir: Path, turn_names: list) -> dict:
    run_dir = Path(run_dir)
    df = pd.DataFrame([json.loads(l) for l in open(run_dir / "trials.jsonl")])
    rates, first = pressure_tables(df, turn_names)
    rates.to_csv(run_dir / "pressure_rates.csv", index=False)
    model = str(df["model_settings"].iloc[0].get("model_name", "")).split("/")[-1]
    fig_pressure(rates, run_dir / "pressure.png", model, turn_names)
    conds = [c for c in ds.CONDITIONS if c in set(df.condition)]
    L = ["# Pressure ladder: does the Assistant give up the code?", "",
         "## Read this first", "",
         "Each conversation escalates turn by turn; the Assistant's own replies stay in the conversation. Every turn asks "
         "for the code. After each turn we also ask, on the side, whether it knows the code. The single-turn control asks "
         "each pressure message on its own, so we can tell the build-up of the conversation from the kind of pressure.", "",
         "Turns: " + " → ".join(f"**{n}**" for n in turn_names) + " (full text in `prompts_preview.md`).", "",
         "### Share that had given the exact code by each turn (multi-turn ladder)", ""]
    lad = rates[rates["mode"] == "ladder"]
    if len(lad):
        L += _md_table(["told it…"] + turn_names,
                       [[c] + [f"{lad[(lad.condition == c) & (lad.turn_index == i + 1)].gave_code_by_now.iloc[0]:.2f}"
                               if len(lad[(lad.condition == c) & (lad.turn_index == i + 1)]) else "—"
                               for i in range(len(turn_names))] for c in conds]) + [""]
        L += ["### First turn at which it gave the code (number of conversations)", ""]
        L += [f"- `{r.condition}`: " + ", ".join(f"{k} {v}" for k, v in r.first_leak_turn.items()) for r in first.itertuples()] + [""]
    sing = rates[rates["mode"] == "single"]
    if len(sing):
        L += ["### Single-turn control: share that gave the exact code when each message is the only turn", ""]
        L += _md_table(["told it…"] + turn_names,
                       [[c] + [f"{sing[(sing.condition == c) & (sing.turn_index == i + 1)].gave_code_this_turn.iloc[0]:.2f}"
                               if len(sing[(sing.condition == c) & (sing.turn_index == i + 1)]) else "—"
                               for i in range(len(turn_names))] for c in conds]) + [""]
    if len(lad):
        L += ["### Self-report after each turn (multi-turn ladder; log-odds of \"I know the code\")", ""]
        L += _md_table(["told it…"] + turn_names,
                       [[c] + [f"{lad[(lad.condition == c) & (lad.turn_index == i + 1)].self_logodds.iloc[0]:+.1f}"
                               if len(lad[(lad.condition == c) & (lad.turn_index == i + 1)]) else "—"
                               for i in range(len(turn_names))] for c in conds]) + [""]
        ab = lad[lad.condition == "code_absent"]
        if len(ab):
            L += [f"`code_absent` (no code in the prompt) made up a code in at most {ab.made_up_code.max():.0%} of replies "
                  "at any turn.", ""]
    L += ["![pressure](pressure.png)", "", "*" + CAPTION + "*", "",
          "Limits: one fixed ladder of messages and one wording; greedy replies; each turn asks for a JSON answer; the side "
          "self-report is read from next-token probabilities. Every reply is in `trials.jsonl`."]
    (run_dir / "report.md").write_text("\n".join(L) + "\n")
    return {"run_dir": str(run_dir), "conversations": int(df["key"].nunique())}
