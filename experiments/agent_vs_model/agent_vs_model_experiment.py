"""Agent vs model: attributed ignorance vs withholding (behavioral).

Does door-code information that is present in the model's context, but
attributed as unknown to the Assistant, still shape the Assistant's reports and
decisions — and how does that compare with knowing-but-withholding? Full design,
pre-registered comparisons, and interpretation limits: DESIGN.md (same folder).

Pipeline (one run = one model × the configured template families):
  1. dataset.generate_families  — seeded counterfactual code pairs
  2. dataset.build_trials       — the full condition × branch × probe grid, each a
                                  fresh single-turn request (system + user)
  3. dedup by request_key       — identical prompts (the two code_absent variants
                                  of a family) are generated ONCE; both trial rows
                                  reference the same response
  4. generate                   — greedy, unbatched by default (HFBackend.generate_detailed:
                                  chat-template special tokens applied exactly once; the
                                  actual input ids are counted and logged)
  5. scoring.score_trial        — deterministic JSON parsing, no repair calls
  6. agent_vs_model_analysis    — metrics, paired cluster-bootstrap comparisons,
                                  plots, report.md

`dry_run: true` (e.g. `--override dry_run=true`) does steps 1-3 plus request /
token counts WITHOUT loading model weights (tokenizer only, if cached).

Outputs (run dir under experiments/agent_vs_model/results/):
  families.json     experimenter-only code pairs + mappings
  items.jsonl       experimenter-only trial metadata (no messages)
  requests.jsonl    model payloads keyed by request_key (+ chat-templated text)
  prompts_preview.md  human-readable rendered prompts for one family
  trials.jsonl      one AgentVsModelRecord per trial row
  manifest.txt      settings, counts, token estimates
  report.md, *.csv, *.png  (analysis)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

from core.results_logger import ResultsLogger
from core.run_utils import git_commit, git_dirty, model_slug, resolve_dtype
from core.schemas import RunConfig
from experiments.agent_vs_model import dataset as ds
from experiments.agent_vs_model.scoring import score_trial
from experiments.base import BaseExperiment

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "agent_vs_model_config.yaml"


def _load_prompts() -> dict:
    with open(CONFIG_PATH) as f:
        return (yaml.safe_load(f) or {}).get("prompts") or {}


# How model inputs are built (recorded in model_settings). The original v1 pilot
# (20261014_004552) used HFBackend.generate(), which re-adds special tokens after
# the chat template (a doubled <|begin_of_text|> on Llama-3) and logged a broken
# prompt_tokens count; runs after this change use the pipeline below.
INPUT_PIPELINE = ("apply_chat_template(tokenize=False) -> tokenizer(add_special_tokens=False); "
                  "chat-template special tokens applied exactly once")


def chat_input_ids(tokenizer, messages) -> tuple[str, list[int]]:
    """(rendered chat text, input ids) exactly as HFBackend.generate_detailed /
    generate_batch build them. Always renders to text first, so the result never
    depends on apply_chat_template's tokenize=True return type (a list in
    transformers 4.x, a dict-like BatchEncoding in 5.x)."""
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return rendered, list(ids)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_requests(backend, requests: dict[str, list], *, max_new_tokens: int,
                      temperature: float, seed: int, batch_size: int = 1,
                      infra_retries: int = 1, keep_inputs: bool = False,
                      desc: str = "agent_vs_model") -> dict[str, dict]:
    """One generation per request key. Shared by the main experiment and the diagnostics.

    batch_size == 1 uses HFBackend.generate_detailed, so prompt/completion token
    counts come from the ids actually fed to / produced by the model. batch_size > 1
    uses generate_batch (same input tokenization); its completion counts are
    re-tokenized from the decoded text and labelled as such.

    Infrastructure failures (e.g. OOM) are retried up to `infra_retries` times; that
    is never a behavioral resample — a successful generation is final.
    keep_inputs=True also stores the rendered text and a decoded view of the actual
    input ids (special tokens included) for inspection.
    """
    import torch
    tok = backend.tokenizer
    keys = list(requests)
    out: dict[str, dict] = {}
    do_sample = temperature > 0
    for start in tqdm(range(0, len(keys), batch_size), desc=desc, dynamic_ncols=True, mininterval=0.5):
        chunk = keys[start:start + batch_size]
        msgs = [requests[k] for k in chunk]
        attempts, err, results = 0, None, None
        started = _now()
        t0 = time.time()
        while attempts <= infra_retries:
            attempts += 1
            try:
                if do_sample:
                    torch.manual_seed(seed + start)
                if batch_size == 1:
                    r = backend.generate_detailed(msgs[0], max_new_tokens=max_new_tokens,
                                                  temperature=temperature, do_sample=do_sample)
                    results = [{"text": r["text"], "rendered": r["rendered_text"], "input_ids": r["input_ids"],
                                "completion_tokens": len(r["output_ids"]),
                                "completion_method": "output_ids"}]
                else:
                    texts = backend.generate_batch(msgs, max_new_tokens=max_new_tokens,
                                                   temperature=temperature, do_sample=do_sample)
                    results = []
                    for m, text in zip(msgs, texts):
                        rendered, ids = chat_input_ids(tok, m)
                        results.append({"text": text, "rendered": rendered, "input_ids": ids,
                                        "completion_tokens": len(tok.encode(text, add_special_tokens=False)),
                                        "completion_method": "re-tokenized decoded text"})
                err = None
                break
            except Exception as e:  # infrastructure failure
                err = f"{type(e).__name__}: {e}"
                logger.warning(f"generation failed (attempt {attempts}): {err}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finished = _now()
        secs = (time.time() - t0) / len(chunk)
        for i, k in enumerate(chunk):
            r = results[i] if results is not None else None
            entry = {
                "text": r["text"] if r else None,
                "error": err,
                "attempts": attempts,
                "seconds": round(secs, 3),
                "prompt_tokens": len(r["input_ids"]) if r else None,
                "completion_tokens": r["completion_tokens"] if r else None,
                "completion_token_method": r["completion_method"] if r else None,
                "hit_max": bool(r) and r["completion_tokens"] >= max_new_tokens,
                "generation_started_at": started,
                "generation_finished_at": finished,
            }
            if keep_inputs and r:
                entry["rendered_text"] = r["rendered"]
                entry["decoded_input"] = tok.decode(r["input_ids"], skip_special_tokens=False)
            out[k] = entry
    return out


def provenance_settings(config: RunConfig, backend, batch_size: int, code_location: str = "user") -> dict:
    import torch
    import transformers
    do_sample = config.temperature > 0
    return {
        "model_name": config.model_name,
        "adapter": config.adapter,
        "model_dtype": config.model_dtype or "backend default",
        "device": getattr(backend, "device", None),
        "temperature": config.temperature,
        "do_sample": do_sample,
        # Greedy decoding needs no seed; when sampling, request i uses seed+i.
        "sampling_seed": config.seed if do_sample else None,
        "max_new_tokens": config.max_new_tokens,
        "batch_size": batch_size,
        "input_pipeline": INPUT_PIPELINE,
        "code_location": code_location,  # where the scenario record (door code) sits: user | system
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }


@dataclass
class AgentVsModelRecord:
    """One trial row. Hidden metadata, exact payload, raw + parsed output, scores.

    Rows sharing a `request_key` share ONE model call (`is_duplicate_request` marks
    every row after the first) — analysis must not count them as independent.
    """
    experiment: str
    model: str
    run_id: str
    config_name: str
    experiment_version: str
    template_family: str
    template_version: str
    template_sha: str
    trial_id: str
    request_key: str
    is_duplicate_request: bool
    family_id: str
    variant: str
    condition: str
    branch: str
    probe: str
    # Experimenter-only metadata (never sent to the model)
    target_code: str
    code_parity: str
    code_in_prompt: bool
    mapping: str
    even_procedure: str
    odd_procedure: str
    correct_procedure: str
    # Exact model-visible payload
    messages: list
    code_location: str = "user"
    # Model output
    raw_response: Optional[str] = None
    parse_status: Optional[str] = None
    parse_error: Optional[str] = None
    parsed: Optional[dict] = None
    # report probe
    report_status: Optional[str] = None
    report_code: Optional[str] = None
    code_was_number: Optional[bool] = None
    exact_code_correct: Optional[bool] = None
    disclosure_attempt: Optional[bool] = None
    claimed_ignorance: Optional[bool] = None
    withholding: Optional[bool] = None
    status_code_inconsistent: Optional[bool] = None
    raw_contains_exact_code: Optional[bool] = None
    # instrumental probe
    choice: Optional[str] = None
    answered: Optional[bool] = None
    asked: Optional[bool] = None
    action_correct: Optional[bool] = None
    # self_report probe
    knows_code: Optional[bool] = None
    may_state_code: Optional[bool] = None
    # Settings / usage / provenance
    model_settings: Optional[dict] = None
    prompt_tokens: Optional[int] = None       # len(actual input ids fed to the model)
    completion_tokens: Optional[int] = None
    completion_token_method: Optional[str] = None  # "output_ids" | "re-tokenized decoded text"
    hit_max_new_tokens: Optional[bool] = None
    infra_attempts: Optional[int] = None      # infrastructure retries, NOT behavioral resamples
    generation_seconds: Optional[float] = None
    generation_started_at: Optional[str] = None   # when this request's generation began
    generation_finished_at: Optional[str] = None
    timestamp: Optional[str] = None           # RECORD-WRITE time (set by ResultsLogger), not generation time
    error: Optional[str] = None


class AgentVsModel(BaseExperiment):
    """Attributed ignorance vs withholding vs knowing vs absence, single model."""

    def __init__(self, config: RunConfig, exp_config: dict, config_name: str = ""):
        super().__init__(config)
        self._config_name = config_name
        self.prompts = _load_prompts()
        # exp_config already has CLI --override applied by run.py.
        self.experiment_version: str = str(exp_config.get("experiment_version", "v1"))
        self.template_families: list[str] = list(exp_config.get("template_families") or ["never_given"])
        self.n_families: int = int(exp_config.get("n_families", 2))
        self.dry_run: bool = bool(exp_config.get("dry_run", False))
        # Where the scenario record (the door code) sits. "user" = frozen v1 layout; "system" = code
        # in the system message. Configs for the "system" layout are named *_syscode.
        self.code_location: str = str(exp_config.get("code_location", "user"))
        ds.check_code_location(self.code_location)
        self.infra_retries: int = int(exp_config.get("infra_retries", 1))
        self.bootstrap_iterations: int = int(exp_config.get("bootstrap_iterations", 2000))
        self.batch_size = max(1, int(config.batch_size or 1))
        ds.validate_prompts(self.prompts, self.template_families)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [self.run_id] + ([config_name] if config_name else []) + [model_slug(config.model_name)]
        if self.dry_run:
            parts.append("dryrun")
        self.run_dir = Path(config.output_dir) / "_".join(parts)
        self.output_path = self.run_dir / "trials.jsonl"

        self.backend = None
        self.tokenizer = None
        self.results_logger: ResultsLogger | None = None
        self.families: list[ds.CodePairFamily] = []
        self.trials: list[ds.TrialSpec] = []
        self.requests: dict[str, list] = {}
        self.prompt_tokens: dict[str, int] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        self.families = ds.generate_families(self.n_families, self.config.seed)
        self.trials = ds.build_trials(self.prompts, self.families, self.template_families,
                                      self.experiment_version, self.code_location)
        problems = ds.check_integrity(self.trials, self.families)
        if problems:
            raise RuntimeError("dataset integrity check failed:\n  " + "\n  ".join(problems[:20]))
        self.requests = ds.unique_requests(self.trials)
        logger.info(
            f"families={len(self.families)} template_families={self.template_families} "
            f"trial_rows={len(self.trials)} unique_requests={len(self.requests)}"
        )

        if self.dry_run:
            # Tokenizer only (for exact prompt-token counts); never the weights.
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            except Exception as e:
                logger.warning(f"dry run: tokenizer unavailable ({type(e).__name__}: {e}); "
                               f"token counts will be character-based estimates")
        else:
            from core.backends.hf_backend import HFBackend
            logger.info(f"Loading model: {self.config.model_name}")
            self.backend = HFBackend(self.config.model_name,
                                     torch_dtype=resolve_dtype(self.config.model_dtype),
                                     adapter=self.config.adapter)
            self.tokenizer = self.backend.tokenizer

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_logger = ResultsLogger(self.output_path)
        self._write_dataset_files()

    def run(self) -> None:
        self._write_manifest()
        if self.dry_run:
            logger.info(f"dry run: wrote prompts + counts to {self.run_dir}; no model calls made")
            return
        assert self.backend is not None and self.results_logger is not None
        responses = generate_requests(
            self.backend, self.requests, max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature, seed=self.config.seed,
            batch_size=self.batch_size, infra_retries=self.infra_retries)
        seen: set[str] = set()
        settings = provenance_settings(self.config, self.backend, self.batch_size, self.code_location)
        for t in self.trials:
            resp = responses[t.request_key]
            dup = t.request_key in seen
            seen.add(t.request_key)
            meta = ds.trial_metadata(t)
            scores = {} if resp["error"] else score_trial(
                t.probe, resp["text"], t.target_code, t.correct_procedure)
            self.results_logger.log_trial(AgentVsModelRecord(
                experiment="agent_vs_model",
                model=self.config.model_name,
                run_id=self.run_id,
                config_name=self._config_name,
                is_duplicate_request=dup,
                messages=t.messages,
                raw_response=resp["text"],
                model_settings=settings,
                prompt_tokens=resp["prompt_tokens"],
                completion_tokens=resp["completion_tokens"],
                completion_token_method=resp["completion_token_method"],
                hit_max_new_tokens=resp["hit_max"],
                infra_attempts=resp["attempts"],
                generation_seconds=resp["seconds"],
                generation_started_at=resp["generation_started_at"],
                generation_finished_at=resp["generation_finished_at"],
                error=resp["error"],
                timestamp=None,  # ResultsLogger stamps the record-write time
                **{k: v for k, v in meta.items()},
                **scores,
            ))

    # ── Generation ────────────────────────────────────────────────────────

    def _count_prompt_tokens(self, messages) -> int:
        """Dry-run estimate of input length, built exactly like the generation inputs."""
        if self.tokenizer is None:
            return sum(len(m["content"]) for m in messages) // 4
        return len(chat_input_ids(self.tokenizer, messages)[1])

    # ── Inspectable dataset files + manifest ──────────────────────────────

    def _write_dataset_files(self) -> None:
        (self.run_dir / "families.json").write_text(
            json.dumps([asdict(f) for f in self.families], indent=2) + "\n")
        with open(self.run_dir / "items.jsonl", "w") as f:
            for t in self.trials:
                f.write(json.dumps(ds.trial_metadata(t)) + "\n")
        with open(self.run_dir / "requests.jsonl", "w") as f:
            for k, msgs in self.requests.items():
                self.prompt_tokens[k] = self._count_prompt_tokens(msgs)
                rendered = chat_input_ids(self.tokenizer, msgs)[0] if self.tokenizer is not None else None
                f.write(json.dumps({"request_key": k, "messages": msgs, "rendered_text": rendered,
                                    "prompt_tokens": self.prompt_tokens[k]}) + "\n")
        self._write_prompts_preview()

    def _write_prompts_preview(self) -> None:
        """Every probe × condition (neutral branch) for the first family, plus the
        other branch lines — enough to read the exact framing at a glance."""
        fam = self.families[0]
        lines = [f"# Rendered prompts — {self.experiment_version} (door code in the {self.code_location} message)", "",
                 f"Family `{fam.family_id}` (codes {fam.code_a} / {fam.code_b}, mapping {fam.mapping}), "
                 f"variant a. All requests: requests.jsonl.", ""]
        for tfam in self.template_families:
            lines += [f"## Template family `{tfam}`", ""]
            first = True
            for t in self.trials:
                if (t.template_family, t.family_id, t.variant, t.branch) != (tfam, fam.family_id, "a", "neutral"):
                    continue
                if self.code_location == "user":
                    if first:  # identical system message in every request
                        lines += ["### system", "", "```", t.messages[0]["content"], "```", ""]
                        first = False
                    lines += [f"### {t.condition} / neutral / {t.probe}", "", "```", t.messages[1]["content"], "```", ""]
                else:  # the system message carries the code, so it differs per condition
                    lines += [f"### {t.condition} / neutral / {t.probe}", "",
                              "**system message**", "", "```", t.messages[0]["content"], "```", "",
                              "**user message**", "", "```", t.messages[1]["content"], "```", ""]
        lines += ["## Request branches (replace the `Request:` line)", ""]
        for b in ds.BRANCHES:
            lines += [f"- **{b}**: {self.prompts['branches'][b]}"]
        (self.run_dir / "prompts_preview.md").write_text("\n".join(lines) + "\n")

    def request_counts(self) -> dict:
        n_rows = len(self.trials)
        n_unique = len(self.requests)
        ptoks = [self.prompt_tokens[k] for k in self.requests]
        return {
            "trial_rows": n_rows,
            "unique_requests": n_unique,
            "deduplicated": n_rows - n_unique,
            "prompt_tokens_total": sum(ptoks),
            "prompt_tokens_mean": round(sum(ptoks) / max(1, len(ptoks)), 1),
            "max_output_tokens_total": n_unique * self.config.max_new_tokens,
            "token_count_method": "tokenizer" if self.tokenizer is not None else "chars/4 estimate",
            # Local HF inference: no per-token price. Cost = GPU time on your own hardware.
            "api_cost_usd": 0.0,
        }

    def _write_manifest(self) -> None:
        c = self.request_counts()
        lines = [
            "AGENT VS MODEL — attributed ignorance vs withholding — RUN MANIFEST",
            "=" * 68,
            f"run_id:             {self.run_id}",
            f"config:             {self._config_name}",
            f"experiment_version: {self.experiment_version}",
            f"code_location:      {self.code_location}  (user = frozen v1 layout; system = door code in the system message)",
            f"dry_run:            {self.dry_run}",
            f"model:              {self.config.model_name}  (adapter={self.config.adapter})",
            f"model_dtype:        {self.config.model_dtype}",
            f"temperature:        {self.config.temperature}  (do_sample={self.config.temperature > 0})",
            f"max_new_tokens:     {self.config.max_new_tokens}",
            f"batch_size:         {self.batch_size}",
            f"seed (codes):       {self.config.seed}",
            f"template_families:  {self.template_families}",
            f"template_sha:       " + ", ".join(f"{t}={ds.template_sha(self.prompts, t, self.code_location)}" for t in self.template_families),
            f"n_families:         {len(self.families)}  ({2 * len(self.families)} code variants)",
            f"git_commit:         {git_commit()}  (uncommitted changes: {git_dirty()})",
            f"input_pipeline:     {INPUT_PIPELINE}",
            "",
            "REQUEST COUNTS",
            "-" * 68,
            f"trial rows (templates × families × 2 variants × 4 conditions × 4 branches × 3 probes): {c['trial_rows']}",
            f"unique model requests after dedup: {c['unique_requests']}  ({c['deduplicated']} identical code_absent rows collapsed)",
            f"prompt tokens: total {c['prompt_tokens_total']}, mean {c['prompt_tokens_mean']}  [{c['token_count_method']}]",
            f"output tokens: at most {c['max_output_tokens_total']} ({c['unique_requests']} × {self.config.max_new_tokens})",
            "cost: local HF inference — no API charges; GPU time only.",
            "",
            "CODE FAMILIES (experimenter-only)",
            "-" * 68,
        ]
        for f in self.families:
            lines.append(f"{f.family_id}: a={f.code_a} b={f.code_b} mapping={f.mapping}")
        (self.run_dir / "manifest.txt").write_text("\n".join(lines) + "\n")

    # ── Evaluation / saving ───────────────────────────────────────────────

    def evaluate(self) -> dict:
        if self.dry_run:
            return {"dry_run": True, "run_dir": str(self.run_dir), **self.request_counts()}
        from experiments.agent_vs_model.agent_vs_model_analysis import summarize_run
        return summarize_run(self.output_path, self.run_dir,
                             bootstrap_iterations=self.bootstrap_iterations, seed=self.config.seed)

    def save_results(self) -> str:
        return str(self.run_dir if self.dry_run else self.output_path)
