"""Supporting machinery for the activation-analysis experiments
(analyze_activations.ipynb).

This file holds everything that is NOT experiment logic: fetching the captured
activations for a `run_name`, reading the on-disk store, joining activations to
the trial-metadata table, the persona/task category maps, the small math/stats
toolkit (cosine, contrast vectors, d', AUROC, bootstrap, train/test split,
project-out), artifact IO, generic plots, the cell-count reporter, the steering
hooks exp14 needs, and the experiment registry. The 14 experiments' own logic
lives in the notebook (one cell each) and calls into here.

STORAGE NOTE. The claude.ai brief calls the store "Zarr"; this repo never adopted
Zarr (it is not installed). The capture pipeline (core.activation_store) shards
into `safetensors` (fp16) + a `metadata.parquet` table, R2-mirrored. We read that
layout here. Each stored value is a [num_layers, hidden] fp16 tensor at one named
token position; the layer list is in the store's manifest.json.

R2 / run_name. A run (e.g. `personacat_v1`) writes BOTH phases under one name:
    runs/<run_name>/activations/         generation phase  (persona behavior etc.)
    runs/<run_name>/eval_activations/    eval phase (decision token, text means …)
`download_run_activations(run_id=run_name, ...)` fetches both into one local dir.
Multi-GPU runs write per-shard subtrees (shard_0/, shard_1/, …); the reader globs
recursively and concatenates, so single- and multi-GPU stores both just work.

CAPTURE-NAME CONVENTIONS (do not invent new ones — these are what the capture
phases wrote; see core.activation_capture / generation_activations /
binary_activations):

  Eval phase (binary_activations.eval_capture_spans), keyed by trial_id:
    DECISION              final_prompt_token_before_answer   (the "decision token")
    PRE_TEXT              pre_text_token
    TEXT1_MEAN/TEXT2_MEAN text1_mean / text2_mean            (read-side text spans)
    TEXT1_LAST10/…        text1_last10_mean / text2_last10_mean
    TEXT1_FINAL/…         text1_final / text2_final
    OTHER_DESC_MEAN       other_description_mean
    OTHER_DESC_1/2_MEAN   other_description_1_mean / other_description_2_mean
    PRE_DESC              pre_description_token
    SYS_FINAL             active_system_prompt_final
    ASSISTANT_HEADER      assistant_header_token

  Generation phase (generation_activations.generation_capture_spans), keyed by
  text_id "{group}/{persona}/{task_id}":
    GEN_TEXT_MEAN         generated_text_mean        (persona_behavior_vector source)
    GEN_TEXT_LAST10       generated_text_last10_mean
    PERSONA_PROMPT_MEAN   persona_prompt_mean        (prompt-state: mean system span)
    PERSONA_PROMPT_FINAL  persona_prompt_final       (prompt-state: final prompt token)
    GEN_PROMPT_FINAL      generation_prompt_final
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent                     # experiments/self_recognition
_REPO = _HERE.parents[1]                                    # repo root
PERSONAS_YAML = _HERE / "self_recognition_personas.yaml"
TASK_JSON = _REPO / "tasks" / "self_recognition_by_persona_category_tasks.json"
# Local cache for downloaded activations + a place for analysis artifacts.
ANALYSIS_ROOT = _HERE / "results" / "activation_analysis"
# Where evaluate_self_recognition.py wrote the trial table (local, not in R2).
TEXT_EVALUATIONS_DIR = _HERE / "results" / "text_evaluations"

_SEP = "\x1f"  # ActivationStore safetensors key separator (id\x1fcapture)
GEN_PHASE = "generation"
EVAL_PHASE = "evaluation"

# ── Capture-name constants (the existing convention; see module docstring) ─────
DECISION = "final_prompt_token_before_answer"
PRE_TEXT = "pre_text_token"
TEXT1_MEAN, TEXT2_MEAN = "text1_mean", "text2_mean"
TEXT1_LAST10, TEXT2_LAST10 = "text1_last10_mean", "text2_last10_mean"
TEXT1_FINAL, TEXT2_FINAL = "text1_final", "text2_final"
OTHER_DESC_MEAN = "other_description_mean"
OTHER_DESC_1_MEAN, OTHER_DESC_2_MEAN = "other_description_1_mean", "other_description_2_mean"
PRE_DESC = "pre_description_token"
SYS_FINAL = "active_system_prompt_final"
ASSISTANT_HEADER = "assistant_header_token"

GEN_TEXT_MEAN = "generated_text_mean"
GEN_TEXT_LAST10 = "generated_text_last10_mean"
PERSONA_PROMPT_MEAN = "persona_prompt_mean"
PERSONA_PROMPT_FINAL = "persona_prompt_final"
GEN_PROMPT_FINAL = "generation_prompt_final"


# ═══════════════════════════════════════════════════════════════════════════
# Persona / task category maps
# ═══════════════════════════════════════════════════════════════════════════

def coarse_category(category: str | None) -> str:
    """Bucket a fine persona category into the access-claim grouping used
    everywhere: suppression / near_twin / neutral / calibration / confound / other.
    Suppression and near-twin are the access-relevant rows; neutral domain experts,
    calibration anchors, and confounds are controls."""
    c = (category or "").strip().lower()
    if c.startswith("suppression"):
        return "suppression"
    if c.startswith("near-twin") or c.startswith("near twin"):
        return "near_twin"
    if c.startswith("neutral"):
        return "neutral"
    if c.startswith("calibration") or c.startswith("baseline"):
        return "calibration"
    if c.startswith("confound"):
        return "confound"
    return "other"


def load_persona_categories(path: Path = PERSONAS_YAML) -> dict[str, str]:
    """{persona_name: fine category} from the roster YAML."""
    import yaml
    with open(path) as f:
        doc = yaml.safe_load(f)
    out: dict[str, str] = {}
    for name, spec in (doc.get("personas") or {}).items():
        out[name] = (spec or {}).get("category", "")
    return out


def load_task_categories(path: Path = TASK_JSON) -> dict[str, str]:
    """{task_id: task-family label} from the task set metadata. Used as the
    'task family' grouping for exp01 (neutral allowlist) and exp10 (transfer)."""
    with open(path) as f:
        items = json.load(f)
    return {it["task_id"]: (it.get("metadata") or {}).get("category", "") for it in items}


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisConfig:
    """One stable name (`run_name`) selects which captured run to analyse; it is
    the same name the gen/eval configs use and the R2 prefix both phases live
    under. Everything else has a sane default.

    dry_run restricts every experiment to a small, category-diverse persona
    subset and caps rows per grouping cell, so shapes / joins / grouping keys can
    be validated before a full run.
    """
    run_name: str
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    task: str = "persona_category"            # locates the local eval trial table (exp14)
    bucket: Optional[str] = None              # else $R2_BUCKET
    local_dir: Optional[str] = None           # else ANALYSIS_ROOT/<run_name>/data
    generation_run_name: Optional[str] = None  # else run_name (gen+eval share it)
    force_download: bool = False
    # Analysis knobs
    confident_threshold: float = 0.60          # decision-conf cutoff for "confident"
    train_frac: float = 0.60
    n_bootstrap: int = 1000
    ci: float = 95.0
    seed: int = 0
    # Which cases anchor the self-rec experiments
    primary_case: str = "case7"                # active, no description (decoding excluded)
    comparison_case: str = "case3"             # active, other described (for comparison)
    # Neutral task family allowlist for persona-behavior vectors (None = all tasks).
    neutral_task_categories: Optional[tuple[str, ...]] = None
    # Download controls (the store can be large; eval activations dominate).
    #   metadata_only — fetch only the tiny index/metadata (no tensors); pair with
    #                   estimate_size() to see exact GB before committing disk.
    #   download_cases — restrict the EVAL tensor download to these case ids (the
    #                   capture writes shards roughly case-contiguous, so this saves
    #                   real space/time). Generation acts are small and always full.
    metadata_only: bool = False
    download_cases: Optional[tuple] = None
    # Dry-run controls
    dry_run: bool = False
    dry_run_personas: tuple[str, ...] = (
        "owl_suppressed", "historian_american", "historian_eastasian",
        "child_five", "default_neutral",
    )
    dry_run_max_per_cell: int = 40
    # exp14 only: re-running forward passes with hooks needs the model on a GPU.
    enable_steering: bool = False

    @property
    def gen_run(self) -> str:
        return self.generation_run_name or self.run_name

    @property
    def local_path(self) -> Path:
        return Path(self.local_dir) if self.local_dir else (ANALYSIS_ROOT / self.run_name / "data")

    @property
    def artifacts_dir(self) -> Path:
        return ANALYSIS_ROOT / self.run_name / "artifacts"


# ═══════════════════════════════════════════════════════════════════════════
# Fetch + read the activation store
# ═══════════════════════════════════════════════════════════════════════════

# Local sub-dir each phase's R2 prefix maps into (matching download_run_activations).
_PHASE_LOCALSUB = {GEN_PHASE: "activations", EVAL_PHASE: "eval_activations"}


def _phase_prefixes(cfg: AnalysisConfig) -> dict:
    return {GEN_PHASE: f"runs/{cfg.gen_run}/activations",
            EVAL_PHASE: f"runs/{cfg.run_name}/eval_activations"}


def _targeted_sync(cfg: AnalysisConfig, gen_filter, eval_filter, *,
                   desc: str = "downloading activations") -> int:
    """Download R2 objects matching a per-phase filter(rel_path, size)->bool into
    cfg.local_path, preserving sub-paths (same mapping as download_run_activations).

    - Shows a tqdm byte-level progress bar (intra-file too, via boto3 Callback).
    - rsync-like: files already present locally are skipped (reported up front).
    - Atomic: each file streams to a `.part` then renames on success, so an
      interrupted download resumes cleanly (no truncated file mistaken for done).
    Returns the count downloaded."""
    import os
    from core.activation_store import _r2_client
    bucket = cfg.bucket or os.environ.get("R2_BUCKET")
    if not bucket:
        raise ValueError("set cfg.bucket or the R2_BUCKET env var")
    client = _r2_client()
    local = cfg.local_path
    filt = {GEN_PHASE: gen_filter, EVAL_PHASE: eval_filter}
    pag = client.get_paginator("list_objects_v2")

    print(f"→ destination: {local}")
    print("  listing R2 objects …", flush=True)
    todo, skipped, skipped_bytes = [], 0, 0
    for phase, prefix in _phase_prefixes(cfg).items():
        base = local / _PHASE_LOCALSUB[phase]
        pref = prefix.rstrip("/") + "/"
        for page in pag.paginate(Bucket=bucket, Prefix=pref):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(pref):]
                size = obj.get("Size", 0)
                if not rel or not filt[phase](rel, size):
                    continue
                dest = base / rel
                if dest.exists() and not cfg.force_download:
                    skipped += 1; skipped_bytes += size; continue
                todo.append((obj["Key"], dest, size))
    total = sum(s for _, _, s in todo)
    print(f"  {len(todo)} file(s) to fetch ({total / 1e9:.2f} GB); "
          f"{skipped} already present ({skipped_bytes / 1e9:.2f} GB) — skipped.", flush=True)
    if not todo:
        return 0

    bar = None
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                   desc=desc, dynamic_ncols=True)
    except Exception:
        pass
    n = 0
    for i, (key, dest, size) in enumerate(todo, 1):
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        if bar is not None:
            bar.set_postfix_str(f"{i}/{len(todo)} {dest.name[:28]}", refresh=False)
            client.download_file(bucket, key, str(tmp), Callback=lambda nb: bar.update(nb))
        else:
            print(f"  [{i}/{len(todo)}] {dest.name} ({size / 1e6:.1f} MB)", flush=True)
            client.download_file(bucket, key, str(tmp))
        tmp.replace(dest)  # atomic: only a complete file lands at dest
        n += 1
    if bar is not None:
        bar.close()
    return n


def _not_tensor(rel, size):
    return not rel.endswith(".safetensors")


def _wanted_eval_shards(cfg: AnalysisConfig) -> set:
    """Shard filenames (acts_NNNNN.safetensors) holding eval ids for the selected
    cases, from the already-downloaded eval index.jsonl + metadata.parquet."""
    cases = set(cfg.download_cases or [])
    want = set()
    base = cfg.local_path / _PHASE_LOCALSUB[EVAL_PHASE]
    for evd in base.rglob(EVAL_PHASE):
        meta, idx = evd / "metadata.parquet", evd / "index.jsonl"
        if not (meta.exists() and idx.exists()):
            continue
        m = pd.read_parquet(meta)
        keep = set(m[m["case"].isin(cases)]["id"]) if cases else set(m["id"])
        for line in idx.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("id") in keep:
                want.add(f"acts_{e['shard']}.safetensors")
    return want


def estimate_size(cfg: AnalysisConfig) -> dict:
    """Fetch ONLY the index/metadata (tiny) and report the EXACT on-disk size of
    the full tensor download per phase, without downloading any tensors. Use this
    before committing disk on a laptop."""
    _targeted_sync(cfg, _not_tensor, _not_tensor)
    out = {}
    for phase in (GEN_PHASE, EVAL_PHASE):
        base = cfg.local_path / _PHASE_LOCALSUB[phase]
        n_ids, n_tensors, nlayers, hidden = 0, 0, None, None
        for d in base.rglob(phase):
            mf = d / "manifest.json"
            if mf.exists():
                man = json.loads(mf.read_text())
                nlayers = len(man.get("layers") or []); hidden = man.get("hidden_dim")
            idx = d / "index.jsonl"
            if idx.exists():
                for line in idx.read_text().splitlines():
                    line = line.strip()
                    if line:
                        n_ids += 1; n_tensors += len(json.loads(line).get("captures", []))
        gb = (n_tensors * (nlayers or 0) * (hidden or 0) * 2) / 1e9
        out[phase] = {"ids": n_ids, "tensors": n_tensors, "layers": nlayers,
                      "hidden": hidden, "GB": round(gb, 2)}
    out["total_GB"] = round(sum(v["GB"] for v in out.values() if isinstance(v, dict)), 2)
    print("FULL download size estimate (tensors only; metadata already fetched):")
    for phase in (GEN_PHASE, EVAL_PHASE):
        v = out[phase]
        print(f"  {phase:11s}: {v['ids']} ids, {v['tensors']} tensors, "
              f"{v['layers']}L x {v['hidden']}h -> {v['GB']} GB")
    print(f"  TOTAL: {out['total_GB']} GB  "
          f"(restrict with cfg.download_cases=(...) to fetch only some eval cases)")
    return out


def download(cfg: AnalysisConfig) -> Path:
    """Mirror this run's activations from R2 into cfg.local_path.

    Modes (set on cfg):
      default          full download of both phases (download_run_activations).
      metadata_only    only index/metadata, no tensors (inspection; pair with
                       estimate_size). The reader will find no vectors.
      download_cases   full generation tensors (small) + ONLY the eval shards
                       holding those cases' trials (saves disk/time on a laptop).

    No-op-friendly: if the requested data is already cached and force_download is
    off, the fetch is skipped. Returns the local dir."""
    local = cfg.local_path
    local.mkdir(parents=True, exist_ok=True)

    if cfg.metadata_only or cfg.download_cases:
        # 1) always pull the small index/metadata for both phases first.
        _targeted_sync(cfg, _not_tensor, _not_tensor)
        if cfg.metadata_only and not cfg.download_cases:
            logger.warning("metadata_only: no activation tensors downloaded "
                           "(inspection mode; use estimate_size).")
            return local
        # 2) full generation tensors (small) + only the wanted eval shards.
        shards = _wanted_eval_shards(cfg)
        logger.info(f"download_cases={cfg.download_cases}: fetching {len(shards)} eval shard(s)")
        n = _targeted_sync(
            cfg,
            gen_filter=lambda rel, sz: True,                                   # all generation
            eval_filter=lambda rel, sz: _not_tensor(rel, sz) or Path(rel).name in shards,
        )
        logger.info(f"downloaded {n} file(s) into {local}")
        return local

    # Full download of both phases — via _targeted_sync so it shows a progress bar
    # and resumes cleanly (lists every object, re-fetches only what's missing or
    # incomplete; atomic .part writes). Listing is cheap; the skip count is shown.
    _targeted_sync(cfg, gen_filter=lambda rel, sz: True, eval_filter=lambda rel, sz: True)
    return local


def _find_phase_dirs(local_dir: Path, phase: str) -> list[Path]:
    """Every store dir for `phase` under local_dir (single tree or shard_*/).
    A store dir is one named `phase` that holds a manifest.json or shards."""
    out = []
    for d in local_dir.rglob(phase):
        if d.is_dir() and ((d / "manifest.json").exists() or list(d.glob("acts_*.safetensors"))
                           or list(d.glob("meta_*.parquet")) or (d / "metadata.parquet").exists()):
            out.append(d)
    return sorted(out)


def _load_phase_meta(dirs: list[Path]) -> pd.DataFrame:
    """Concatenate every store dir's metadata, deduped on the join id."""
    frames = []
    for d in dirs:
        combined = d / "metadata.parquet"
        if combined.exists():
            frames.append(pd.read_parquet(combined))
        else:  # fall back to the per-shard parquets (run that never close()d cleanly)
            for p in sorted(d.glob("meta_*.parquet")):
                frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    return df


class PhaseStore:
    """Lazy reader over one phase's safetensors shards + metadata table.

    `vector(id, capture)` returns the [num_layers, hidden] fp32 array for one
    capture (or None if absent); `stack(ids, capture)` returns ([m, L, H], ids
    actually present) so an experiment can build a matrix and drop missing rows.
    """

    def __init__(self, dirs: list[Path], phase: str):
        self.phase = phase
        self.dirs = dirs
        self.meta = _load_phase_meta(dirs)
        self.layers: list[int] = []
        for d in dirs:
            mf = d / "manifest.json"
            if mf.exists():
                self.layers = list(json.loads(mf.read_text()).get("layers") or [])
                break
        self._handles: dict[Path, object] = {}
        self._index: dict[str, Path] = {}
        self._build_index()

    def _build_index(self) -> None:
        from safetensors import safe_open
        for d in self.dirs:
            for f in sorted(d.glob("acts_*.safetensors")):
                h = safe_open(str(f), framework="np")
                self._handles[f] = h
                for k in h.keys():
                    self._index.setdefault(k, f)  # first wins (dedup like meta)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def layer_index(self, layer: int) -> int:
        return self.layers.index(layer)

    def captures(self) -> list[str]:
        names = {k.split(_SEP, 1)[1] for k in self._index if _SEP in k}
        return sorted(names)

    def has(self, id_: str, capture: str) -> bool:
        return f"{id_}{_SEP}{capture}" in self._index

    def vector(self, id_: str, capture: str) -> Optional[np.ndarray]:
        key = f"{id_}{_SEP}{capture}"
        f = self._index.get(key)
        if f is None:
            return None
        return self._handles[f].get_tensor(key).astype(np.float32)

    def stack(self, ids, capture: str) -> tuple[np.ndarray, list[str]]:
        """([m, num_layers, hidden] fp32, present_ids) for the ids that have this
        capture, in input order."""
        vecs, present = [], []
        for i in ids:
            v = self.vector(i, capture)
            if v is not None:
                vecs.append(v)
                present.append(i)
        if not vecs:
            return np.zeros((0, self.n_layers, 0), dtype=np.float32), []
        return np.stack(vecs).astype(np.float32), present


@dataclass
class Dataset:
    """Both phases of one run, with enriched metadata tables and the category
    maps. `eval_meta` / `gen_meta` are the trial-metadata tables joined to the
    activations on `id` (= trial_id / text_id)."""
    cfg: AnalysisConfig
    gen: PhaseStore
    eval: PhaseStore
    eval_meta: pd.DataFrame
    gen_meta: pd.DataFrame
    persona_category: dict[str, str]
    task_category: dict[str, str]

    @property
    def layers(self) -> list[int]:
        return self.eval.layers or self.gen.layers


def _softmax2(la, lb):
    la = np.asarray(la, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    m = np.maximum(la, lb)
    ea, eb = np.exp(la - m), np.exp(lb - m)
    return ea / (ea + eb)


def _enrich_eval_meta(df: pd.DataFrame, cfg: AnalysisConfig,
                      pcat: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    # base_trial_id: the logical trial id minus the |sp=…|desc=… condition suffix.
    df["base_trial_id"] = df["trial_id"].astype(str).str.rsplit("|sp=", n=1).str[0]
    # 2-way A/B decision confidence from the full-vocab logprobs.
    if {"logprob_A", "logprob_B"}.issubset(df.columns):
        pa = _softmax2(df["logprob_A"].to_numpy(), df["logprob_B"].to_numpy())
        df["prob_A2"] = pa
        df["prob_B2"] = 1.0 - pa
        df["decision_conf"] = np.maximum(pa, 1.0 - pa)
    else:
        df["decision_conf"] = df.get("answer_confidence", np.nan)
    # Confidence cells. prob_correct (= answer_confidence) for correct trials,
    # decision_conf for the chosen answer in general.
    conf = df["decision_conf"].fillna(0.0)
    corr = df["correct"].fillna(False).astype(bool)
    df["is_confident"] = conf >= cfg.confident_threshold
    df["confident_correct"] = df["is_confident"] & corr
    df["confident_incorrect"] = df["is_confident"] & ~corr
    # Persona categories for the evaluator and (where defined) the true author.
    df["evaluator_category"] = df["evaluator_persona"].map(pcat).fillna("")
    df["evaluator_coarse"] = df["evaluator_category"].map(coarse_category)
    if "true_author" in df.columns:
        df["true_author_category"] = df["true_author"].map(pcat).fillna("")
    for col in ("text1_source_persona", "text2_source_persona"):
        if col in df.columns:
            df[col + "_category"] = df[col].map(pcat).fillna("")
    # Which text slot holds the evaluator's own (self) text, for paired cases.
    if {"text1_source_persona", "text2_source_persona", "evaluator_persona"}.issubset(df.columns):
        df["self_slot"] = np.where(
            df["text1_source_persona"] == df["evaluator_persona"], "text1",
            np.where(df["text2_source_persona"] == df["evaluator_persona"], "text2", "none"))
    return df


def _enrich_gen_meta(df: pd.DataFrame, cfg: AnalysisConfig, pcat: dict[str, str],
                     tcat: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "persona_category" not in df.columns or df["persona_category"].eq("").all():
        df["persona_category"] = df["persona"].map(pcat).fillna("")
    df["persona_coarse"] = df["persona_category"].map(coarse_category)
    df["task_category"] = df["task_id"].map(tcat).fillna("")
    if cfg.neutral_task_categories:
        df["is_neutral_task"] = df["task_category"].isin(cfg.neutral_task_categories)
    else:
        df["is_neutral_task"] = True  # all spbc prompts are neutral writing prompts
    return df


def load_dataset(cfg: AnalysisConfig) -> Dataset:
    """Fetch (if needed) and read both phases for `cfg.run_name`, returning a
    Dataset with enriched, category-tagged metadata joined to the activations."""
    local = download(cfg)
    gen_dirs = _find_phase_dirs(local, GEN_PHASE)
    eval_dirs = _find_phase_dirs(local, EVAL_PHASE)
    if not eval_dirs and not gen_dirs:
        raise FileNotFoundError(
            f"no activation store found under {local}. Check run_name "
            f"({cfg.run_name!r}), R2 credentials, or that the run captured activations.")
    pcat = load_persona_categories()
    tcat = load_task_categories()
    gen = PhaseStore(gen_dirs, GEN_PHASE)
    ev = PhaseStore(eval_dirs, EVAL_PHASE)
    return Dataset(
        cfg=cfg, gen=gen, eval=ev,
        eval_meta=_enrich_eval_meta(ev.meta, cfg, pcat),
        gen_meta=_enrich_gen_meta(gen.meta, cfg, pcat, tcat),
        persona_category=pcat, task_category=tcat,
    )


def load_trial_table(cfg: AnalysisConfig) -> pd.DataFrame:
    """The local PersonaBinaryEvalRecord rows (trials.jsonl / <slice>.jsonl) the
    eval wrote under results/text_evaluations/<task>/<run_name>/. Unlike the
    activation metadata.parquet (which travels in R2), these carry the exact
    `prompt_text` / `system_prompt_text` exp14 needs to re-run forward passes.
    Empty DataFrame if the eval output is not on this machine."""
    from core.run_utils import model_slug
    # Evals now write under <task>/<model_slug>/<run_name>/ (mirrors generation);
    # fall back to the legacy non-slugged path for pre-existing runs.
    rd = TEXT_EVALUATIONS_DIR / cfg.task / model_slug(cfg.model_name) / cfg.run_name
    if not rd.is_dir():
        rd = TEXT_EVALUATIONS_DIR / cfg.task / cfg.run_name
    rows = []
    if rd.is_dir():
        for f in sorted(rd.glob("*.jsonl")):
            if f.name.endswith("manifest.jsonl") or f.name == "trial_manifest.jsonl":
                continue
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize(ds: Dataset) -> None:
    """Print store shape, captured layers, captures, and per-case row counts."""
    print(f"run_name           : {ds.cfg.run_name}")
    print(f"layers (captured)  : {ds.layers}")
    print(f"eval trials        : {len(ds.eval_meta)}  | gen texts: {len(ds.gen_meta)}")
    print(f"eval captures      : {ds.eval.captures()}")
    print(f"gen  captures      : {ds.gen.captures()}")
    if not ds.eval_meta.empty:
        print("\neval rows by case × system_prompt_present:")
        print(ds.eval_meta.groupby(["case", "system_prompt_present"]).size().to_string())
        print("\neval rows by evaluator coarse-category:")
        print(ds.eval_meta.groupby("evaluator_coarse").size().to_string())
    if not ds.gen_meta.empty:
        print("\ngen rows by persona coarse-category:")
        print(ds.gen_meta.groupby("persona_coarse").size().to_string())


# ── dry-run / pooling-guard views ─────────────────────────────────────────────

def eval_view(ds: Dataset, cfg: AnalysisConfig | None = None,
              cases=None) -> pd.DataFrame:
    """The eval trial table, optionally filtered to a case set, and (in dry-run)
    to the diverse persona subset and capped per case × condition cell so shapes
    can be validated cheaply."""
    cfg = cfg or ds.cfg
    df = ds.eval_meta
    if cases is not None:
        df = df[df["case"].isin(list(cases))]
    if cfg.dry_run:
        keep = set(cfg.dry_run_personas)
        cols = [c for c in ("evaluator_persona", "true_author", "text1_source_persona",
                            "text2_source_persona", "other_persona") if c in df.columns]
        mask = np.zeros(len(df), dtype=bool)
        for c in cols:
            mask |= df[c].isin(keep).to_numpy()
        df = df[mask]
        df = (df.groupby(["case", "system_prompt_present"], group_keys=False)
                .apply(lambda g: g.head(cfg.dry_run_max_per_cell)))
    return df.reset_index(drop=True)


def gen_view(ds: Dataset, cfg: AnalysisConfig | None = None,
             neutral_only: bool = True) -> pd.DataFrame:
    cfg = cfg or ds.cfg
    df = ds.gen_meta
    if neutral_only and "is_neutral_task" in df.columns:
        df = df[df["is_neutral_task"]]
    if cfg.dry_run:
        df = df[df["persona"].isin(set(cfg.dry_run_personas))]
    return df.reset_index(drop=True)


def report_cells(df: pd.DataFrame, keys: list[str], title: str, max_rows: int = 40) -> None:
    """Print n-per-cell for the grouping keys so silent pooling is caught."""
    print(f"\n[{title}] grouping keys = {keys}; total n = {len(df)}")
    if df.empty or not keys:
        return
    g = df.groupby(keys).size().sort_values(ascending=False)
    print(g.head(max_rows).to_string())
    if len(g) > max_rows:
        print(f"… (+{len(g) - max_rows} more cells)")


# ═══════════════════════════════════════════════════════════════════════════
# Math / stats toolkit  (all fp32 in → small results out)
# ═══════════════════════════════════════════════════════════════════════════

def unit(v: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def cosine(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sum(unit(a, axis) * unit(b, axis), axis=axis)


def cosine_matrix(M: np.ndarray) -> np.ndarray:
    """[k, H] -> [k, k] pairwise cosine."""
    U = unit(M, axis=-1)
    return U @ U.T


def mean_diff(pos: np.ndarray, neg: np.ndarray, axis: int = 0) -> np.ndarray:
    """Contrast vector: mean(pos) - mean(neg) along `axis`."""
    return pos.mean(axis=axis) - neg.mean(axis=axis)


def proj_scalar(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Signed projection of rows of X [..., H] onto direction v [H] (v need not
    be unit; result is the dot with the unit vector)."""
    return X @ unit(v)


def project_out(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Remove the component of X [..., H] along v [H] (returns X_perp). The exp04
    helper returns BOTH X and project_out(X, v); nothing auto-removes."""
    vh = unit(v)
    return X - np.outer(X @ vh, vh).reshape(X.shape) if X.ndim == 2 else X - (X @ vh)[..., None] * vh


def project_out_vec(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Remove direction v from a single vector w (both [H])."""
    vh = unit(v)
    return w - (w @ vh) * vh


def top_principal_components(M: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors (principal directions) of mean-centered M
    [n, H] -> [min(k, rank), H]. Used to build a multi-dimensional style/identity
    subspace (a single PC1 is far too weak — persona identity is many-dimensional)."""
    Mc = M - M.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Mc, full_matrices=False)
    return vt[:k]


def project_out_subspace(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Remove the component of X [n, H] in the span of basis B [k, H] (B is
    orthonormalised internally). Returns X with the whole subspace projected out —
    the right tool for removing a k-dimensional style/identity subspace, as
    opposed to project_out which removes a single direction."""
    if B is None or len(B) == 0:
        return X
    u, s, vt = np.linalg.svd(np.asarray(B, np.float64), full_matrices=False)
    basis = vt[s > 1e-6]                       # orthonormal rows spanning B
    if len(basis) == 0:
        return X
    return (X - (X @ basis.T) @ basis).astype(X.dtype)


def dprime(pos: np.ndarray, neg: np.ndarray) -> float:
    """Sensitivity index between two 1-D score samples."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    s = np.sqrt(0.5 * (pos.var(ddof=1) + neg.var(ddof=1)))
    return float((pos.mean() - neg.mean()) / s) if s > 0 else float("nan")


def auroc(labels, scores) -> float:
    """AUROC of `scores` separating binary `labels` (needs both classes)."""
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores, float)))


def bootstrap_ci(stat_fn: Callable, arrays: tuple, n: int = 1000, seed: int = 0,
                 ci: float = 95.0) -> dict:
    """Bootstrap CI for a statistic over trials. `arrays` are equal-length and
    resampled jointly (paired). Returns {point, lo, hi, n}."""
    arrays = tuple(np.asarray(a) for a in arrays)
    m = len(arrays[0])
    point = stat_fn(*arrays)
    if m < 2:
        return {"point": float(point) if point == point else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n": m}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            v = stat_fn(*(a[idx] for a in arrays))
        except Exception:
            v = float("nan")
        if v == v:
            vals.append(v)
    lo, hi = (np.nan, np.nan) if not vals else np.percentile(vals, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n": m}


def train_test_idx(n: int, frac: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(round(frac * n))
    return perm[:k], perm[k:]


def split_by_groups(groups: np.ndarray, frac: float, seed: int = 0):
    """Train/test split that keeps whole groups (e.g. base_trial_id, task_id) on
    one side — so a contrast vector is never evaluated on a trial it was built
    from. Returns boolean masks (train, test)."""
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    k = int(round(frac * len(uniq)))
    train_g = set(uniq[perm[:k]].tolist())
    train = np.array([g in train_g for g in groups])
    return train, ~train


def logistic_auroc(Xtr, ytr, Xte, yte, C: float = 1.0) -> float:
    """Fit an L2 logistic probe on (Xtr,ytr); AUROC on held-out (Xte,yte)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ytr = np.asarray(ytr).astype(int); yte = np.asarray(yte).astype(int)
    if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
        return float("nan")
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(sc.transform(Xtr), ytr)
    s = clf.decision_function(sc.transform(Xte))
    return auroc(yte, s)


# ═══════════════════════════════════════════════════════════════════════════
# Activation assembly helpers (the joins every experiment shares)
# ═══════════════════════════════════════════════════════════════════════════

def stack_eval(ds: Dataset, meta: pd.DataFrame, capture: str):
    """([m, L, H], meta_present) for a captured eval position over the trials in
    `meta` (rows realigned to the activations actually present)."""
    arr, present = ds.eval.stack(meta["id"].tolist(), capture)
    sub = meta.set_index("id").loc[present].reset_index() if present else meta.iloc[:0]
    return arr, sub


def stack_gen(ds: Dataset, meta: pd.DataFrame, capture: str):
    arr, present = ds.gen.stack(meta["id"].tolist(), capture)
    sub = meta.set_index("id").loc[present].reset_index() if present else meta.iloc[:0]
    return arr, sub


def span_samples(ds: Dataset, meta: pd.DataFrame, *, position: str = "mean"):
    """Read-side text-span samples with a rich label table — the join exp08
    (probes), exp10 (transfer), exp12 (switching) share. Each paired trial yields
    two spans (text1, text2); each single-text trial one. Returns:
      V  [m, L, H] fp32, and a DataFrame with columns:
      author, evaluator, is_self, slot ("text1"/"text2"), task_id, task_category,
      base_trial_id, case, group.
    """
    cap = {"mean": (TEXT1_MEAN, TEXT2_MEAN), "last10": (TEXT1_LAST10, TEXT2_LAST10),
           "final": (TEXT1_FINAL, TEXT2_FINAL)}[position]
    V, rows = [], []
    pcat = ds.persona_category
    tcat = ds.task_category
    for r in meta.itertuples(index=False):
        ev = getattr(r, "evaluator_persona")
        for slot, capn, srccol in (("text1", cap[0], "text1_source_persona"),
                                    ("text2", cap[1], "text2_source_persona")):
            v = ds.eval.vector(r.id, capn)
            if v is None:
                continue
            author = getattr(r, srccol, None)
            if author is None or (isinstance(author, float) and author != author):
                author = getattr(r, "true_author", None)  # single-text slot
            V.append(v)
            rows.append({
                "author": author, "evaluator": ev, "is_self": author == ev,
                "slot": slot, "task_id": getattr(r, "task_id", None),
                "task_category": tcat.get(getattr(r, "task_id", None), ""),
                "author_category": pcat.get(author, ""),
                "base_trial_id": getattr(r, "base_trial_id", None),
                "case": getattr(r, "case", None), "group": getattr(r, "group", None),
            })
    if not V:
        return np.zeros((0, ds.eval.n_layers, 0), np.float32), pd.DataFrame()
    return np.stack(V).astype(np.float32), pd.DataFrame(rows)


def authorship_samples(ds: Dataset, meta: pd.DataFrame, *, position: str = "mean"):
    """Per-text-span read-side samples labelled by authorship (self vs other).

    Each paired trial yields TWO samples (text1 span + text2 span); each
    single-text trial yields ONE (text1_* holds the single text). Returns:
      V        [m, L, H] fp32 read-side activations,
      is_self  [m] bool   (this span was authored by the evaluator persona),
      persona  [m] evaluator persona, base [m] base_trial_id (group key).
    position ∈ {"mean","last10","final"} selects the span reduction.
    """
    cap1 = {"mean": TEXT1_MEAN, "last10": TEXT1_LAST10, "final": TEXT1_FINAL}[position]
    cap2 = {"mean": TEXT2_MEAN, "last10": TEXT2_LAST10, "final": TEXT2_FINAL}[position]
    V, is_self, persona, base = [], [], [], []
    for r in meta.itertuples(index=False):
        ev = getattr(r, "evaluator_persona")
        # text1 slot (also the single-text slot for case1/2)
        v1 = ds.eval.vector(r.id, cap1)
        if v1 is not None:
            src1 = getattr(r, "text1_source_persona", None)
            if src1 is None or (isinstance(src1, float) and src1 != src1):
                src1 = getattr(r, "true_author", None)   # single-text case
            V.append(v1); is_self.append(src1 == ev); persona.append(ev)
            base.append(getattr(r, "base_trial_id"))
        # text2 slot (paired cases only)
        v2 = ds.eval.vector(r.id, cap2)
        if v2 is not None:
            src2 = getattr(r, "text2_source_persona", None)
            V.append(v2); is_self.append(src2 == ev); persona.append(ev)
            base.append(getattr(r, "base_trial_id"))
    if not V:
        return (np.zeros((0, ds.eval.n_layers, 0), np.float32),
                np.array([], bool), np.array([]), np.array([]))
    return (np.stack(V).astype(np.float32), np.array(is_self, bool),
            np.array(persona), np.array(base))


# ═══════════════════════════════════════════════════════════════════════════
# Artifact IO
# ═══════════════════════════════════════════════════════════════════════════

def art_dir(cfg: AnalysisConfig, exp: str) -> Path:
    d = cfg.artifacts_dir / exp
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_npz(path: Path, **arrays) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def save_df(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Plot helpers (generic; experiment-specific plot fns live in the notebook)
# ═══════════════════════════════════════════════════════════════════════════

def _ax(ax=None, figsize=(7, 4)):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_layer_curves(layers, series: dict, *, ylabel: str, title: str,
                      hline: float | None = None, bands: dict | None = None, ax=None):
    """One per-layer curve per series-name; optional (lo,hi) CI bands and an
    hline (e.g. chance=0.5)."""
    ax = _ax(ax)
    for name, ys in series.items():
        ax.plot(layers, ys, marker="o", label=name)
        if bands and name in bands:
            lo, hi = bands[name]
            ax.fill_between(layers, lo, hi, alpha=0.15)
    if hline is not None:
        ax.axhline(hline, ls="--", c="grey", lw=1)
    ax.set_xlabel("layer"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_heatmap(M, *, xticks, yticks, title: str, xlabel="", ylabel="",
                 cmap="viridis", ax=None, annotate=False, vmin=None, vmax=None):
    ax = _ax(ax, figsize=(1 + 0.5 * len(xticks), 1 + 0.4 * len(yticks)))
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xticks))); ax.set_xticklabels(xticks, rotation=90, fontsize=7)
    ax.set_yticks(range(len(yticks))); ax.set_yticklabels(yticks, fontsize=7)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_bars(labels, vals, errs=None, *, ylabel: str, title: str,
              hline: float | None = None, ax=None):
    ax = _ax(ax, figsize=(1 + 0.5 * len(labels), 4))
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, capsize=3)
    if hline is not None:
        ax.axhline(hline, ls="--", c="grey", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(title)
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# Experiment registry  (the experiments themselves live in the notebook)
# ═══════════════════════════════════════════════════════════════════════════

# Canonical run order: exp01 + exp04 are prerequisites (building-block vectors /
# nuisance directions); exp14 (causal) runs last; the rest in numeric order.
ORDER = ["exp01", "exp04", "exp02", "exp03", "exp05", "exp06", "exp07", "exp08",
         "exp09", "exp10", "exp11", "exp12", "exp13", "exp14"]

# Which question each experiment answers + its role (for the README / suite log).
ROLE = {
    "exp01": "enabling", "exp02": "access", "exp03": "control", "exp04": "enabling",
    "exp05": "access", "exp06": "access", "exp07": "access", "exp08": "access",
    "exp09": "access", "exp10": "access", "exp11": "control", "exp12": "depth",
    "exp13": "depth", "exp14": "access",
}

EXPERIMENTS: dict[str, dict] = {}


def register(key: str, title: str):
    """Decorator: register an experiment's run(ds, cfg) under a numbered key."""
    def deco(fn):
        EXPERIMENTS[key] = {"key": key, "title": title, "run": fn,
                            "role": ROLE.get(key, "")}
        return fn
    return deco


def _result_cache_path(cfg: AnalysisConfig, key: str) -> Path:
    return art_dir(cfg, key) / "_result.pkl"


def cached_run(ds: Dataset, cfg: AnalysisConfig, key: str, store: dict,
               *, recompute: bool = False) -> dict:
    """Return experiment `key`'s result, reusing it (a) from the in-memory store,
    (b) from a pickled `_result.pkl` on disk, else computing it and caching both.

    This is what makes re-running cheap: once an experiment has run, its full
    result object (vectors, arrays, tables) is on disk and loads in milliseconds,
    so dependents (e.g. exp12 -> exp01) never recompute. Pass recompute=True to
    force a fresh run (and overwrite the cache)."""
    if not recompute and isinstance(store.get(key), dict) and not store[key].get("error"):
        return store[key]
    p = _result_cache_path(cfg, key)
    if not recompute and p.exists():
        import pickle
        try:
            store[key] = pickle.load(open(p, "rb"))
            print(f"[{key}] loaded cached result from {p}")
            return store[key]
        except Exception as e:
            print(f"[{key}] cache load failed ({e}); recomputing")
    res = EXPERIMENTS[key]["run"](ds, cfg, store)
    store[key] = res
    try:
        import pickle
        pickle.dump(res, open(p, "wb"))
    except Exception as e:
        print(f"[{key}] could not pickle result ({e}); artifacts still saved")
    return res


def run_suite(ds: Dataset, cfg: AnalysisConfig, keys=None, store=None,
              recompute=False) -> dict:
    """Run experiments in canonical order, reusing cached results on disk so
    re-runs are cheap (see cached_run). `store` caches in memory so later
    experiments consume earlier ones (e.g. exp02 needs exp04). exp14 is skipped
    unless cfg.enable_steering.

    recompute: False reuses any cached result; True forces a fresh run of every
    requested key; or pass a set/list of keys to force just those."""
    store = {} if store is None else store
    keys = keys or [k for k in ORDER if k in EXPERIMENTS]
    keys = [k for k in ORDER if k in keys and k in EXPERIMENTS]
    force = set(keys) if recompute is True else set(recompute or ())
    for k in keys:
        if k == "exp14" and not cfg.enable_steering:
            print(f"\n=== skip {k} (causal; set cfg.enable_steering=True with a GPU) ===")
            continue
        print(f"\n=== {k}: {EXPERIMENTS[k]['title']}  [{EXPERIMENTS[k]['role']}] ===")
        try:
            cached_run(ds, cfg, k, store, recompute=k in force)
        except Exception as e:  # keep the suite going; report the failure
            logger.exception(f"{k} failed")
            print(f"  !! {k} failed: {e}")
            store[k] = {"error": str(e)}
    return store


# ═══════════════════════════════════════════════════════════════════════════
# Steering hooks for exp14 (re-run forward passes with residual edits)
# ═══════════════════════════════════════════════════════════════════════════

class ResidualSteering:
    """Context manager that edits the residual stream during a forward pass.

    `edits` is a list of dicts:
      {"layer": int,            # actual decoder-block index (a captured layer)
       "vec": np.ndarray[H],    # direction (will be unit-normalised)
       "alpha": float,          # add strength (ignored for project_out)
       "mode": "add"|"project_out"|"sub",
       "positions": callable(seq_len)->list[int] | None}   # None = all positions

    Reuses core.activation_capture.decoder_layers so it tracks PEFT wrapping. Use
    with a torch.no_grad() forward over a single primed sequence (see
    steered_choice_probs)."""

    def __init__(self, model, edits: list[dict]):
        from core.activation_capture import decoder_layers
        self.blocks = decoder_layers(model)
        self.edits = edits
        self.handles = []

    def __enter__(self):
        import torch
        for e in self.edits:
            self.handles.append(self.blocks[e["layer"]].register_forward_hook(self._hook(e)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, e):
        import torch
        v = torch.tensor(np.asarray(e["vec"], np.float32))
        v = v / (v.norm() + 1e-8)
        alpha = float(e.get("alpha", 0.0))
        mode = e.get("mode", "add")
        positions = e.get("positions")

        def hook(_m, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out  # [b, seq, H]
            vv = v.to(hs.dtype).to(hs.device)
            seq = hs.shape[1]
            idx = positions(seq) if positions else list(range(seq))
            if not idx:
                return out
            sel = hs[:, idx, :]
            if mode == "project_out":
                coef = sel @ vv
                sel = sel - coef.unsqueeze(-1) * vv
            elif mode == "sub":
                sel = sel - alpha * vv
            else:  # add
                sel = sel + alpha * vv
            hs[:, idx, :] = sel
            return (hs, *out[1:]) if isinstance(out, tuple) else hs
        return hook


def steered_choice_probs(backend, messages, choices, edits, primer=None) -> dict:
    """A/B (or any-choice) probabilities under residual edits, read the same way
    the eval does (last-sub-token at the primed answer position). Single
    unbatched forward so positions line up with the unpadded tokenization."""
    import torch
    import torch.nn.functional as F
    from core.self_recognition import ANSWER_PRIMER, render_with_primer
    text = render_with_primer(backend, messages, primer or ANSWER_PRIMER)
    enc = backend.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(backend.input_device)
    am = torch.ones_like(ids)
    tok = [backend.tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]
    with torch.no_grad(), ResidualSteering(backend.model, edits):
        logits = backend.model(ids, attention_mask=am).logits[0, -1, :]
    sub = torch.tensor([logits[t].item() for t in tok])
    p = F.softmax(sub, dim=0)
    return {c: float(p[j]) for j, c in enumerate(choices)}
