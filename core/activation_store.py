"""Sharded on-disk store for captured activations + a columnar metadata table.

Realizes the activation-capture spec's STORAGE intent with the libraries this repo
actually has (zarr/boto3 are not installed): instead of Zarr we shard into
`safetensors` files (fp16), and metadata goes to Parquet — both consistent with
the repo's existing activation convention (core.train_probes reads tensor files +
a metadata table). The layout is flat and R2-friendly:

    <root>/<phase>/
        manifest.json              run-level config (layers, hidden_dim, names…)
        index.jsonl                {id, shard, captures} per stored id (resume)
        acts_00000.safetensors     ~shard_size ids; keys "<id>\x1f<capture_name>"
        meta_00000.parquet         metadata rows for that shard
        ...
        metadata.parquet           all rows, written at close()

Each stored value is a [num_layers, hidden] fp16 tensor (one capture at one named
position). None captures (e.g. text2 for single-text cases) are simply omitted.
Shards cap at ~shard_size ids so a crashed run keeps every flushed shard, and
add() can skip ids already present (resume). sync_to_r2() mirrors the tree to a
Cloudflare R2 (S3-compatible) bucket.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

_SEP = "\x1f"  # unit separator: safe in safetensors keys, absent from ids/names


class ActivationStore:
    """Accumulate per-id activation captures and flush them in shards.

    Usage:
        store = ActivationStore(root, phase="evaluation", layers=layers,
                                hidden_dim=4096, model="...", shard_size=1000)
        if not store.has(trial_id):
            store.add(trial_id, {capture_name: tensor[L,hidden] | None}, meta_row)
        store.close()
    """

    def __init__(self, root, phase: str, *, layers, hidden_dim: int | None = None,
                 model: str = "", shard_size: int = 1000, capture_names=None):
        self.dir = Path(root) / phase
        self.dir.mkdir(parents=True, exist_ok=True)
        self.phase = phase
        self.layers = list(layers)
        self.shard_size = max(1, int(shard_size))
        self._buf_tensors: dict[str, torch.Tensor] = {}
        self._buf_meta: list[dict] = []
        self._buf_ids: list[str] = []
        self._shard_idx = self._next_shard_index()
        self._done = self._load_done_ids()
        self._manifest = {
            "phase": phase, "model": model, "layers": self.layers,
            "hidden_dim": hidden_dim, "dtype": "float16", "shard_size": self.shard_size,
            "capture_names": list(capture_names) if capture_names else None,
            "separator": "\\x1f",
        }
        self._write_manifest()

    # ── resume bookkeeping ────────────────────────────────────────────────
    def _index_path(self) -> Path:
        return self.dir / "index.jsonl"

    def _load_done_ids(self) -> set[str]:
        done: set[str] = set()
        p = self._index_path()
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:
                        pass
        return done

    def _next_shard_index(self) -> int:
        existing = sorted(self.dir.glob("acts_*.safetensors"))
        return (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0

    def _write_manifest(self) -> None:
        (self.dir / "manifest.json").write_text(json.dumps(self._manifest, indent=2))

    def has(self, id_: str) -> bool:
        return id_ in self._done or id_ in self._buf_ids

    # ── accumulation + flush ──────────────────────────────────────────────
    def add(self, id_: str, captures: dict, meta: dict | None = None) -> None:
        """Buffer one id's captures ({name: [L,hidden] tensor | None}) + a metadata
        row. None captures are skipped. Flushes a shard when the buffer is full."""
        if self.has(id_):
            return
        stored_names = []
        for name, tensor in captures.items():
            if tensor is None:
                continue
            self._buf_tensors[f"{id_}{_SEP}{name}"] = tensor.to(torch.float16).contiguous()
            stored_names.append(name)
        row = dict(meta or {})
        row["id"] = id_
        self._buf_meta.append(row)
        self._buf_ids.append(id_)
        self._pending_index = getattr(self, "_pending_index", [])
        self._pending_index.append({"id": id_, "captures": stored_names})
        if len(self._buf_ids) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf_ids:
            return
        from safetensors.torch import save_file
        shard = f"{self._shard_idx:05d}"
        if self._buf_tensors:
            save_file(self._buf_tensors, str(self.dir / f"acts_{shard}.safetensors"))
        _write_parquet(self.dir / f"meta_{shard}.parquet", self._buf_meta)
        # append index entries (with the shard they landed in)
        with open(self._index_path(), "a") as f:
            for e in self._pending_index:
                f.write(json.dumps({**e, "shard": shard}) + "\n")
        self._done.update(self._buf_ids)
        self._buf_tensors, self._buf_meta, self._buf_ids, self._pending_index = {}, [], [], []
        self._shard_idx += 1

    def close(self) -> Path:
        """Flush the final shard and write the combined metadata.parquet."""
        self._flush()
        combined = _concat_parquets(sorted(self.dir.glob("meta_*.parquet")))
        out = self.dir / "metadata.parquet"
        if combined is not None:
            combined.to_parquet(out)
        return out


def _write_parquet(path: Path, rows: list[dict]) -> None:
    import pandas as pd
    if rows:
        pd.DataFrame(rows).to_parquet(path)


def _concat_parquets(paths):
    import pandas as pd
    frames = [pd.read_parquet(p) for p in paths if Path(p).exists()]
    return pd.concat(frames, ignore_index=True) if frames else None


# ── Cloudflare R2 (S3-compatible) sync ───────────────────────────────────────

def _r2_client(endpoint_url=None, access_key_id=None, secret_access_key=None):
    try:
        import boto3  # noqa: F401
    except ImportError as e:
        raise RuntimeError("R2 functions need boto3 (`pip install boto3`)") from e
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or os.environ.get("R2_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL"),
        aws_access_key_id=access_key_id or os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=secret_access_key or os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def sync_to_r2(local_dir, bucket: str, prefix: str = "", *, endpoint_url: str | None = None,
               access_key_id: str | None = None, secret_access_key: str | None = None) -> int:
    """Upload every file under `local_dir` to an R2 bucket (recursively), keyed by
    its path relative to local_dir under `prefix`. Returns the number of files uploaded.

    Credentials/endpoint default to env: R2_ENDPOINT / R2_ACCESS_KEY_ID /
    R2_SECRET_ACCESS_KEY (falling back to the AWS_* names). Requires boto3.
    """
    client = _r2_client(endpoint_url, access_key_id, secret_access_key)
    local_dir = Path(local_dir)
    n = 0
    for f in local_dir.rglob("*"):
        if f.is_file():
            key = "/".join([p for p in (prefix.rstrip("/"), str(f.relative_to(local_dir))) if p])
            client.upload_file(str(f), bucket, key)
            n += 1
    return n


def sync_from_r2(bucket: str, prefix: str, local_dir, *, skip_existing: bool = True,
                 endpoint_url: str | None = None, access_key_id: str | None = None,
                 secret_access_key: str | None = None) -> int:
    """Download every object under `prefix` in `bucket` to `local_dir`, preserving
    the sub-path structure. Returns the number of files downloaded.

    With skip_existing=True (default), files already present locally are skipped —
    so repeated calls only fetch what is new (safe to re-run).
    """
    client = _r2_client(endpoint_url, access_key_id, secret_access_key)
    local_dir = Path(local_dir)
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    paginator = client.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]  # strip the prefix to get local sub-path
            dest = local_dir / rel
            if skip_existing and dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))
            n += 1
    return n


def download_run_activations(run_id: str, local_dir, *,
                             generation_run_id: str | None = None,
                             bucket: str | None = None,
                             skip_existing: bool = True,
                             endpoint_url: str | None = None,
                             access_key_id: str | None = None,
                             secret_access_key: str | None = None) -> dict[str, int]:
    """Download activations for one experiment pair from R2 to a local directory.

    The generation phase and eval phase have SEPARATE run_ids (different timestamps)
    and are stored under separate R2 prefixes:

        runs/<generation_run_id>/activations/       ← generation-phase store
        runs/<eval_run_id>/eval_activations/        ← eval-phase store

    This function downloads both into one local directory so analysis can load
    them together:

        <local_dir>/activations/generation/          (metadata.parquet + safetensors)
        <local_dir>/eval_activations/evaluation/     (metadata.parquet + safetensors)

    Args:
        run_id:              The EVAL run_id (the timestamp from evaluate_self_recognition).
                             Also used as the generation run_id if generation_run_id is None.
        generation_run_id:   The GENERATION run_id if different from run_id (the timestamp
                             from generate_text). Read from the eval activation metadata
                             parquet's `generation_run_id` column if you need to look it up.
        local_dir:           Where to download. Created if it doesn't exist.
        skip_existing:       Skip files already present locally (safe to re-run).

    Returns {phase_prefix: n_files_downloaded}.

    Example (in a notebook):
        from core.activation_store import download_run_activations
        import pandas as pd

        counts = download_run_activations(
            run_id="20260624_191234",            # eval run_id
            generation_run_id="20260624_181802", # gen run_id (printed when gen ran)
            local_dir="analysis/my_run",
        )

        gen_meta  = pd.read_parquet("analysis/my_run/activations/generation/metadata.parquet")
        eval_meta = pd.read_parquet("analysis/my_run/eval_activations/evaluation/metadata.parquet")

        # join on text_id to link generation activations to eval trials
        joined = eval_meta.merge(gen_meta, on="text_id", suffixes=("_eval", "_gen"))
    """
    bucket = bucket or os.environ.get("R2_BUCKET")
    if not bucket:
        raise ValueError("bucket must be set or R2_BUCKET env var must be present")
    kw = dict(skip_existing=skip_existing, endpoint_url=endpoint_url,
              access_key_id=access_key_id, secret_access_key=secret_access_key)
    results: dict[str, int] = {}

    # Generation-phase store — under the generation run_id.
    gen_rid = generation_run_id or run_id
    n_gen = sync_from_r2(bucket, f"runs/{gen_rid}/activations",
                         Path(local_dir) / "activations", **kw)
    results[f"runs/{gen_rid}/activations"] = n_gen

    # Eval-phase store — under the eval run_id.
    n_eval = sync_from_r2(bucket, f"runs/{run_id}/eval_activations",
                          Path(local_dir) / "eval_activations", **kw)
    results[f"runs/{run_id}/eval_activations"] = n_eval

    return results
