# Activation analysis — persona self-recognition

The **analysis layer** over the activations captured during the self-recognition
experiment. It consumes the captured residual-stream features + the trial
metadata table; it does **not** modify the capture (`generate_text.py`) or
evaluation (`evaluate_self_recognition.py` / `evaluation_cases.py`) phases.

- **`analyze_activations.ipynb`** — the experiments (exp01…exp14), one cell each.
- **`analyze_activations_helpers.py`** — all non-experiment plumbing: R2 fetch,
  the safetensors+parquet store reader, the metadata joins, persona/task category
  maps, the math/stats toolkit (cosine, contrast, d′, AUROC, bootstrap, train/test
  split, project-out), artifact IO, generic plots, the cell-count reporter, the
  exp14 steering hooks, and the experiment registry.

## Running

Set **one knob** at the top of the notebook — the `run_name` from
`self_recognition_config.yaml` (e.g. `personacat_v1`). That name is the R2 prefix
both phases live under; `H.load_dataset(cfg)` fetches and joins them.

```python
RUN_NAME = "personacat_v1"
cfg = H.AnalysisConfig(run_name=RUN_NAME, dry_run=True)   # flip dry_run=False for the full run
ds  = H.load_dataset(cfg)
```

`dry_run=True` restricts every experiment to a small category-diverse persona
subset and caps rows per grouping cell, validating shapes / joins / that grouping
keys are populated before any full run. Each experiment prints **n-per-cell** and
its grouping keys, so silent pooling is caught.

Run experiments individually (each cell defines + runs + plots and caches into
`STORE`), or run the whole ordered suite at the bottom (`H.run_suite`). Build
order is respected: **exp01 and exp04 first**, **exp14 last** (skipped unless
`cfg.enable_steering`). Cross-experiment dependencies are materialised on demand
via `dep()`.

### Download size (laptop disk / time)

`load_dataset` downloads the activations from R2 the first time (then re-runs are
incremental — cached files are skipped). The eval activations dominate; for an
8B model with 9 banded layers each stored tensor is ~72 KB, so the eval store is
typically single-digit-to-low-tens of GB, generation ~150 MB. To stay in control
on a laptop:

```python
H.estimate_size(cfg)                 # fetches only the tiny index/metadata, prints exact GB
cfg.download_cases = ("case1", "case12")   # then fetch ONLY these eval cases' tensors (+ full gen)
ds = H.load_dataset(cfg)
cfg.metadata_only = True             # inspection only: index/metadata, no tensors
```

`download_cases` works because the capture writes shards roughly case-contiguous,
so restricting to the cases you're analysing genuinely cuts the transfer. Just
make sure the cases you list cover what the experiments you run actually read
(e.g. exp02/06/07/09 use the self-recognition case from `pick_self_case`; exp11
compares across the cases present).

### Storage / naming notes
- The capture pipeline shards into **safetensors (fp16) + `metadata.parquet`**
  (the brief's "Zarr" — this repo never adopted Zarr). Each capture is
  `[n_layers, hidden]`; the layer list is in the store's `manifest.json`. We read
  fp16, compute in fp32, store results small (parquet/npz for arrays, json for
  scalars) under `results/activation_analysis/<run_name>/artifacts/<expNN>/`.
- One `run_name` covers both phases: `runs/<run_name>/activations/` (generation)
  and `runs/<run_name>/eval_activations/` (eval). Multi-GPU runs write `shard_*/`
  subtrees; the reader globs recursively.
- Capture-name conventions are **not reinvented** — `final_prompt_token_before_answer`
  (decision token), `pre_text_token`, `text1/2_mean`, `generated_text_mean`
  (persona behavior), `persona_prompt_{mean,final}` (prompt-state), etc. See the
  constants block in `analyze_activations_helpers.py`.
- **Grouping keys are first-class everywhere:** `case`, `system_prompt_present`
  (eval_system_prompt_enabled), `other_description_style`, persona **coarse
  category** (suppression / near_twin / calibration / confound), `true_author`,
  correct/incorrect, answer/text position, confidence. Contrast vectors are built
  on a TRAIN split (grouped by `base_trial_id`) and evaluated on HELD-OUT trials;
  d′ and AUROC with bootstrap CIs are the defaults, not raw accuracy.

## Experiment map

Roles: **enabling/control** = the scaffolding the access claim is read against;
**access** = the central introspection/access rows; **depth** = mechanism/depth
findings independent of recognition score.

| exp | role | question | key inputs | key outputs / artifact |
|-----|------|----------|------------|------------------------|
| **exp01** | enabling | What is each persona's behavior vector (neutral generation) and prompt-state vector? | `generated_text_mean` (neutral gen), `persona_prompt_{mean,final}` | `persona_vectors.npz` (behavior + prompt_state per persona/layer) |
| **exp02** | access | Is there a self-recognition direction at the decision/read side, and does it **survive style removal**? | `text1/2_mean` (read side), decision token, exp04 nuisances, style dir | `self_rec_read_dirs.npz`; per-layer held-out AUROC raw vs style-removed |
| **exp03** | control | Which cases/personas actually recognize (behavioral baseline)? | trial table (`correct`, `decision_conf`) | `behavioral_baseline.parquet` (accuracy/confidence by case×condition×category) |
| **exp04** | enabling | What are the nuisance directions (A/B, text-position), and how aligned are they with the self-rec direction? | decision token, `predicted_answer`, `answer_order` | `nuisance_dirs.npz`; cosine vs self-rec; `H.project_out` (raw + cleaned, never auto-removed) |
| **exp05** | access | Are self-rec vectors persona-specific or shared, and aligned with each persona's behavior signature? | exp02 read dirs, exp01 behavior | shared-vs-distinct cosine by layer; self-rec↔behavior alignment |
| **exp06** | access | Self-prior (pre-text) vs evidence-driven uptake (decision)? | `pre_text_token`, decision, exp02 dir | `prior_vs_evidence.parquet` (prior, delta, false-self rate by category) |
| **exp07** | access | Online vs decision-time recognition (layer × token map); does the self signal follow the self-authored text across the position swap? | pre_text / text1 / text2 / decision, exp02 dir | `layer_token_maps.npz` (per coarse category) |
| **exp08** | access | Where is **authorship** decodable but **style/identity** not (candidate access locus)? | read-side spans (`span_samples`) | per-layer probe AUROC/bal-acc for author / persona-id / task / position |
| **exp09** | access | Genuine recognition vs confabulation (correct-confident vs incorrect-confident self-claims)? | decision token, confident self-claims | `genuine_dir.npz`; per-layer AUROC; cosine vs style / read dirs |
| **exp10** | access | Do the persona/self vectors transfer across task families (within vs cross)? | read-side spans by `task_category` | within/cross AUROC; **transfer gap** headline |
| **exp11** | control | Does "which is mine" share a mechanism with "which non-me persona wrote which"? | decision-token centroids per case (nuisance-cleaned) | `case_centroids.npz`; self-vs-third-party cosine |
| **exp12** | depth | Does the active evaluator drift toward the Other while reading it? | other-authored text spans, exp01 behavior | `switching.parquet` (drift by layer) — *stability, not access* |
| **exp13** | depth | Do confound / hidden-goal personas cluster (vs the historian control), independent of surface text? | exp01 behavior vectors | `behavior_clustering.npz` (cosine + cluster labels) — *no accuracy used* |
| **exp14** | access | **Causal:** does steering / projecting out the self-rec direction change the authorship judgment? | exp02 dir + local trial table (exact prompts) + the model | `steer_sweep.parquet`, `project_out_sweep.parquet` (layer × multiplier) |

Access-relevant: **02, 05, 06, 07, 08, 09, 10, 14**. Depth-relevant: **12, 13**.
Enabling/control: **01, 03, 04, 11**.

## exp14 caveat (causal)

exp14 re-runs forward passes with residual edits, so it needs a **GPU + the
model + the local eval trial table** (`results/text_evaluations/<task>/<run_name>/*.jsonl`,
which carries the exact `prompt_text` / `system_prompt_text`; unlike the
activations these are not in R2). Set `cfg.enable_steering=True`. It reuses the
eval's model/hook infrastructure (`HFBackend`,
`core.activation_capture.decoder_layers`) via `H.ResidualSteering` /
`H.steered_choice_probs` — nothing is rebuilt. On any other machine it prints
guidance and skips.

## Operationalization notes (documented choices)

- **exp02 self-rec direction.** Built read-side (mean reading self-authored span −
  mean reading other-authored span, confident-correct only) from the cleanest
  case present — case7 if available, else auto-fallback to case1 (single-text,
  decode-free) / case12. This direction is position-free and is what exp06/exp07
  project positions onto. A decision-token, position-entangled variant is also
  produced (`decision_dirs`); exp04 nuisance handling is its prerequisite.
- **Style direction** (projected out in exp02, reported in exp09) is defined once,
  order-independently, as PC1 of the exp01 persona-behavior vectors per layer (the
  dominant axis of surface stylistic variation). A Case-5 third-party style
  contrast is the documented alternative.
- **Nuisances present.** This A/B binary design has A/B (answer-letter) and
  text-position nuisances; Yes/No and answer-format directions don't exist here
  and are reported as absent rather than fabricated.
- **exp03** fills the `exp03` slot referenced in the brief's enabling/control group
  (its body was unspecified); implemented as the behavioral table the activation
  experiments are read against.
