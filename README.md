# PSM Introspection

Testing how complete the Persona Selection Model (PSM) is as an explanation of LLM behavior. PSM claims that LLMs learn diverse personas during pre-training and post-training refines one "Assistant" persona. This framework measures whether personas change the model's actual computation and self-monitoring, or are just surface-level output filters.

## Experiments

### 1. Cross-Persona Prediction

Put the model in **reporter persona A** and ask it to predict what **target persona B** would do on a task. Then run target B directly. Compare predictions to actuals.

Measures: choice match rate, confidence-bin error, refusal match.

### 2. Source × Reporter Matrix (shared-prefix KV cache reuse)

For each task and **source persona S**:
1. Run the source prefix (system prompt + task + pause cue) through the model
2. Save `past_key_values` — this captures the source's computational state
3. Measure source-state metrics at the pause point (answer distribution, entropy, confidence proxy)
4. **Source completion**: let S continue from the saved state and produce its answer
5. **Reporter continuation**: for each reporter R, append a reporter suffix that reuses the same `past_key_values` and asks R to report the model's current inclination

The key insight: the reporter continuation shares the exact same computational prefix as the source. If reporter R can accurately report source S's state, it has **privileged access** to that state through the shared KV cache — not just general knowledge about what S would do.

A **no-cache baseline** asks reporters to predict source behavior from scratch (fresh generation, no shared state). The difference between cache and no-cache accuracy measures the privileged access effect.

**Terminology note**: This uses shared-prefix KV cache reuse — we run a prefix, save `past_key_values`, then continue decoding with different suffixes. This is NOT activation patching.

## How to Run

```bash
# Install
pip install -e .

# List available experiment configs
python run_experiment.py --list

# Run the small dev experiment
python run_experiment.py cross_persona_prediction_dev

# Run with overrides
python run_experiment.py cross_persona_prediction_dev --override sample_size=3

# Run the source-reporter matrix
python run_experiment.py source_reporter_matrix_dev

# Run all experiments
python run_experiment.py --all
```

Results are written as JSONL to `results/raw/`.

## Project Structure

```
config/              # YAML configs for experiments, personas, models
tasks/               # Task JSON files (factual, opinion, borderline)
cross_persona_introspection/
  schemas.py         # Typed dataclasses
  backends/
    hf_backend.py    # HuggingFace local model backend
    openrouter_backend.py  # Optional LLM judge backend
  core/
    persona_inducer.py     # System-prompt persona induction
    task_loader.py         # Load and sample tasks
    response_parser.py     # Parse constrained/free-form responses
    kv_cache.py            # Shared-prefix KV cache reuse helpers
    results_logger.py      # JSONL logging
  experiments/
    base.py                # Base experiment class
    cross_persona_prediction.py
    source_reporter_matrix.py
  evaluation/
    choice_matching.py     # Exact match scoring
    calibration.py         # Confidence calibration
    llm_judge.py           # Optional LLM judge wrapper
analysis/
  summarize.py       # Result summarization
  analysis.ipynb     # Analysis notebook
run_experiment.py    # CLI entrypoint
tests/
  test_smoke.py      # Smoke tests
```

## What Is Implemented

- Full cross-persona prediction experiment (end-to-end)
- Full source × reporter matrix with shared-prefix KV cache reuse
- No-cache baseline for the matrix experiment
- System-prompt persona induction (5 personas)
- HF backend with generation, logit access, and KV cache reuse
- Constrained response parsing
- JSONL incremental logging
- Choice matching, confidence calibration evaluation
- CLI with config overrides

## What Is NOT Implemented Yet

- Vector-steering persona induction
- Activation patching
- RunPod orchestration
- Fine-tuning code
- Free-form LLM judge evaluation (stub only)
- Advanced analysis/visualization
- Base model (non-instruct) comparison
