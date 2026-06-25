# Paper replication — Panickssery, Bowman & Feng (2024)

Faithful replication of **"LLM Evaluators Recognize and Favor Their Own
Generations"** (arXiv:2404.13076), used as a sanity check that this repo's
self-recognition measurement machinery reproduces a published result.

Reference code: https://github.com/ArjunPanickssery/self_recognition

## What this reproduces

The **out-of-the-box** experiment (paper §2) for a single local evaluator: one
model judges candidate news-article summaries drawn from five sources and we
measure whether it recognizes / prefers its own.

Four measurements (all via constrained first-token probabilities — the paper's
confidence measure — read by `HFBackend.get_choice_probs`):

| phase | setting | prompt | paper |
|---|---|---|---|
| `detection` | pairwise self-recognition | "which did you write?" (1/2) | Table 5, Fig 2-left |
| `comparison` | pairwise self-preference | "which do you prefer?" (1/2) | Table 5, Fig 4-left |
| `recognition` | individual self-recognition | "did you write this?" (Yes/No) | Table 10, Fig 2-right/3 |
| `scoring` | individual self-preference | Likert 1–5 | Table 10, Fig 4-right |

Pairwise scores are ordering-bias-corrected exactly as the reference repo: each
pair is run in both orders and the evaluator's confidence in its **own** summary
is averaged, `0.5·(P_fwd(self) + P_bwd(self))`.

## No API keys needed

The candidate summaries for **all** sources (`human`, `gpt4`, `gpt35`, `llama`,
`claude`) are the paper authors' own released outputs, vendored under
[data/](data/) (articles + summaries, 1000 each for XSUM and CNN, from the
reference repo). Only the **evaluator** runs locally. Its "own" text is the
released `llama` summary set, so the faithful evaluator is
`meta-llama/Llama-2-7b-chat-hf`.

Prompts in [paper_replication_config.yaml](paper_replication_config.yaml) are
copied **verbatim** from the reference repo (including the original `Aricle:`
typo in the recognition template). Do not "fix" them.

## Run

```bash
python run.py paper_replication_dev                 # TinyLlama smoke test (3 articles)
python run.py paper_replication_llama_xsum          # faithful XSUM run (Llama-2-7b)
python run.py paper_replication_llama_cnn           # faithful CNN run
# override knobs, e.g. a quick 20-article check:
python run.py paper_replication_llama_xsum --override sample_size=20
```

Outputs land in `experiments/paper_replication/results/<run>/`:
`trials.jsonl`, `pairwise_scores.csv`, `individual_scores.csv`, `scores.png`,
`summary.md` (which prints the paper's published Llama-2-7b values next to the
observed ones), and `manifest.txt`.

## What "working" looks like

Per the paper, Llama-2-7b out-of-the-box has **weak** self-recognition: ~chance
(0.5) at distinguishing itself from other LLMs, with **Human the only** reliably
distinguishable alternative. The machinery is working if the Human column stands
out from the model columns — not if exact values match (model snapshots and
sampling differ).

## Not yet implemented

Paper §3 (fine-tuning to vary self-recognition, control tasks, and the
recognition↔preference correlation, Figs 1/7) is intentionally out of scope for
this first pass. API-model evaluators (GPT-4/GPT-3.5 via logprobs) can be added
with an OpenAI-compatible backend when keys are available.
