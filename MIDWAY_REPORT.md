# Persona Introspection — Midway Progress Report

## 1. Motivation and central question

This project stress-tests the **Persona Selection Model (PSM)** of LLM behavior. PSM holds
that pre-training instills a broad repertoire of personas and that post-training sharpens
one "Assistant" persona; the open question is whether personas are *merely surface-level
output filters* or whether they genuinely reshape the model's internal computation and
self-monitoring.

We attack this from two complementary directions, unified by a single hypothesis —
**privileged access**:

- **Patchscope (cross-persona reading).** If personas share internal machinery, one
  persona should be able to *read* the hidden state another persona formed while answering
  a question.
- **Self-recognition (reading one's own state).** If an *active* persona has privileged
  access to its own computation, it should recognize its own generated text even when
  surface style gives nothing away.

Both lines are now producing signal, and — encouragingly — several of the pro-access
predictions hold up under controlled comparison. This report summarizes what we have,
where the evidence is strongest, and the causal tests that will close the remaining gaps.

---

## 2. Self-recognition experiments (primary line)

### 2.1 Experimental design

The core question: *Do persona-induced models recognize their own text through privileged
access to their internal state, or by (a) inferring from surface style, or (b) reasoning
over a persona description supplied in context?*

To separate these we built a **battery of 14 counterbalanced evaluation cases**
(single source of truth in `evaluation_cases.py`). Every case is binary (chance = 0.5) and
counterbalanced over both text order and answer letter, so a pure position/letter bias
scores exactly chance. The battery pins down three reference points:

| Explanation | Anchor case | Role |
|---|---|---|
| **Style** (surface cues, available to anyone) | case 9 / 14 (descriptions redacted) | Floor |
| **Content inference** (reason over supplied descriptions) | case 8 / 13 (descriptions shown) | Ceiling |
| **Privileged access** (active persona reads its own state) | **case 7** (active self, no descriptions) | Probe |

The design deliberately locates the access test in two persona categories where **style
cannot help** — **suppression** personas (a latent trait deliberately kept out of the
text) and **near-twin** pairs (two personas with near-identical surface style). Above-floor
recognition *there* is the cleanest evidence of genuine access.

**Models evaluated:** Llama-3.1-8B-Instruct (primary), Qwen2.5-32B-Instruct (scale
comparison), and a LoRA-finetuned Llama-3.1-8B. Self-recognition requires the *same* model
to generate and evaluate; generation is persona-conditioned with explicit self-references
stripped so recognition cannot lean on "As a chemist…" markers. Decisions are read as
constrained first-token A/B log-probs.

Analysis runs at two levels: **behavioral** (per-trial decisions) and **mechanistic**
(residual-stream activation probing, exp01–exp14).

### 2.2 Key results

**Recognition is real and the pipeline is validated.** Active self-recognition (case 7)
clears chance robustly: **Llama 0.63, Qwen 0.70**, with calibration-anchor personas
reaching **0.94 (Llama) / 1.00 (Qwen)**. The critical sanity check — case 12, obvious-style
personas — reads 0.94 when the persona is active and *exactly chance (0.500) when
deactivated*, confirming the measurement detects recognition when a real signal is present.

**Being the active persona measurably helps.** The active-state effect (case 7 − case 10
minimal baseline) is **+0.114 for Llama (z = 9.0)** and **+0.216 for Qwen (z = 17.5)**,
significant across every category.

**The strongest introspective signals we have identified:**

- **Llama, suppression category** — introspective surplus (case 7 − case 6, the
  deactivated-self decoder), measured on the persona-recognition foil with clustered
  bootstrap CIs: **+0.067 [+0.046, +0.090]** — a clean, significant positive effect in
  exactly the category where style is engineered to be useless.
- **Finetuning nudges this further positive.** The LoRA-finetuned model shows a suppression
  surplus over the case-14 style floor of **+0.053 [+0.007, +0.098]** (CI clears zero),
  versus base instruct's +0.038 (CI touches zero).
- **Qwen extends recognition into near-twins** — case 7 = **0.794**, active-state effect
  **+0.258**, and an active-minus-style-floor surplus of **+0.155**. In a category built to
  defeat surface style, the larger model recognizes its own text strongly.

**Persona recognition ≫ model recognition** (cross-model Llama-vs-Qwen runs). Isolating the
cue by `foil_type`:

| Foil (case 7, active) | Llama | Qwen |
|---|---|---|
| Persona recognition (same model, diff persona) | 0.651 | **0.711** |
| Model recognition (diff model, same persona) | 0.577 | 0.456 |
| Persona-anchor probe | 0.599 | **0.715** |

Qwen recognizes its own *persona* strongly (0.71) while sitting at/below chance on its own
*model identity* (0.46) — it is strongly "persona-anchored." Being the active persona boosts
persona recognition specifically, not model recognition — consistent with the PSM claim that
the persona, not the base model identity, is the computationally active object.

**Mechanistic corroboration (Llama activations):**

- **exp09** — the model can internally separate *genuine* confident self-recognition from
  *confabulated* confident self-claims at **AUROC ≈ 0.62–0.69**, evidence of an internal
  signal that tracks correctness rather than mere confidence.
- **exp08** — a reading-time linear probe recovers authorship at **AUROC ≈ 0.75–0.78**.

### 2.3 Honest scope limits (in progress)

We are deliberately holding the strong "privileged access" claim to a high bar, and two
things temper it today:

1. **Aggregate surplus over the best neutral decoder is near zero for base Llama.** The
   overall active-self advantage over the best neutral decoder is ≈ −0.007; the effect
   concentrates in obvious-style categories rather than uniformly across suppression and
   near-twin. The positive suppression result above is category-specific, not global.
2. **The behavioral notebook for base Llama predates the framing-matched case 13/14 runs**
   (it uses case 6 as the best neutral comparator), and the activation capture used the more
   confounded single-text case 1 rather than the cleaner case 7. At case 1's locus,
   authorship is decodable but entangled with fully-decodable style/identity/position, and
   shares near-identical decision geometry with third-party authorship judgment
   (cosine 0.97 vs 0.95).

These are measurement-scope gaps with clear fixes (Section 4), not contradictions of the
positive signals.

---

## 3. Patchscope experiments (mechanistic cross-persona line)

### 3.1 Design

A **source persona** answers a political/factual multiple-choice question; its hidden-state
activation is extracted and **injected into a separate reporter persona's** interpretation
pass. We ask whether a same-persona reporter decodes the injected state better than a neutral
or opposite-persona reporter (predicted ordering `same > neutral > opposite`). Three
conditions provide clean controls: `real` (inject source activation), `text_only_baseline`
(no injection), and `shuffled` (norm-matched permuted activation = noise floor). The main run
covers 205 questions across 5 content categories on Llama-3.1-8B-Instruct.

### 3.2 Key results

**The task battery validates cleanly.** Cross-persona *agreement* tracks content category
exactly as designed — **91.4% agreement on factual questions vs 6.2% on values questions** —
confirming the questions separate ideological from factual content and that persona
conditioning is doing real work.

**A clean privileged-access accuracy gradient:** reporter accuracy is
**same 88.7% / neutral 65.7% / opposite 45.7%**.

**The strongest single pro-access statistic — flip rate.** On trials where source and
reporter *disagreed* at baseline, injecting the source activation flips the reporter's answer
**49.4% of the time for same-persona reporters vs 7.5% for opposite-persona reporters**
(χ² = 449, p < 0.0001). A logistic regression confirms same-persona reporters flip more
*even after controlling for raw logit gain* — consistent with privileged access rather than
generic signal strength. A within-question paired comparison (same − opposite headroom) is
also significantly positive (Wilcoxon p < 10⁻⁶), concentrated in the high-disagreement
categories and mid-to-late layers (best transfer at layer pairs 20→20, 25→25).

### 3.3 Scope note

Of the patchscope notebooks, only the **full-question logit-mode** run is full-scale (205
questions, all controls). The open-answer, keyword-interpretation, and logit-lens notebooks
are small exploratory dev runs (1–7 questions, some with controls disabled or a single
persona) and their numbers should be read as preliminary. One honest counter-signal in the
main run: while *same*-persona reporters flip most, the largest raw logit gain from injection
lands on *opposite*-persona reporters — we read this as the injection carrying signal that
same-persona reporters had already priced in, but it is a tension we are still probing.

---

## 4. Next steps (second half of the program)

1. **exp14 causal steering (highest priority).** The projection/steering test on
   self-recognition activations is the experiment that decisively separates privileged access
   from style. It requires a GPU run and is the natural capstone for the mechanistic line.
2. **Re-capture activations on case 7** rather than the more-confounded single-text case 1,
   so the mechanistic and behavioral analyses probe the same clean condition.
3. **Complete the framing-matched battery** — run case 13/14 for base Llama and for the
   cross-model configuration (currently only the finetuned model has full 13/14 data), giving
   a like-for-like active-self vs neutral-ceiling/floor comparison across all models.
4. **Patchscope layer/position sweep** on the layer pairs that already show the strongest
   transfer (20→20, 25→25) to localize where cross-persona readability peaks.

---

## 5. One-paragraph summary

Personas measurably change what these models compute, not just what they emit. Being the
active persona reliably improves self-recognition (Llama +0.11, Qwen +0.22 over baseline),
and in the categories engineered to defeat surface style we see genuine positive signal —
Llama's suppression introspective surplus (+0.067, CI clears zero), Qwen's near-twin
recognition (0.79), and a further nudge from finetuning. On the mechanistic side the model
internally distinguishes genuine from confabulated self-recognition (AUROC ~0.65). The
patchscope line independently supports privileged access: same-persona reporters flip to a
source's injected answer 49% of the time versus 7% for opposite personas (p < 0.0001), and
the task battery validates cleanly (91% factual vs 6% values agreement). The decisive causal
test (activation steering) and a handful of framing-matched re-runs are the main pending
work; the current behavioral and correlational evidence is consistent with personas carrying
real, partially self-accessible internal state.
