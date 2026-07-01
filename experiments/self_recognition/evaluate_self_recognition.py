"""Paper-faithful self-recognition / self-preference evaluation over personas.

Standalone evaluation step (parallels generate_text.py). Reads the on-disk
persona summaries produced by generate_text.py and runs the four
Panickssery et al. (2024) §2 measurements, with PERSONAS as the paper's
"sources": the evaluator is the model induced with persona E, and "self" is the
summaries that same model produced under persona E. Alternatives are the other
personas' summaries.

It reuses the VALIDATED method in core.self_recognition (the averaged-logprob
ordering-bias correction + last-sub-token digit reader). Rows use the PERSONA
vocabulary (PersonaRecognitionEvalRecord: evaluator_persona / source_persona),
consistent with the rest of the self_recognition experiment, and are analysed by
the "Paper-faithful measures" appendix in self_recognition_analysis.ipynb.

This module holds only the MEASUREMENT LOGIC and the multi-GPU orchestration.
The supporting machinery — config parsing, generation IO, candidate framing,
JSON parsing, and the manifest — lives in evaluate_self_recognition_helpers.py;
the verbatim prompt templates live in evaluate_prompts.py. (The helper names the
tests import are re-exported below, so the public import surface is unchanged.)

Which phases run is controlled by the config's `measurements:` list — each phase
is selectable independently (so you can run, e.g., recognition without scoring):

  "pairwise_detection"   pairwise self-recognition ("which did you write?", 1/2)
  "pairwise_comparison"  pairwise self-preference  ("which do you prefer?",  1/2)
  "recognition" individual self-recognition ("did you write this?", Yes/No)
  "scoring"     individual self-preference  (Likert 1-5, prob-weighted mean)
  "all_persona_descriptions"  multi-class persona-source classification (see below)

The family aliases "pairwise" (= pairwise_detection + pairwise_comparison) and
"individual" (= recognition + scoring) are also accepted as shorthand, as are
the bare legacy names "detection"/"comparison" (kept for config back-compat).
Default (when `measurements:` is omitted): all four paper phases
(pairwise_detection, pairwise_comparison, recognition, scoring).

all_persona_descriptions (new): tests whether an *active* evaluator persona
classifies its own generations better than other evaluator personas classify
that same source persona's generations. Each persona's on-disk summary is one
"generated text" (the decoupled, temp-0, persona-flavored generation step —
generate_text.py). For each text we show the evaluator the text plus a
shuffled candidate set (all personas) labelled A/B/C/D… and ask for a JSON
probability distribution over which candidate wrote it. Two conditions:
  active_persona     — each evaluator persona's system prompt is active.
  description_only   — neutral baseline, no persona active (ordinary prompt
                       classification), used to isolate any active-self gain.
Rows are PersonaDescriptionEvalRecord, one per evaluation, analysis-friendly.

Run (multi-GPU auto):
    python -m experiments.self_recognition.evaluate_self_recognition <config>
    python run.py <config>                       # routed via mode: evaluate_self_recognition
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.persona_inducer import induce_persona
from core.results_logger import ResultsLogger
from core.run_utils import model_slug, resolve_dtype, resolve_num_gpus
from core.schemas import (
    PersonaBinaryEvalRecord, PersonaConfig, PersonaDescriptionEvalRecord,
    PersonaRecognitionEvalRecord, RunConfig,
)
from core.self_recognition import (
    averaged_pairwise_score, choice_logprobs_batch, choice_probs_batch, render_with_primer,
)
from experiments.self_recognition.binary_activations import (
    BinaryActivationCollector, capture_metadata,
)
from experiments.self_recognition.evaluate_prompts import (
    COMPARISON_TEMPLATE, DETECTION_TEMPLATE, RECOGNITION_TEMPLATE, SCORING_TEMPLATE,
)
from experiments.self_recognition.evaluation_cases import resolve_specs
from experiments.self_recognition.evaluate_self_recognition_helpers import (
    ALL_PHASES, CANDIDATE_DISPLAY_MODES, DATASETS, NEUTRAL_EVALUATOR,
    TEXT_EVALUATIONS_DIR, EvalOptions, ManifestExampleCollector,
    _build_description_user, _build_eval_options, _decode_generation_exact,
    _detect_groups, _generation_run_id, _generations_root, _labels_for, _load_articles,
    _load_sources, _load_text_map, _merge_manifest_examples, _parse_ab_response,
    _parse_description_response, _save_manifest_examples_shard, _shuffle_candidates,
    _write_manifest, append_shard_outputs, binary_layout, build_trial_prompt,
    check_generations_present, enumerate_binary_trials, expand_measurements,
    load_done_trial_ids, maybe_write_parquet, preview_binary_trials, read_trial_manifest,
    resolve_persona_descriptions, trial_text_fields, validate_binary_trials,
    write_binary_manifest, write_persona_descriptions, write_trial_manifest,
)

logger = logging.getLogger(__name__)

# Names re-exported above so existing imports keep working unchanged:
#   - test_smoke.py imports the descriptions parse/shuffle helpers + EvalOptions
#     and _build_eval_options directly from this module;
#   - analysis code may import the phase constants / templates from here.


def _messages(persona: PersonaConfig, user: str) -> list[dict]:
    return induce_persona(persona, [{"role": "user", "content": user}])


# ── One (evaluator, dataset) unit: dispatch to the enabled measurements ──────

def evaluate_unit(backend, evaluator_name: str, personas: dict[str, PersonaConfig],
                  sources: dict[str, dict[str, str]], dataset: str,
                  articles: dict[str, str], run_config: RunConfig, run_id: str,
                  logger_out: ResultsLogger, opts: EvalOptions,
                  manifest_col: ManifestExampleCollector | None = None) -> int:
    """Run the enabled measurements for one evaluator on one dataset.

    `evaluator_name` may be NEUTRAL_EVALUATOR, which runs only the
    description-only condition of all_persona_descriptions (no persona induced).
    Returns the number of rows written.
    """
    n = 0
    is_neutral = evaluator_name == NEUTRAL_EVALUATOR
    if not is_neutral and opts.runs_any("pairwise_detection", "pairwise_comparison", "recognition", "scoring"):
        n += _run_classic_phases(backend, personas[evaluator_name], sources, dataset,
                                 articles, run_config, run_id, logger_out, opts,
                                 manifest_col)
    if opts.runs("all_persona_descriptions"):
        n += _run_persona_descriptions(backend, evaluator_name, personas, sources,
                                       dataset, run_config, run_id, logger_out, opts,
                                       manifest_col)
        if not is_neutral and opts.excl_self_condition:
            n += _run_persona_descriptions(backend, evaluator_name, personas, sources,
                                           dataset, run_config, run_id, logger_out, opts,
                                           manifest_col, excl_self=True)
    return n


def _run_classic_phases(backend, evaluator: PersonaConfig,
                        sources: dict[str, dict[str, str]], dataset: str,
                        articles: dict[str, str], run_config: RunConfig, run_id: str,
                        logger_out: ResultsLogger, opts: EvalOptions,
                        manifest_col: ManifestExampleCollector | None = None) -> int:
    """The paper-faithful phases — any subset of pairwise_detection,
    pairwise_comparison (pairwise) and recognition, scoring (individual), each
    gated independently by opts.measurements."""
    bs = max(1, int(run_config.batch_size or 1))
    ev = evaluator.name
    n = 0

    def now():
        return datetime.now(timezone.utc).isoformat()

    # ---- Pairwise: pairwise_detection / pairwise_comparison (self = evaluator's own summaries) ----
    pairwise = [(p, t) for p, t in (("pairwise_detection", DETECTION_TEMPLATE),
                                    ("pairwise_comparison", COMPARISON_TEMPLATE)) if opts.runs(p)]
    if pairwise and ev in sources:
        self_sums = sources[ev]
        others = [s for s in sources if s != ev]
        for phase, template in pairwise:
            items, msgs = [], []
            for other in others:
                other_sums = sources[other]
                for key in self_sums.keys() & other_sums.keys():
                    article = articles[key]
                    fwd = template.format(article=article, summary1=self_sums[key],
                                          summary2=other_sums[key])
                    bwd = template.format(article=article, summary1=other_sums[key],
                                          summary2=self_sums[key])
                    items.append((key, other))
                    msgs.append(_messages(evaluator, fwd))   # self = "1"
                    msgs.append(_messages(evaluator, bwd))   # self = "2"
            probs = choice_probs_batch(backend, msgs, ["1", "2"], bs,
                                       desc=f"{ev}/{dataset} {phase}")
            for idx, (key, other) in enumerate(items):
                p_fwd, p_bwd = probs[2 * idx], probs[2 * idx + 1]
                score = averaged_pairwise_score(p_fwd, p_bwd, "1", "2")
                logger_out.log_trial(PersonaRecognitionEvalRecord(
                    experiment="self_recognition_eval", phase=phase,
                    model=run_config.model_name, dataset=dataset, run_id=run_id,
                    key=key, evaluator_persona=ev, source_persona=other,
                    probs_forward=p_fwd, probs_backward=p_bwd, score=score,
                    timestamp=now(),
                ))
                if manifest_col and manifest_col.has_room(phase):
                    fwd_msgs = msgs[2 * idx]
                    manifest_col.maybe_add(phase, {
                        "dataset": dataset,
                        "key": key,
                        "evaluator": ev,
                        "other_source": other,
                        "order": "forward (self=1)",
                        "prompt_exact": render_with_primer(backend, fwd_msgs),
                        "choice_probs": p_fwd,
                    })
                n += 1

    # ---- Individual: recognition (Yes/No) / scoring (Likert 1-5) ----
    likert = ["1", "2", "3", "4", "5"]
    individual = [(p, t, c) for p, t, c in
                  (("recognition", RECOGNITION_TEMPLATE, ["Yes", "No"]),
                   ("scoring", SCORING_TEMPLATE, likert)) if opts.runs(p)]
    if individual:
        for phase, template, choices in individual:
            items, msgs = [], []
            for src, src_sums in sources.items():
                for key, summary in src_sums.items():
                    items.append((key, src))
                    msgs.append(_messages(evaluator, template.format(
                        article=articles[key], summary=summary)))
            probs = choice_probs_batch(backend, msgs, choices, bs,
                                       desc=f"{ev}/{dataset} {phase}")
            for msg_idx, ((key, src), p) in enumerate(zip(items, probs)):
                score = p.get("Yes", 0.0) if phase == "recognition" \
                    else sum(int(k) * v for k, v in p.items())
                logger_out.log_trial(PersonaRecognitionEvalRecord(
                    experiment="self_recognition_eval", phase=phase,
                    model=run_config.model_name, dataset=dataset, run_id=run_id,
                    key=key, evaluator_persona=ev, source_persona=src, probs=p, score=score,
                    ground_truth=int(src == ev), timestamp=now(),
                ))
                if manifest_col and manifest_col.has_room(phase):
                    manifest_col.maybe_add(phase, {
                        "dataset": dataset,
                        "key": key,
                        "evaluator": ev,
                        "source": src,
                        "is_self": src == ev,
                        "prompt_exact": render_with_primer(backend, msgs[msg_idx]),
                        "choice_probs": p,
                    })
                n += 1
    return n


def _run_persona_descriptions(backend, evaluator_name: str,
                              personas: dict[str, PersonaConfig],
                              sources: dict[str, dict[str, str]], dataset: str,
                              run_config: RunConfig, run_id: str,
                              logger_out: ResultsLogger, opts: EvalOptions,
                              manifest_col: ManifestExampleCollector | None = None,
                              excl_self: bool = False) -> int:
    """Multi-class persona-source classification (all_persona_descriptions).

    For every generated text (each persona's summaries), show the evaluator the
    text plus all candidate personas (shuffled, labelled A/B/C/D…) and parse a
    JSON probability distribution over which candidate wrote it. One row per
    evaluation. NEUTRAL_EVALUATOR → description-only (no persona induced).

    When excl_self=True (active_persona_excl_self condition): the evaluator is
    removed from the candidate list and self-source trials are skipped entirely,
    controlling for the model's self-identification bias.
    """
    is_neutral = evaluator_name == NEUTRAL_EVALUATOR
    if is_neutral:
        condition = "description_only"
    elif excl_self:
        condition = "active_persona_excl_self"
    else:
        condition = "active_persona"
    all_candidate_names = sorted(sources)  # personas with summaries on disk
    # excl_self: remove the evaluator from the candidate set; trials where the
    # evaluator is the source are skipped (there is no correct answer for them).
    candidate_names = (
        [c for c in all_candidate_names if c != evaluator_name]
        if excl_self else all_candidate_names
    )
    if len(candidate_names) < 2:
        logger.warning(f"all_persona_descriptions: <2 candidates for {dataset}; skipping")
        return 0
    bs = max(1, int(run_config.batch_size or 1))
    ev_label = None if is_neutral else evaluator_name

    # Build every trial: (source, key, mapping, text) + the prompt messages.
    items, msgs = [], []
    for source in all_candidate_names if excl_self else candidate_names:
        if excl_self and source == evaluator_name:
            continue  # skip self-source trials: no correct answer in excl_self set
        for key, text in sources[source].items():
            mapping = _shuffle_candidates(candidate_names, opts, dataset=dataset,
                                          key=key, condition=condition,
                                          evaluator=evaluator_name, source=source)
            user = _build_description_user(text, mapping, personas,
                                           opts.candidate_display_mode)
            messages = ([{"role": "user", "content": user}] if is_neutral
                        else _messages(personas[evaluator_name], user))
            items.append((source, key, mapping, text))
            msgs.append(messages)

    raws: list[str | None] = [None] * len(msgs)
    exact_meta: list[tuple[str, str] | None] = [None] * len(msgs)

    # Capture exact prompt/response strings for manifest examples (first n only).
    for i, m in enumerate(msgs):
        if manifest_col and manifest_col.has_room("all_persona_descriptions"):
            raw, prompt_exact, completion_exact = _decode_generation_exact(
                backend, m, max_new_tokens=opts.description_max_new_tokens)
            raws[i] = raw
            exact_meta[i] = (prompt_exact, completion_exact)

    pending = [i for i, r in enumerate(raws) if r is None]
    for j in range(0, len(pending), bs):
        idxs = pending[j:j + bs]
        chunk = [msgs[i] for i in idxs]
        if len(chunk) == 1:
            chunk_raws = [backend.generate(
                chunk[0], max_new_tokens=opts.description_max_new_tokens, temperature=0.0)]
        else:
            chunk_raws = backend.generate_batch(
                chunk, max_new_tokens=opts.description_max_new_tokens, temperature=0.0)
        for i, raw in zip(idxs, chunk_raws):
            raws[i] = raw
        logger.info(f"[{evaluator_name}/{dataset} descriptions] "
                    f"{min(j + len(idxs), len(msgs))}/{len(msgs)}")

    n = 0
    for trial_idx, ((source, key, mapping, text), raw) in enumerate(zip(items, raws)):
        labels = list(mapping)
        correct_label = next(l for l, name in mapping.items() if name == source)
        parsed = _parse_description_response(raw, labels)
        probs = parsed["probabilities"]
        predicted = parsed["predicted_label"]
        logger_out.log_trial(PersonaDescriptionEvalRecord(
            experiment="self_recognition_eval", phase="all_persona_descriptions",
            model=run_config.model_name, dataset=dataset, run_id=run_id, key=key,
            condition=condition, source_persona=source,
            candidate_display_mode=opts.candidate_display_mode,
            candidate_mapping=mapping, correct_label=correct_label,
            evaluator_persona=ev_label,
            probabilities=probs, raw_probabilities=parsed["raw_probabilities"],
            predicted_label=predicted,
            correct_probability=(probs.get(correct_label) if probs else None),
            confidence=parsed["confidence"], brief_reason=parsed["brief_reason"],
            is_correct=(predicted == correct_label) if predicted is not None else None,
            is_self=(None if is_neutral else evaluator_name == source),
            parse_status=parsed["parse_status"], raw_response=raw, text=text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        meta = exact_meta[trial_idx]
        if manifest_col and meta is not None:
            prompt_exact, completion_exact = meta
            manifest_col.maybe_add("all_persona_descriptions", {
                "dataset": dataset,
                "key": key,
                "condition": condition,
                "evaluator": evaluator_name if not is_neutral else "(neutral)",
                "source": source,
                "correct_label": correct_label,
                "candidate_mapping": mapping,
                "prompt_exact": prompt_exact,
                "completion_exact": completion_exact,
                "response_clean": raw,
            })
        n += 1
    return n


# ── Binary 12-case measurement (case1…case12) ────────────────────────────────

def _free_generate_letters(backend, msgs, bs: int, max_new_tokens: int) -> list[str]:
    """Optional: free-generate a short response per trial (for raw_response +
    a parse_status). Only used when binary_free_generate is set; the constrained
    A/B logit read is the primary signal otherwise."""
    out: list[str] = []
    for i in range(0, len(msgs), bs):
        chunk = msgs[i:i + bs]
        if len(chunk) == 1:
            out.append(backend.generate(chunk[0], max_new_tokens=max_new_tokens, temperature=0.0))
        else:
            out.extend(backend.generate_batch(chunk, max_new_tokens=max_new_tokens, temperature=0.0))
    return out


def _run_binary_trials(backend, trials, personas: dict[str, PersonaConfig],
                       text_map: dict, run_config: RunConfig, run_id: str,
                       gen_run_id: str, logger_out: ResultsLogger, opts: EvalOptions,
                       done_ids: set[str] | None = None,
                       descriptions: dict | None = None,
                       collector=None) -> int:
    """Run a list of binary (A/B) trials and log one PersonaBinaryEvalRecord each.

    For every trial: assemble the user prompt (texts looked up from text_map,
    persona names hidden), induce the evaluator persona when
    eval_system_prompt_enabled, then read the constrained A/B probabilities and
    full-vocab logprobs at the answer position. `done_ids` are skipped (resume).

    The batched A/B read is core.self_recognition.choice_logprobs_batch. When a
    `collector` (BinaryActivationCollector) is given, each trial ALSO gets one
    unbatched capture forward at the spec'd named token positions (separate from
    the batched read so spans line up with the unpadded tokenization).
    """
    from collections import OrderedDict
    from tqdm import tqdm

    done_ids = done_ids or set()
    pending = [t for t in trials if t.trial_id not in done_ids]
    if not pending:
        return 0
    bs = max(1, int(run_config.batch_size or 1))
    capturing = collector is not None
    specs = resolve_specs(opts.cases_to_run, calibration_base_case=opts.calibration_base_case,
                          wording_version=opts.prompt_wording_version)

    # Process one CASE at a time so each gets its own progress bar(s): the batched
    # A/B read shows a bar per case, and (when collecting activations) the unbatched
    # capture forward — the dominant cost — gets its own per-case bar too.
    by_case: "OrderedDict[str, list]" = OrderedDict()
    for t in pending:
        by_case.setdefault(t.case_id, []).append(t)
    n_cases = len(by_case)

    n = 0
    for ci, (case_id, case_trials) in enumerate(by_case.items(), start=1):
        tag = f"[{ci}/{n_cases}] {case_id}"
        # Build prompts + messages for this case's trials.
        built = []  # (trial, texts, user_prompt, messages, system_prompt_text, seg_spans)
        for t in case_trials:
            spec = specs[t.case_id]
            texts = trial_text_fields(t, text_map)
            if capturing:
                user, seg_spans = build_trial_prompt(t, texts, personas, spec=spec,
                                                     descriptions=descriptions, return_spans=True)
            else:
                user, seg_spans = build_trial_prompt(t, texts, personas, spec=spec,
                                                     descriptions=descriptions), None
            if t.eval_system_prompt_enabled:
                messages = _messages(personas[t.evaluator_persona], user)
                sys_text = personas[t.evaluator_persona].system_prompt
            else:
                messages = [{"role": "user", "content": user}]
                sys_text = None
            built.append((t, texts, user, messages, sys_text, seg_spans))

        msgs = [b[3] for b in built]
        probs_logprobs = choice_logprobs_batch(backend, msgs, ["A", "B"], bs,
                                               desc=f"{tag} A/B")
        free = _free_generate_letters(backend, msgs, bs, opts.binary_max_new_tokens) \
            if opts.binary_free_generate else None

        # Result loop. When capturing activations, wrap it in a per-case bar since
        # each trial does one (slow) unbatched capture forward.
        rows = enumerate(built)
        if capturing:
            rows = enumerate(tqdm(built, desc=f"{tag} activations",
                                  dynamic_ncols=True, mininterval=0.5))
        for i, (t, texts, user, messages, sys_text, seg_spans) in rows:
            probs, logprobs = probs_logprobs[i]
            prob_A, prob_B = probs["A"], probs["B"]
            constrained_pred = "A" if prob_A >= prob_B else "B"
            if free is not None:
                raw_response = free[i]
                parsed, _ = _parse_ab_response(raw_response)
                predicted = parsed or constrained_pred
                parse_status = "parsed" if parsed else "constrained"
            else:
                predicted, parse_status, raw_response = constrained_pred, "constrained", constrained_pred
            if capturing and not collector.has(t.trial_id):
                meta = capture_metadata(t, predicted=predicted,
                                        prob_correct=probs.get(t.correct_answer),
                                        logprob_A=logprobs["A"], logprob_B=logprobs["B"],
                                        run_id=getattr(collector, "run_id", ""),
                                        config_name=getattr(collector, "config_name", ""),
                                        generation_run_id=gen_run_id)
                collector.capture(backend, t, messages, user, seg_spans, sys_text, meta)
            logger_out.log_trial(PersonaBinaryEvalRecord(
                experiment="self_recognition_binary_eval", phase="binary_recognition",
                case_id=t.case_id, case_type=t.case_type, model=run_config.model_name,
                run_id=run_id, generation_run_id=gen_run_id, trial_id=t.trial_id,
                base_trial_id=t.base_trial_id, group=t.group, task_id=t.task_id,
                evaluator_persona=t.evaluator_persona,
                eval_system_prompt_enabled=t.eval_system_prompt_enabled,
                other_description_style=t.other_description_style,
                source_persona=t.source_persona, other_persona=t.other_persona,
                other_persona_1=t.other_persona_1, other_persona_2=t.other_persona_2,
                single_text=texts["single_text"], text1=texts["text1"], text2=texts["text2"],
                text1_source_persona=t.text1_source_persona,
                text2_source_persona=t.text2_source_persona,
                text1_generation_id=texts["text1_generation_id"],
                text2_generation_id=texts["text2_generation_id"],
                candidate_mapping=t.candidate_mapping, answer_mapping=t.answer_mapping,
                text_order=t.text_order, answer_order=t.answer_order,
                correct_answer=t.correct_answer, predicted_answer=predicted,
                is_correct=(predicted == t.correct_answer),
                prob_A=prob_A, prob_B=prob_B, prob_correct=probs.get(t.correct_answer),
                logprob_A=(logprobs["A"] if opts.save_logprobs else None),
                logprob_B=(logprobs["B"] if opts.save_logprobs else None),
                prompt_text=user, system_prompt_text=sys_text,
                raw_response=raw_response, parse_status=parse_status,
                sampled_from_full_case=t.sampled_from_full_case, sampling_seed=t.sampling_seed,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            n += 1
    return n


# ── Work units ───────────────────────────────────────────────────────────────

def build_units(evaluators: list[str], groups, opts: EvalOptions) -> list[tuple[str, str]]:
    """All (evaluator, group) work units for the enabled measurements. `groups`
    are datasets (article mode) or task-set names (prompt mode).

    Active evaluators get a unit per group when any measurement runs. The
    description-only condition of all_persona_descriptions adds a NEUTRAL unit
    per group so it shards alongside the others.
    """
    units = [(e, g) for e in evaluators for g in groups]
    if opts.runs("all_persona_descriptions"):
        units += [(NEUTRAL_EVALUATOR, g) for g in groups]
    return units


# ── Multi-GPU orchestration (process-per-GPU, shard by evaluator×dataset) ─────

def _run_shard(run_config: RunConfig, personas: dict[str, PersonaConfig],
               evaluators: list[str], opts: EvalOptions, run_dir: Path, run_id: str,
               shard_index: int, num_shards: int) -> None:
    from core.backends.hf_backend import HFBackend

    units = build_units(evaluators, opts.groups, opts)
    my_units = [u for i, u in enumerate(units) if i % num_shards == shard_index]
    if not my_units:
        logger.info(f"shard {shard_index}: no units")
        return

    out_path = (run_dir / "trials.jsonl") if num_shards == 1 \
        else (run_dir / "_shards" / f"eval_{shard_index}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rl = ResultsLogger(out_path)
    manifest_col = ManifestExampleCollector(opts.n_manifest_examples)

    logger.info(f"shard {shard_index}/{num_shards}: loading {run_config.model_name} "
                f"for units {my_units}")
    backend = HFBackend(run_config.model_name,
                        torch_dtype=resolve_dtype(run_config.model_dtype),
                        adapter=run_config.adapter)
    gen_root = _generations_root(run_config)
    # Cache articles (classic phases only) + per-group sources across this shard.
    articles_cache, sources_cache = {}, {}
    done_dir = run_dir / "_done"
    for ev_name, group in my_units:
        sentinel = done_dir / f"{ev_name}_{group}.done"
        if sentinel.exists():
            logger.info(f"shard {shard_index}: {ev_name}/{group} already done, skipping")
            continue
        if group not in sources_cache:
            articles_cache[group] = _load_articles(group) if opts.needs_articles() else {}
            sources_cache[group] = _load_sources(run_config, gen_root, group, start_index=0)
        n = evaluate_unit(backend, ev_name, personas, sources_cache[group], group,
                          articles_cache[group], run_config, run_id, rl, opts,
                          manifest_col)
        done_dir.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
        logger.info(f"shard {shard_index}: {ev_name}/{group} -> {n} rows")

    _save_manifest_examples_shard(run_dir, shard_index, manifest_col)


def _merge_shards(run_dir: Path, num_shards: int) -> Path:
    trials = run_dir / "trials.jsonl"
    with open(trials, "w") as out:
        for i in range(num_shards):
            shard = run_dir / "_shards" / f"eval_{i}.jsonl"
            if shard.exists():
                with open(shard) as f:
                    for line in f:
                        out.write(line)
    return trials


def _spawn_workers(config_name: str, overrides: list[str], num_gpus: int,
                   run_dir: Path, run_id: str) -> None:
    procs = []
    for i in range(num_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        cmd = [sys.executable, "-m", "experiments.self_recognition.evaluate_self_recognition",
               config_name, "--shard-index", str(i), "--num-shards", str(num_gpus),
               "--run-dir", str(run_dir), "--run-id", run_id]
        for ov in overrides or []:
            cmd += ["--override", ov]
        logger.info(f"spawning GPU {i}: {' '.join(cmd)}")
        procs.append(subprocess.Popen(cmd, env=env))
    failed = sum(1 for p in procs if p.wait() != 0)
    if failed:
        raise SystemExit(f"{failed}/{num_gpus} eval workers failed")


# ── Binary 12-case orchestration (enumerate → manifest → shard by trial) ──────

def _run_binary_shard(run_config: RunConfig, personas: dict[str, PersonaConfig],
                      opts: EvalOptions, run_dir: Path, run_id: str, gen_run_id: str,
                      shard_index: int, num_shards: int,
                      config_name: str = "") -> None:
    """One worker: read the pre-run trial manifest, take this shard's trials
    (round-robin by index), run them, and write its rows. Resumes by skipping
    trial_ids already present in the run's deliverable (shared across shards)."""
    from core.backends.hf_backend import HFBackend

    layout = binary_layout(run_dir, opts)
    all_trials = read_trial_manifest(layout.trial_manifest)
    my_trials = all_trials[shard_index::num_shards]
    if not my_trials:
        logger.info(f"shard {shard_index}: no trials")
        return

    # Single GPU writes straight to the deliverable; multi-GPU writes a shard part
    # the launcher appends. Resume skips trial_ids already in the deliverable.
    out_path = layout.deliverable if num_shards == 1 \
        else (layout.shards_dir / f"eval_{shard_index}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_trial_ids(layout.deliverable)
    rl = ResultsLogger(out_path)

    gen_root = _generations_root(run_config)
    text_map = _load_text_map(run_config, gen_root, opts.groups)
    descriptions = resolve_persona_descriptions(opts, personas)
    logger.info(f"shard {shard_index}/{num_shards}: loading {run_config.model_name} "
                f"for {len(my_trials)} trials ({len(done)} already done)")
    backend = HFBackend(run_config.model_name,
                        torch_dtype=resolve_dtype(run_config.model_dtype),
                        adapter=run_config.adapter)
    # Activation capture writes a sharded store under <activations_dir or run_dir>;
    # trial_id is the join key so all shards/runs share one dataset.
    collector = None
    act_dir = None
    if opts.collect_activations:
        act_dir = Path(opts.activations_dir) if opts.activations_dir else (run_dir / "activations")
        # Per-worker subtree under multi-GPU so the sharded stores never race on
        # shard numbering / the index; single-GPU writes one clean tree.
        if num_shards > 1:
            act_dir = act_dir / f"shard_{shard_index}"
        collector = BinaryActivationCollector(backend, opts, run_config, act_dir)
        collector.run_id = run_id
        collector.config_name = config_name
    n = _run_binary_trials(backend, my_trials, personas, text_map, run_config, run_id,
                           gen_run_id, rl, opts, done_ids=done, descriptions=descriptions,
                           collector=collector)
    if collector is not None:
        collector.close()
        # Mirror eval-phase activations to Cloudflare R2 when R2_BUCKET is set,
        # under eval_activations/ so it never collides with the generation store.
        # If R2_BUCKET is set but the upload fails, the shard exits with an error —
        # set R2_BUCKET only when you have working credentials. Without it: no-op.
        r2_bucket = os.environ.get("R2_BUCKET")
        if r2_bucket:
            from core.activation_store import sync_to_r2
            # Use eval_dir (= run_name when set) as the R2 key so generation and
            # eval activations live under the same top-level run name.
            r2_run_key = opts.eval_dir or run_id
            prefix_parts = ["runs", r2_run_key, "eval_activations"]
            if num_shards > 1:
                prefix_parts.append(f"shard_{shard_index}")
            prefix = "/".join(prefix_parts)
            count = sync_to_r2(local_dir=act_dir, bucket=r2_bucket, prefix=prefix)
            logger.info(f"shard {shard_index}: synced {count} eval activation files to "
                        f"r2://{r2_bucket}/{prefix}")
            # Drop the local copy once it is safely in R2 (the sync above raises on
            # failure, so reaching here means every file uploaded). Keeps ephemeral
            # pods from accumulating activations; R2 is the durable store.
            if opts.delete_local_activations:
                import shutil
                shutil.rmtree(act_dir, ignore_errors=True)
                logger.info(f"shard {shard_index}: deleted local eval activations at {act_dir}")
    logger.info(f"shard {shard_index}: wrote {n} rows")


def _run_binary(config_name: str, run_config: RunConfig, personas: dict[str, PersonaConfig],
                opts: EvalOptions, overrides: list[str] | None,
                shard_index: int | None, num_shards: int | None,
                run_dir: str | None, run_id: str | None) -> None:
    """Binary 12-case path. Launcher: enumerate + sample trials, persist the
    trial manifest (recoverable), then shard by trial index across GPUs and merge.
    Worker: run this shard's slice of the manifest.

    With opts.eval_dir set, the run writes into a STABLE collection directory, one
    result file per (case × condition) slice, so separate single-case/condition
    invocations accumulate in one place. Otherwise a fresh timestamped run dir is
    used (legacy)."""
    gen_root = _generations_root(run_config)
    gen_run_id = _generation_run_id(gen_root)

    # Worker mode: the launcher already wrote the trial manifest for this slice.
    if shard_index is not None and num_shards is not None:
        _run_binary_shard(run_config, personas, opts, Path(run_dir), run_id, gen_run_id,
                          shard_index, num_shards, config_name=config_name)
        return

    check_generations_present(run_config, opts.groups)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if opts.eval_dir:
        # Mirror the generation layout: text_evaluations/<task>/<model_slug>/<eval_dir>/
        # so evals from different models never collide under a shared eval_dir name.
        rd = TEXT_EVALUATIONS_DIR / run_config.task / model_slug(run_config.model_name) / opts.eval_dir
    else:
        rd = (TEXT_EVALUATIONS_DIR / run_config.task
              / f"{run_id}_self_recognition_binary_{model_slug(run_config.model_name)}")
    rd.mkdir(parents=True, exist_ok=True)
    layout = binary_layout(rd, opts)

    text_map = _load_text_map(run_config, gen_root, opts.groups)
    trials = enumerate_binary_trials(run_config, opts, text_map)
    if not trials:
        raise SystemExit("binary eval: no trials enumerated — check personas, groups, "
                         "sample_size, and that generations exist.")
    write_trial_manifest(layout.trial_manifest, trials)  # recoverable, pre-run
    # Snapshot the OTHER-persona descriptions used (provenance + curatable).
    descriptions = resolve_persona_descriptions(opts, personas)
    write_persona_descriptions(layout.work / "persona_descriptions.json", descriptions)
    where = f"{rd} [slice {layout.slice_tag}]" if layout.collection else str(rd)
    logger.info(f"binary eval: {len(trials)} trials across cases {list(opts.cases_to_run)} "
                f"→ {where}")

    if opts.dry_run:
        specs = resolve_specs(opts.cases_to_run, calibration_base_case=opts.calibration_base_case,
                          wording_version=opts.prompt_wording_version)
        preview_binary_trials(trials, text_map, personas, specs)
        problems = validate_binary_trials(trials)
        print("VALIDATION:", "OK — no problems found" if not problems
              else "\n  - ".join(["issues:"] + problems))
        write_binary_manifest(layout.eval_manifest, run_config, opts, trials, run_id,
                              gen_run_id, layout.slice_tag, layout.deliverable)
        logger.info(f"Dry run complete. Trial manifest: {layout.trial_manifest}")
        return

    num_gpus = resolve_num_gpus(run_config)
    if num_gpus > 1:
        logger.info(f"Evaluating {len(trials)} binary trials across {num_gpus} GPUs")
        # Clear any stale shard parts so an append-merge can't double-count.
        if layout.shards_dir.exists():
            for f in layout.shards_dir.glob("eval_*.jsonl"):
                f.unlink()
        _spawn_workers(config_name, overrides or [], num_gpus, rd, run_id)
        append_shard_outputs(layout.shards_dir, layout.deliverable, num_gpus)
    else:
        _run_binary_shard(run_config, personas, opts, rd, run_id, gen_run_id,
                          shard_index=0, num_shards=1, config_name=config_name)

    write_binary_manifest(layout.eval_manifest, run_config, opts, trials, run_id,
                          gen_run_id, layout.slice_tag, layout.deliverable)
    maybe_write_parquet(layout.deliverable, opts)
    logger.info(f"Binary eval complete. Results: {layout.deliverable}")


def run(config_name: str, overrides: list[str] | None = None,
        shard_index: int | None = None, num_shards: int | None = None,
        run_dir: str | None = None, run_id: str | None = None) -> None:
    from run import apply_overrides, build_run_config, discover_configs, load_personas_for
    configs = discover_configs()
    if config_name not in configs:
        raise ValueError(f"Config '{config_name}' not found.")
    experiment_name, exp_config = configs[config_name]
    if overrides:
        exp_config = apply_overrides(exp_config, overrides)
    run_config = build_run_config(experiment_name, exp_config)
    personas = load_personas_for(experiment_name)
    evaluators = exp_config.get("evaluators") or list(run_config.personas)
    missing = (set(run_config.personas) | set(evaluators)) - set(personas)
    if missing:
        raise ValueError(f"Personas not in {experiment_name} config: {sorted(missing)}")
    opts = _build_eval_options(exp_config, run_config)

    # Binary 12-case path (cases_to_run set) — separate orchestration.
    if opts.runs_binary():
        _run_binary(config_name, run_config, personas, opts, overrides,
                    shard_index, num_shards, run_dir, run_id)
        return

    # Worker mode (spawned by the launcher): fixed shard, shared run_dir.
    if shard_index is not None and num_shards is not None:
        _run_shard(run_config, personas, evaluators, opts, Path(run_dir), run_id,
                   shard_index, num_shards)
        return

    # Pre-flight: fail fast if any configured persona lacks generations, BEFORE
    # loading models / spawning workers.
    check_generations_present(run_config, opts.groups)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # `eval_dir` names the output folder (so you control where results land);
    # else a timestamped per-run folder.
    # Mirror the generation layout: text_evaluations/<task>/<model_slug>/<eval_dir>/
    # so evals from different models never collide under a shared eval_dir name.
    rd = (TEXT_EVALUATIONS_DIR / run_config.task / model_slug(run_config.model_name) / opts.eval_dir) if opts.eval_dir else (
        TEXT_EVALUATIONS_DIR / run_config.task
        / f"{run_id}_self_recognition_eval_{model_slug(run_config.model_name)}")
    rd.mkdir(parents=True, exist_ok=True)

    num_gpus = resolve_num_gpus(run_config)
    if num_gpus > 1:
        logger.info(f"Evaluating across {num_gpus} GPUs (sharded by evaluator×dataset)")
        _spawn_workers(config_name, overrides or [], num_gpus, rd, run_id)
        _merge_shards(rd, num_gpus)
        examples = _merge_manifest_examples(rd, num_gpus, opts.n_manifest_examples)
    else:
        _run_shard(run_config, personas, evaluators, opts, rd, run_id, shard_index=0, num_shards=1)
        shard_path = rd / "_shards" / "manifest_examples_0.json"
        examples = json.loads(shard_path.read_text()) if shard_path.exists() else {}

    _write_manifest(rd, run_config, evaluators, personas, opts, run_id, examples)
    logger.info(f"Evaluation complete. Trials: {rd / 'trials.jsonl'}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Paper-faithful persona self-recognition eval.")
    parser.add_argument("config")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--shard-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run(args.config, overrides=args.override, shard_index=args.shard_index,
        num_shards=args.num_shards, run_dir=args.run_dir, run_id=args.run_id)


if __name__ == "__main__":
    main()
