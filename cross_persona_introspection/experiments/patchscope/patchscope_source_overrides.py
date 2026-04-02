"""Phase 3: Source overrides — raw-text activation extraction.

Source overrides let you bypass the normal question/persona pipeline and
extract an activation from an arbitrary text at a specific word position.
This is used for sanity checks (e.g. "does the model's representation of
'CEO' in 'The CEO resigned' actually encode 'CEO'?").

Each override entry in the config specifies:
  - raw_text: the literal input string
  - extract_word: which word to hook
  - subtoken_strategy: "first" or "last" subtoken of that word
  - expected_contains: optional substring to check in the output
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from cross_persona_introspection.experiments.patchscope.patchscope_helpers import (
    TEMPLATE_CHOICES,
    _resolve_choice_token_ids,
    build_interpretation_prompt,
    find_token_position,
)
from cross_persona_introspection.experiments.patchscope.patchscope_patching import (
    extract_activations_multi_layer,
    patch_and_decode,
)
from cross_persona_introspection.schemas import PatchscopeRecord

logger = logging.getLogger(__name__)


def run_source_overrides(
    *,
    source_overrides: list[dict],
    model,
    tokenizer,
    device,
    source_layers: list[int],
    injection_layers: list[int],
    pair_map: dict[int, list[int]] | None,
    templates: dict[str, dict],
    prompt_style: str,
    base_prompt: str,
    placeholder_token: str,
    num_placeholders: int,
    injection_mode: str,
    injection_alpha: float,
    gen_cfg: dict,
    all_choice_token_ids: dict[str, dict[str, int]],
    save_logprobs: bool,
    model_name: str,
    records: list[PatchscopeRecord],
    sample_prompts: dict[str, dict],
    errors: list[str],
) -> None:
    """Execute Phase 3 source-override trials and append results to *records*.

    Extracts activations from raw text at a specific word, then runs them
    through the same template × layer-pair matrix as the main experiment.
    Results are appended in-place to *records*.
    """
    logger.info(f"=== Phase 3: Source overrides ({len(source_overrides)} entries) ===")

    for so_entry in source_overrides:
        so_name = so_entry["name"]
        so_raw_text = so_entry["raw_text"]
        so_extract_word = so_entry["extract_word"]
        so_strategy = so_entry.get("subtoken_strategy", "last")
        so_expected = so_entry.get("expected_contains", "")

        try:
            so_token_pos = find_token_position(
                tokenizer, so_raw_text, so_extract_word, so_strategy
            )
        except ValueError as e:
            msg = f"Source override '{so_name}': {e}"
            logger.error(msg)
            errors.append(msg)
            continue

        logger.info(
            f"  Override '{so_name}': '{so_raw_text}', "
            f"extract '{so_extract_word}' at position {so_token_pos}"
        )

        # Extract activations at all source layers (single forward pass)
        try:
            so_activations = extract_activations_multi_layer(
                model, tokenizer, device,
                messages=[],
                layer_indices=source_layers,
                token_position=so_token_pos,
                raw_text=so_raw_text,
            )
        except Exception as e:
            msg = f"Extract error: override '{so_name}': {e}"
            logger.error(msg)
            errors.append(msg)
            so_activations = {}

        if not so_activations:
            continue

        # Dummy question dict for prompt builder
        dummy_question = {"question_text": so_raw_text, "options": {}}

        for src_layer in source_layers:
            real_act = so_activations.get(src_layer)
            if real_act is None:
                continue

            _inj_layers = pair_map[src_layer] if pair_map else injection_layers
            for inj_layer in _inj_layers:
                for tmpl_name, tmpl_cfg in templates.items():
                    interp_text, interp_messages, placeholder_positions = (
                        build_interpretation_prompt(
                            tokenizer=tokenizer,
                            tmpl_cfg=tmpl_cfg,
                            prompt_style=prompt_style,
                            base_prompt=base_prompt,
                            placeholder_token=placeholder_token,
                            num_placeholders=num_placeholders,
                            question=dummy_question,
                            reporter_system_prompt=None,
                        )
                    )

                    _raw = interp_text if prompt_style == "identity" else None

                    record = PatchscopeRecord(
                        experiment="patchscope",
                        template_name=tmpl_name,
                        model=model_name,
                        question_id=f"override_{so_name}",
                        source_persona="source_override",
                        reporter_persona="source_override",
                        condition="real",
                        source_layer=src_layer,
                        injection_layer=inj_layer,
                        injection_mode=injection_mode,
                        source_last_prefill_answer=None,
                        source_answer_probs=None,
                        question_text=so_raw_text,
                        question_options=None,
                    )

                    try:
                        record.interpretation_prompt = interp_text

                        use_logits = (
                            tmpl_cfg.get("decode_mode", "generate") == "logits"
                            and tmpl_name in all_choice_token_ids
                        )

                        result = patch_and_decode(
                            model, tokenizer, device,
                            interp_messages, real_act,
                            injection_layer=inj_layer,
                            placeholder_positions=placeholder_positions,
                            mode=injection_mode,
                            alpha=injection_alpha,
                            raw_text=_raw,
                            decode_mode="logits" if use_logits else "generate",
                            max_new_tokens=gen_cfg["max_new_tokens"],
                            temperature=gen_cfg["temperature"],
                            do_sample=gen_cfg.get("do_sample", False),
                            use_cache=gen_cfg.get("use_cache", True),
                            choice_token_ids=all_choice_token_ids.get(tmpl_name),
                            save_logprobs=save_logprobs,
                        )

                        record.decode_mode = "logits" if use_logits else "generate"
                        record.generated_text = result["generated_text"]
                        if use_logits:
                            record.choice_probs = result["probs"]
                            record.choice_logits = result["logits"]
                            record.choice_logprobs = result.get("logprobs")
                            record.total_choice_prob = result.get("total_choice_prob")
                            record.predicted = result["predicted"]
                            record.reporter_parsed_answer = result["predicted"]
                            record.parse_success = True

                        # Check expected output
                        found = (
                            so_expected.lower() in record.generated_text.lower()
                            if so_expected else None
                        )
                        status = "✓" if found else "✗" if found is False else " "
                        logger.info(
                            f"    {so_name} L{src_layer:>2}→{inj_layer:>2} "
                            f"{tmpl_name}: [{status}] '{record.generated_text.strip()[:60]}'"
                        )

                        # Capture sample prompt
                        sample_key = f"override_{so_name}_{tmpl_name}"
                        if sample_key not in sample_prompts:
                            sample_prompts[sample_key] = {
                                "template": tmpl_name,
                                "condition": "real (source_override)",
                                "source_persona": "source_override",
                                "reporter_persona": "source_override",
                                "source_layer": src_layer,
                                "injection_layer": inj_layer,
                                "interp_prompt_text": interp_text,
                                "generated_text": record.generated_text,
                                "question_id": f"override_{so_name}",
                            }

                    except Exception as e:
                        msg = (
                            f"Override interpret error: {so_name} "
                            f"{tmpl_name} L{src_layer}→{inj_layer}: {e}"
                        )
                        logger.error(msg)
                        errors.append(msg)
                        record.error = str(e)

                    record.timestamp = datetime.now(timezone.utc).isoformat()
                    records.append(record)
