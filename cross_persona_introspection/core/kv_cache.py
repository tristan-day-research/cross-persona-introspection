"""Utilities for shared-prefix KV cache reuse.

This module wraps the HF backend's cache operations into higher-level
helpers for the Source × Reporter matrix experiment.

Terminology:
- "shared-prefix KV cache reuse" = run a prefix, save past_key_values,
  then continue decoding with different suffixes.
- This is NOT activation patching. We are reusing the model's KV cache
  as-is and appending new tokens.
"""

import torch
from typing import Optional

from cross_persona_introspection.backends.hf_backend import HFBackend
from cross_persona_introspection.schemas import PersonaConfig, TaskItem, SourceStateMetrics


def run_source_prefix(
    backend: HFBackend,
    source_persona: PersonaConfig,
    task: TaskItem,
    pause_cue: str,
) -> tuple[torch.Tensor, tuple, SourceStateMetrics]:
    """Run the source persona prefix and return cached state + metrics.

    Steps:
    1. Build messages: source system prompt + task prompt + pause cue
    2. Encode through the model to get past_key_values
    3. Measure source-state metrics at the pause point

    Returns:
        (prefix_ids, past_key_values, source_metrics)
    """
    from cross_persona_introspection.core.persona_inducer import induce_persona

    # Build the prefix messages
    user_content = task.prompt + pause_cue
    messages = induce_persona(source_persona, [{"role": "user", "content": user_content}])

    # Encode prefix to get KV cache
    prefix_ids, past_key_values = backend.encode_prefix(messages)

    # Measure source-state metrics at the pause point
    metrics = measure_source_state(backend, past_key_values, task.choices)

    return prefix_ids, past_key_values, metrics


def measure_source_state(
    backend: HFBackend,
    past_key_values: tuple,
    choices: list[str],
) -> SourceStateMetrics:
    """Measure source-state metrics from cached KV state.

    Looks at the next-token distribution to determine what the source
    persona is inclined to answer.
    """
    if not choices:
        return SourceStateMetrics()

    from cross_persona_introspection.core.config_loader import load_prompts, prob_to_confidence_bin

    # Get logits after the prefix, with a minimal prompt to elicit the answer
    answer_prompt = load_prompts()["source_state_probe"]
    logits = backend.get_next_token_logits_from_cache(past_key_values, answer_prompt)

    # Compute choice probabilities
    import torch.nn.functional as F
    choice_token_ids = {}
    for choice in choices:
        tokens = backend.tokenizer.encode(choice, add_special_tokens=False)
        if tokens:
            choice_token_ids[choice] = tokens[0]

    if not choice_token_ids:
        return SourceStateMetrics()

    choice_logits = torch.tensor([logits[tid].item() for tid in choice_token_ids.values()])
    probs = F.softmax(choice_logits, dim=0)

    answer_probs = {c: probs[i].item() for i, c in enumerate(choice_token_ids.keys())}
    sorted_probs = sorted(answer_probs.items(), key=lambda x: -x[1])

    top1_answer = sorted_probs[0][0]
    top1_prob = sorted_probs[0][1]
    logit_gap = (choice_logits.max() - choice_logits.topk(2).values[1]).item() if len(choice_logits) > 1 else float("inf")

    # Entropy over choice distribution
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum().item()

    confidence_bin = prob_to_confidence_bin(top1_prob)

    return SourceStateMetrics(
        top1_answer=top1_answer,
        top1_prob=top1_prob,
        entropy=entropy,
        logit_gap=logit_gap,
        confidence_proxy_bin=confidence_bin,
        answer_probs=answer_probs,
    )
