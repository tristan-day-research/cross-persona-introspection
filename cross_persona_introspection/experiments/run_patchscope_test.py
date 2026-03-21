#!/usr/bin/env python3
"""Patchscope test — reproduces the canonical example from the paper.

Verifies the core extract→inject→decode pipeline works by reproducing
Figure 1 from Patchscopes (Ghandeharioun et al., ICML 2024, arXiv:2401.06102):

  Source: "Amazon's former CEO attended Oscars"
  Target: "cat -> cat; 135 -> 135; hello -> hello; ?"
  Patch:  hidden state of "CEO" → "?" in target
  Expect: "Jeff Bezos" or similar CEO-related tokens

Usage:
    python -m cross_persona_introspection.experiments.run_patchscope_test
    python -m cross_persona_introspection.experiments.run_patchscope_test --config patchscope_test.yaml
    python -m cross_persona_introspection.experiments.run_patchscope_test --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress the max_new_tokens/max_length warning
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results" / "raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the core hook functions from the patchscope experiment
sys.path.insert(0, str(ROOT))
from cross_persona_introspection.experiments.patchscope import (
    _get_transformer_layers,
)


# ── Core functions (raw text versions, no chat template) ─────────────


def find_token_position(tokenizer, text: str, word: str, strategy: str = "last") -> int:
    """Find the token position of a word in tokenized text.

    Args:
        tokenizer: HuggingFace tokenizer.
        text: Full input text.
        word: Word to find (e.g., "CEO").
        strategy: "last" uses the last subtoken of the word, "first" uses the first.

    Returns:
        Token position (0-indexed).
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Find contiguous span that decodes to the target word
    word_ids = tokenizer.encode(word, add_special_tokens=False)

    # Search for the subsequence
    for i in range(len(token_ids) - len(word_ids) + 1):
        if token_ids[i : i + len(word_ids)] == word_ids:
            if strategy == "last":
                return i + len(word_ids) - 1
            else:
                return i

    # Fallback: search by decoded string matching
    for i, tok_str in enumerate(tokens):
        if word.lower() in tok_str.lower().strip():
            return i

    raise ValueError(
        f"Could not find '{word}' in tokenized text. "
        f"Tokens: {list(enumerate(tokens))}"
    )


def extract_activation_raw(
    model, tokenizer, device, text: str, layer_idx: int, token_position: int
) -> torch.Tensor:
    """Extract activation from raw text (no chat template) at a specific token position."""
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)

    captured = {}
    layers = _get_transformer_layers(model)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["activation"] = hidden[0, token_position, :].detach().clone()

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    return captured["activation"]


def inject_and_generate_raw(
    model,
    tokenizer,
    device,
    text: str,
    activation: torch.Tensor,
    injection_layer: int,
    inject_token_position: int,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> tuple[str, list[str]]:
    """Inject activation into raw text at a specific position and generate.

    Disables KV caching so the hook fires on every generation step with the
    full sequence. This ensures the patched activation at position
    inject_token_position is visible to all generated tokens via attention.

    Returns:
        (generated_text, list_of_generated_tokens)
    """
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids)

    layers = _get_transformer_layers(model)
    act = activation.to(device)

    def injection_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if inject_token_position < hidden.shape[1]:
            hidden[0, inject_token_position, :] = act
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = layers[injection_layer].register_forward_hook(injection_hook)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=False,  # Disable KV cache so hook fires every step
            )
    finally:
        handle.remove()

    new_token_ids = output_ids[0, input_ids.shape[1] :]
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    generated_tokens = [tokenizer.decode([tid]) for tid in new_token_ids]

    return generated_text, generated_tokens


def run_without_injection(
    model, tokenizer, device, text: str, max_new_tokens: int = 10
) -> str:
    """Baseline: generate from the target prompt without any activation injection."""
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Main ─────────────────────────────────────────────────────────────


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Patchscope sanity check")
    parser.add_argument(
        "--config",
        default="patchscope_test.yaml",
        help="Config file in config/ directory",
    )
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--device", default=None, help="Override device")
    args = parser.parse_args()

    # Load config
    config_path = CONFIG_DIR / args.config
    cfg = load_config(config_path)
    model_name = args.model or cfg["model"]["name"]
    dtype_str = cfg["model"].get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    logger.info(f"Loading model: {model_name} ({dtype_str})")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=cfg["model"].get("device_map", "auto"),
    )
    model.eval()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif hasattr(model, "hf_device_map"):
        device = next(model.parameters()).device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_layers = len(_get_transformer_layers(model))
    logger.info(f"Model loaded: {n_layers} layers, device={device}")

    # ── Build source and target prompts ─────────────────────────────
    source_text = cfg["source"]["text"]
    extract_word = cfg["source"]["extract_word"]
    subtoken_strategy = cfg["source"].get("subtoken_strategy", "last")

    target_style = cfg["target"].get("style", "raw")

    if target_style == "chat":
        # Build chat-templated target prompt
        chat_messages = [
            {"role": "system", "content": cfg["target"]["chat_system"]},
            {"role": "user", "content": cfg["target"]["chat_user"]},
        ]
        target_text = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        inject_token = cfg["target"]["chat_inject_token"]
        target_display = cfg["target"]["chat_user"]
    else:
        target_text = cfg["target"]["raw_text"]
        inject_token = cfg["target"]["raw_inject_token"]
        target_display = target_text

    # Tokenize source (always raw, no chat template)
    source_ids = tokenizer.encode(source_text, add_special_tokens=False)
    source_tokens = [tokenizer.decode([tid]) for tid in source_ids]
    extract_pos = find_token_position(tokenizer, source_text, extract_word, subtoken_strategy)

    # Tokenize target (may be chat-templated)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    target_tokens = [tokenizer.decode([tid]) for tid in target_ids]
    inject_pos = find_token_position(tokenizer, target_text, inject_token, "last")

    logger.info(f"Source tokens: {list(enumerate(source_tokens))}")
    logger.info(f"Extract '{extract_word}' at position {extract_pos} = '{source_tokens[extract_pos]}'")
    logger.info(f"Target style: {target_style}")
    logger.info(f"Target prompt: '{target_display}'")
    logger.info(f"Target tokens (around inject): ...{list(enumerate(target_tokens))[max(0,inject_pos-3):inject_pos+3]}")
    logger.info(f"Inject at '{inject_token}' position {inject_pos} = '{target_tokens[inject_pos]}'")

    # ── Verify positions match actual model input ────────────────────
    # These assertions catch the add_special_tokens mismatch bug
    source_verify = tokenizer.decode([source_ids[extract_pos]])
    target_verify = tokenizer.decode([target_ids[inject_pos]])
    logger.info(f"VERIFY extract token at pos {extract_pos}: '{source_verify}'")
    logger.info(f"VERIFY inject token at pos {inject_pos}: '{target_verify}'")
    assert extract_word.lower() in source_verify.lower().strip() or source_verify.strip() in extract_word, (
        f"Position mismatch! Expected '{extract_word}' at pos {extract_pos}, "
        f"got '{source_verify}'. Token list: {list(enumerate(source_tokens))}"
    )

    # ── Baseline (no injection) ──────────────────────────────────────
    logger.info("=== Baseline (no injection) ===")
    baseline_text = run_without_injection(
        model, tokenizer, device, target_text,
        max_new_tokens=cfg["generation"]["max_new_tokens"],
    )
    logger.info(f"Baseline output: '{baseline_text}'")

    # ── Build layer pairs ───────────────────────────────────────────
    match_layers = cfg["layers"].get("match_source_target", True)
    layer_pairs = []

    # Sweep mode: same layer for source and target
    if match_layers:
        sweep = cfg["layers"]["sweep"]
        if sweep == "all":
            sweep = list(range(n_layers))
        sweep = [l for l in sweep if l < n_layers]
        layer_pairs.extend([(l, l) for l in sweep])

    # Explicit pairs (always appended if present)
    explicit = cfg["layers"].get("layer_pairs", [])
    for pair in explicit:
        src, tgt = pair[0], pair[1]
        if src < n_layers and tgt < n_layers:
            layer_pairs.append((src, tgt))

    # Deduplicate while preserving order
    seen = set()
    unique_pairs = []
    for p in layer_pairs:
        if p not in seen:
            seen.add(p)
            unique_pairs.append(p)
    layer_pairs = unique_pairs

    gen_cfg = cfg["generation"]
    results = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1].lower().replace(".", "-")

    logger.info(f"=== Layer pairs ({len(layer_pairs)}): {layer_pairs} ===")
    for source_layer, target_layer in layer_pairs:

        # Extract
        activation = extract_activation_raw(
            model, tokenizer, device, source_text, source_layer, extract_pos
        )

        # Inject and generate
        generated_text, generated_tokens = inject_and_generate_raw(
            model, tokenizer, device, target_text, activation,
            injection_layer=target_layer,
            inject_token_position=inject_pos,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg.get("temperature", 0.0),
            do_sample=gen_cfg.get("do_sample", False),
        )

        result = {
            "source_text": source_text,
            "target_text": target_text,
            "extract_word": extract_word,
            "extract_position": extract_pos,
            "inject_position": inject_pos,
            "source_layer": source_layer,
            "injection_layer": target_layer,
            "generated_text": generated_text.strip(),
            "generated_tokens": generated_tokens,
            "baseline_text": baseline_text.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result)

        status = "✓" if any(
            kw in generated_text.lower()
            for kw in ["bezos", "jeff", "ceo", "amazon"]
        ) else " "
        logger.info(
            f"  Layer {source_layer:>2} → {target_layer:>2}: "
            f"[{status}] '{generated_text.strip()[:80]}'"
        )

    # ── Extra test cases (run all layer pairs for each) ─────────────
    extra_tests = cfg.get("extra_tests", [])
    for test in extra_tests:
        t_extract_word = test["extract_word"]
        t_source_text = test["source_text"]
        t_expected = test.get("expected_contains", "")

        t_extract_pos = find_token_position(
            tokenizer, t_source_text, t_extract_word, subtoken_strategy
        )

        logger.info(f"\n=== Extra test: '{t_source_text}' (extract '{t_extract_word}' @ pos {t_extract_pos}) ===")

        for src_layer, tgt_layer in layer_pairs:
            act = extract_activation_raw(
                model, tokenizer, device, t_source_text, src_layer, t_extract_pos
            )
            gen_text, gen_tokens = inject_and_generate_raw(
                model, tokenizer, device, target_text, act,
                injection_layer=tgt_layer,
                inject_token_position=inject_pos,
                max_new_tokens=gen_cfg["max_new_tokens"],
            )

            found = t_expected.lower() in gen_text.lower() if t_expected else None
            status = "✓" if found else "✗" if found is False else " "
            logger.info(
                f"  Layer {src_layer:>2} → {tgt_layer:>2}: "
                f"[{status}] '{gen_text.strip()[:80]}'"
            )

            results.append({
                "source_text": t_source_text,
                "target_text": target_text,
                "extract_word": t_extract_word,
                "extract_position": t_extract_pos,
                "inject_position": inject_pos,
                "source_layer": src_layer,
                "injection_layer": tgt_layer,
                "generated_text": gen_text.strip(),
                "generated_tokens": gen_tokens,
                "expected_contains": t_expected,
                "found_expected": found,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    # ── Save results ─────────────────────────────────────────────────
    jsonl_path = RESULTS_DIR / f"patchscope_{model_short}_{timestamp}_test.jsonl"
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Write log file ───────────────────────────────────────────────
    log_path = jsonl_path.with_suffix(".txt")
    with open(log_path, "w") as f:
        f.write(f"Patchscope Test\n{'=' * 60}\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dtype: {dtype_str}\n")
        f.write(f"Layers: {n_layers}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Target style: {target_style}\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write(f"Source prompt (exact):\n  {source_text}\n\n")
        f.write(f"Source tokens:\n  {list(enumerate(source_tokens))}\n\n")
        f.write(f"Extract word: '{extract_word}' at position {extract_pos}\n\n")

        f.write(f"Target prompt (exact):\n  {target_text}\n\n")
        f.write(f"Target tokens:\n  {list(enumerate(target_tokens))}\n\n")
        f.write(f"Inject at: '{inject_token}' position {inject_pos}\n\n")

        f.write(f"Baseline (no injection): '{baseline_text.strip()}'\n\n")

        f.write(f"{'=' * 60}\nLayer Sweep Results\n{'=' * 60}\n\n")
        for r in results:
            if "expected_contains" not in r:
                f.write(
                    f"  Layer {r['source_layer']:>2} → {r['injection_layer']:>2}: "
                    f"'{r['generated_text']}'\n"
                )

        if extra_tests:
            f.write(f"\n{'=' * 60}\nExtra Tests\n{'=' * 60}\n\n")
            current_source = None
            for r in results:
                if "expected_contains" in r:
                    if r["source_text"] != current_source:
                        current_source = r["source_text"]
                        f.write(f"\n  Source: '{current_source}' (extract '{r['extract_word']}')\n")
                    found = "✓" if r.get("found_expected") else "✗"
                    f.write(
                        f"    Layer {r['source_layer']:>2} → {r['injection_layer']:>2}: "
                        f"[{found}] '{r['generated_text']}'\n"
                    )

    logger.info(f"\nResults: {jsonl_path}")
    logger.info(f"Log: {log_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
