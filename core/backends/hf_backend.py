"""Hugging Face local model backend."""

import logging
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

logger = logging.getLogger(__name__)


def _require_gpu(resolved_device: str) -> None:
    """Hard-fail if the backend would run on CPU.

    Experiments in this repo are too slow on CPU to be meaningful (TinyLlama-1.1B
    takes ~5s per call on CPU vs ~50ms on a modern GPU — a smoke run balloons
    from minutes to ~40+ minutes). Refuse to start rather than silently fall back.

    Common cause of accidental CPU runs: the installed torch wheel is built for
    a newer CUDA than the host driver supports, so `torch.cuda.is_available()`
    returns False and the device resolution falls through to "cpu".

    Set ALLOW_CPU=1 in the environment to bypass (e.g. CI or notebooks on a
    machine without a GPU).
    """
    if os.environ.get("ALLOW_CPU") == "1":
        return

    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if resolved_device != "cpu" and (cuda_ok or mps_ok):
        return

    built_cuda = torch.version.cuda or "(none — CPU-only torch wheel)"
    raise RuntimeError(
        "HFBackend refusing to run on CPU — experiments are far too slow to be useful.\n"
        f"  torch={torch.__version__}, built for CUDA {built_cuda}\n"
        f"  torch.cuda.is_available() = {cuda_ok}\n"
        f"  torch.backends.mps.is_available() = {mps_ok}\n"
        f"  resolved device = {resolved_device!r}\n"
        "\n"
        "Most common cause: the torch wheel is built for a newer CUDA than the\n"
        "system NVIDIA driver supports. Check `nvidia-smi` for the driver's\n"
        "CUDA version and reinstall a matching wheel, e.g.:\n"
        "  pip install --index-url https://download.pytorch.org/whl/cu128 torch\n"
        "\n"
        "To intentionally run on CPU (tests, local dev without a GPU), set\n"
        "ALLOW_CPU=1 in the environment."
    )


class HFBackend:
    """Backend for local HuggingFace causal language models."""

    def __init__(self, model_name: str, device: Optional[str] = None, torch_dtype=None,
                 adapter: Optional[str] = None):
        self.model_name = model_name
        self.adapter = adapter
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        _require_gpu(self.device)
        self.torch_dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device == "auto" else None,
            )
        except ValueError as e:
            # Some instruct checkpoints (e.g. Mistral-Small-3.1 and its
            # derivatives like Dolphin-Mistral-24B-Venice-Edition) ship a
            # multimodal `*ForConditionalGeneration` architecture that
            # AutoModelForCausalLM cannot build. They still run text-only, so
            # fall back to the image-text-to-text class, which exposes the same
            # `.generate()` / `.logits` text interface this backend relies on.
            if "AutoModelForCausalLM" not in str(e):
                raise
            from transformers import AutoModelForImageTextToText

            logger.warning(
                "%s is not a plain causal LM (%s); loading it via "
                "AutoModelForImageTextToText and using it text-only.",
                model_name,
                str(e).splitlines()[0],
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device == "auto" else None,
            )
        if self.device != "auto":
            self.model = self.model.to(self.device)

        # Optionally load a LoRA (PEFT) adapter on top of the base model. The
        # adapter is the HF repo id / local path of the trained adapter; its
        # adapter_config.json's base_model must match `model_name`.
        if adapter:
            from peft import PeftModel

            logger.info("Loading LoRA adapter: %s", adapter)
            self.model = PeftModel.from_pretrained(self.model, adapter, torch_dtype=self.torch_dtype)
            if self.device != "auto":
                self.model = self.model.to(self.device)

        self.model.eval()

    @property
    def input_device(self):
        """Device for input tensors. With device_map='auto', detects the actual first device."""
        if self.device == "auto":
            return next(self.model.parameters()).device
        return self.device

    def _last_position_logits(self, input_ids, attention_mask=None, position_ids=None):
        """Next-token logits at the FINAL position only, shape (batch, vocab).

        Requests `logits_to_keep=1` so the lm_head runs on just the last position
        instead of the whole sequence. Materializing full (batch, seq_len, vocab)
        logits is a major memory sink on large-vocab models — Qwen2.5's vocab is
        ~152k, so a batched forward over a long prompt can allocate multiple GiB
        of logits and OOM the GPU — yet every caller here uses only the last
        position. Falls back to a full forward (older transformers name the kwarg
        `num_logits_to_keep`, oldest accept neither) and slices [-1].
        """
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        with torch.no_grad():
            for keep_kw in ("logits_to_keep", "num_logits_to_keep"):
                try:
                    out = self.model(input_ids, **{keep_kw: 1}, **kwargs)
                    return out.logits[:, -1, :]
                except TypeError:
                    continue
            out = self.model(input_ids, **kwargs)
        return out.logits[:, -1, :]

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """Generate text from chat-style messages. Returns the assistant response."""
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.input_device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def get_next_token_logits(self, messages: list[dict[str, str]]) -> torch.Tensor:
        """Return logits for the next token given chat messages."""
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.input_device)

        return self._last_position_logits(input_ids)[0]  # (vocab_size,)

    def get_choice_probs(
        self, messages: list[dict[str, str]], choices: list[str]
    ) -> dict[str, float]:
        """Get probability distribution over constrained choices (e.g. ["A","B","C","D"]).

        Tokenizes each choice label, takes the first token of each, and returns
        softmax probabilities restricted to those tokens.
        """
        logits = self.get_next_token_logits(messages)

        # Map each choice label to its best first-token logit, considering both
        # the bare ("YES") and space-prefixed (" YES") variants. Depending on
        # the tokenizer and chat template, the model may emit either, and they
        # are distinct token ids — taking the max avoids silently scoring the
        # wrong token (which would corrupt the recognition metric). Mirrors
        # get_choice_probs_and_logits_from_text.
        choice_best_logits = {}
        for choice in choices:
            candidate_logit_values = []
            for variant in (choice, " " + choice):
                tokens = self.tokenizer.encode(variant, add_special_tokens=False)
                if tokens:
                    candidate_logit_values.append(logits[tokens[0]].item())
            if candidate_logit_values:
                choice_best_logits[choice] = max(candidate_logit_values)

        choice_logits = torch.tensor(list(choice_best_logits.values()))
        probs = F.softmax(choice_logits, dim=0)

        return {
            choice: probs[i].item()
            for i, choice in enumerate(choice_best_logits.keys())
        }

    def get_choice_probs_and_logprobs(
        self, messages: list[dict[str, str]], choices: list[str]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Constrained choice probs *and* full-vocab logprobs in one forward pass.

        Returns (probs_dict, logprobs_dict) where:
        - probs_dict: softmax over the choices only (sums to 1) — identical to
          get_choice_probs, including the max-over-{bare, space-prefixed} trick.
        - logprobs_dict: full-vocabulary log-softmax value at the *winning*
          first-token variant for each choice. This is the absolute logprob the
          model assigned to that token, not renormalized over the choice set —
          useful for a logprob_yes - logprob_no style margin.

        Mirrors get_choice_probs' token selection so the choice and its logprob
        always refer to the same token id.
        """
        logits = self.get_next_token_logits(messages)
        full_logprobs = F.log_softmax(logits, dim=0)

        choice_best_logit: dict[str, float] = {}
        choice_best_logprob: dict[str, float] = {}
        for choice in choices:
            best_logit = None
            best_logprob = None
            for variant in (choice, " " + choice):
                tokens = self.tokenizer.encode(variant, add_special_tokens=False)
                if tokens:
                    lg = logits[tokens[0]].item()
                    if best_logit is None or lg > best_logit:
                        best_logit = lg
                        best_logprob = full_logprobs[tokens[0]].item()
            if best_logit is not None:
                choice_best_logit[choice] = best_logit
                choice_best_logprob[choice] = best_logprob

        choice_logits = torch.tensor(list(choice_best_logit.values()))
        probs = F.softmax(choice_logits, dim=0)
        probs_dict = {
            choice: probs[i].item()
            for i, choice in enumerate(choice_best_logit.keys())
        }
        return probs_dict, choice_best_logprob

    def get_choice_probs_and_logits(
        self, messages: list[dict[str, str]], choices: list[str],
        save_logprobs: bool = False,
    ) -> tuple[dict[str, float], dict[str, float], ...]:
        """Get both probabilities and raw logits for constrained choices.

        Single forward pass. Returns (probs_dict, logits_dict) where:
        - probs_dict: softmax probabilities over choices only (sum to 1)
        - logits_dict: raw pre-softmax logits (preserve scale information)

        If save_logprobs=True, returns (probs_dict, logits_dict, logprobs_dict, total_choice_prob)
        where logprobs_dict has full-vocab log-softmax values at each choice token.
        """
        logits = self.get_next_token_logits(messages)

        choice_token_ids = {}
        for choice in choices:
            tokens = self.tokenizer.encode(choice, add_special_tokens=False)
            if tokens:
                choice_token_ids[choice] = tokens[0]

        choice_logits = torch.tensor(
            [logits[tid].item() for tid in choice_token_ids.values()]
        )
        probs = F.softmax(choice_logits, dim=0)

        probs_dict = {
            choice: probs[i].item()
            for i, choice in enumerate(choice_token_ids.keys())
        }
        logits_dict = {
            choice: choice_logits[i].item()
            for i, choice in enumerate(choice_token_ids.keys())
        }

        if save_logprobs:
            full_logprobs = F.log_softmax(logits, dim=0)
            full_probs = F.softmax(logits, dim=0)
            logprobs_dict = {
                choice: full_logprobs[tid].item()
                for choice, tid in choice_token_ids.items()
            }
            total_choice_prob = sum(
                full_probs[tid].item() for tid in choice_token_ids.values()
            )
            return probs_dict, logits_dict, logprobs_dict, total_choice_prob

        return probs_dict, logits_dict

    # ── Batched chat methods (multiple message-lists in one forward pass) ──
    #
    # These mirror generate / get_choice_probs / get_choice_probs_and_logprobs
    # but accept a list of message-lists and process them in a single padded
    # batch. They are pure speedups: for batch size 1 they return the same
    # values as the single-item methods. Left padding is used so the final
    # (-1) position is the true last token of every sequence regardless of
    # length, which is what both constrained-choice logit reads and generation
    # require for a decoder-only model.

    def _encode_chat_batch(self, messages_list: list[list[dict[str, str]]]):
        """Apply the chat template to each message-list and left-pad into a batch.

        Returns (input_ids, attention_mask) on the input device.
        """
        texts = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_list
        ]
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            self.tokenizer.padding_side = prev_side
        return enc["input_ids"].to(self.input_device), enc["attention_mask"].to(self.input_device)

    @staticmethod
    def _position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """Left-padding-correct position ids for a plain forward pass.

        With left padding, a bare forward defaults position_ids to arange, which
        would assign the pad columns positions 0..k and shift the real tokens —
        corrupting the logits. generate() derives these from the mask internally,
        but a direct model(...) call does not, so we compute them explicitly:
        cumsum over the mask, zeroed where padded.
        """
        pos = attention_mask.long().cumsum(-1) - 1
        return pos.clamp(min=0).masked_fill(attention_mask == 0, 0)

    def _choice_probs_from_logits(self, logits: torch.Tensor, choices: list[str]) -> dict[str, float]:
        """Softmax over the per-choice best first-token logit. Mirrors get_choice_probs."""
        choice_best_logits = {}
        for choice in choices:
            candidate_logit_values = []
            for variant in (choice, " " + choice):
                tokens = self.tokenizer.encode(variant, add_special_tokens=False)
                if tokens:
                    candidate_logit_values.append(logits[tokens[0]].item())
            if candidate_logit_values:
                choice_best_logits[choice] = max(candidate_logit_values)
        choice_logits = torch.tensor(list(choice_best_logits.values()))
        probs = F.softmax(choice_logits, dim=0)
        return {choice: probs[i].item() for i, choice in enumerate(choice_best_logits.keys())}

    def generate_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> list[str]:
        """Batched generate(). Returns one decoded response per message-list."""
        if not messages_list:
            return []
        input_ids, attention_mask = self._encode_chat_batch(messages_list)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Left padding means every row's prompt occupies the same number of
        # columns, so new tokens start at input_ids.shape[1] for all rows.
        prompt_len = input_ids.shape[1]
        return [
            self.tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True)
            for i in range(len(messages_list))
        ]

    def get_choice_probs_batch(
        self, messages_list: list[list[dict[str, str]]], choices: list[str]
    ) -> list[dict[str, float]]:
        """Batched get_choice_probs(). Returns one probs-dict per message-list."""
        if not messages_list:
            return []
        input_ids, attention_mask = self._encode_chat_batch(messages_list)
        position_ids = self._position_ids_from_mask(attention_mask)
        # (batch, vocab) — left-padded, so the last position is the real last token
        last_logits = self._last_position_logits(input_ids, attention_mask, position_ids)
        return [self._choice_probs_from_logits(last_logits[i], choices) for i in range(len(messages_list))]

    def last_token_logits_for_texts(self, texts: list[str]) -> "torch.Tensor | None":
        """Last-position next-token logits for PRE-RENDERED prompt strings (batched).

        Takes finished prompt strings rather than chat messages — the caller owns
        the full construction (chat template + any assistant-turn priming). No
        template is applied here and add_special_tokens=False, since a rendered
        chat string already carries its own BOS/special tokens. Unlike
        get_choice_probs_batch it applies no choice/token-selection policy: it
        returns raw logits and the caller picks token ids (useful when the answer
        token is not the first sub-token of a choice label — e.g. SentencePiece
        encodes "1" as [▁, 1], so the digit is the *last* sub-token). Left-padding
        + mask-derived position ids make column -1 the true final token of every
        row. Returns (batch, vocab), or None for empty input.
        """
        if not texts:
            return None
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            self.tokenizer.padding_side = prev_side
        input_ids = enc["input_ids"].to(self.input_device)
        attention_mask = enc["attention_mask"].to(self.input_device)
        position_ids = self._position_ids_from_mask(attention_mask)
        return self._last_position_logits(input_ids, attention_mask, position_ids)

    def get_choice_probs_and_logprobs_batch(
        self, messages_list: list[list[dict[str, str]]], choices: list[str]
    ) -> list[tuple[dict[str, float], dict[str, float]]]:
        """Batched get_choice_probs_and_logprobs(). Returns (probs, logprobs) per row."""
        if not messages_list:
            return []
        input_ids, attention_mask = self._encode_chat_batch(messages_list)
        position_ids = self._position_ids_from_mask(attention_mask)
        last_logits = self._last_position_logits(input_ids, attention_mask, position_ids)
        results = []
        for i in range(len(messages_list)):
            logits = last_logits[i]
            full_logprobs = F.log_softmax(logits, dim=0)
            choice_best_logit: dict[str, float] = {}
            choice_best_logprob: dict[str, float] = {}
            for choice in choices:
                best_logit = None
                best_logprob = None
                for variant in (choice, " " + choice):
                    tokens = self.tokenizer.encode(variant, add_special_tokens=False)
                    if tokens:
                        lg = logits[tokens[0]].item()
                        if best_logit is None or lg > best_logit:
                            best_logit = lg
                            best_logprob = full_logprobs[tokens[0]].item()
                if best_logit is not None:
                    choice_best_logit[choice] = best_logit
                    choice_best_logprob[choice] = best_logprob
            choice_logits = torch.tensor(list(choice_best_logit.values()))
            probs = F.softmax(choice_logits, dim=0)
            probs_dict = {c: probs[j].item() for j, c in enumerate(choice_best_logit.keys())}
            results.append((probs_dict, choice_best_logprob))
        return results

    # ── Raw text methods (for base / non-instruct models) ────────────────

    def generate_from_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """Generate text from a raw prompt string (no chat template).

        Use this for base (non-instruct) models where the prompt is a
        plain text completion prefix (e.g. few-shot examples).
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.input_device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def get_next_token_logits_from_text(self, prompt: str) -> torch.Tensor:
        """Return logits for the next token given a raw text prompt.

        For N input tokens, outputs.logits has shape (1, N, vocab_size).
        outputs.logits[0, -1, :] is the distribution over token_{N+1} —
        i.e., exactly what the model predicts comes after the full prompt.
        This is the correct position for extracting answer letter logprobs.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.input_device)

        return self._last_position_logits(input_ids)[0]  # (vocab_size,) — next-token after full prompt

    def get_choice_probs_and_logits_from_text(
        self, prompt: str, choices: list[str]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Get both probabilities and raw logits for constrained choices from raw text.

        Same as get_choice_probs_and_logits but takes a raw string prompt
        instead of chat messages. Use for base (non-instruct) models.

        For each choice (e.g. "B"), looks up logits for both the bare token
        ("B") and the space-prefixed token (" B"), taking the max. This is
        necessary because in Llama's tokenizer these are different token IDs,
        and depending on prompt formatting the model may predict either variant.
        """
        logits = self.get_next_token_logits_from_text(prompt)

        choice_best_logits = {}
        for choice in choices:
            # Get token IDs for both bare and space-prefixed variants
            candidate_logit_values = []
            for variant in [choice, " " + choice]:
                tokens = self.tokenizer.encode(variant, add_special_tokens=False)
                if tokens:
                    candidate_logit_values.append(logits[tokens[0]].item())

            if candidate_logit_values:
                # Take the max logit across variants — the model "meant"
                # this choice regardless of whether it predicted a space first
                choice_best_logits[choice] = max(candidate_logit_values)

        choice_logits = torch.tensor(list(choice_best_logits.values()))
        probs = F.softmax(choice_logits, dim=0)

        probs_dict = {
            choice: probs[i].item()
            for i, choice in enumerate(choice_best_logits.keys())
        }
        logits_dict = {
            choice: choice_best_logits[choice]
            for choice in choice_best_logits.keys()
        }
        return probs_dict, logits_dict

    # ── Shared-prefix KV cache reuse ──────────────────────────────────────

    def encode_prefix(self, messages: list[dict[str, str]]) -> tuple[torch.Tensor, tuple]:
        """Run a prefix through the model, return (input_ids, past_key_values).

        Use this to build the shared prefix for the Source × Reporter matrix.
        The returned past_key_values can be reused with continue_from_cache().
        """
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.input_device)

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)

        return input_ids, outputs.past_key_values

    def continue_from_cache(
        self,
        prefix_ids: torch.Tensor,
        past_key_values: tuple,
        suffix_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> str:
        """Continue generation from saved past_key_values with a new suffix.

        This reuses the KV cache from a prior prefix run and appends suffix_text
        before generating. This is NOT activation patching — we are simply reusing
        past_key_values and continuing decoding with a different suffix.
        """
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False, return_tensors="pt").to(self.input_device)

        # Run the suffix through the model using the cached prefix KV
        with torch.no_grad():
            suffix_out = self.model(suffix_ids, past_key_values=past_key_values, use_cache=True)

        # Now generate from the combined state
        combined_len = prefix_ids.shape[1] + suffix_ids.shape[1]
        combined_past = suffix_out.past_key_values

        # Seed generation with the last suffix token's logits
        next_token_logits = suffix_out.logits[0, -1, :]
        generated_ids = []

        for _ in range(max_new_tokens):
            if do_sample and temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = next_token_logits.argmax(dim=-1, keepdim=True)

            token_id = next_id.item()
            if token_id == self.tokenizer.eos_token_id:
                break
            generated_ids.append(token_id)

            with torch.no_grad():
                out = self.model(next_id.unsqueeze(0), past_key_values=combined_past, use_cache=True)
            combined_past = out.past_key_values
            next_token_logits = out.logits[0, -1, :]

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def get_next_token_logits_from_cache(
        self, past_key_values: tuple, suffix_text: str
    ) -> torch.Tensor:
        """Get next-token logits after appending suffix to a cached prefix.

        Useful for measuring source-state metrics at the pause point when
        the suffix is just the choice prompt.
        """
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False, return_tensors="pt").to(self.input_device)

        with torch.no_grad():
            outputs = self.model(suffix_ids, past_key_values=past_key_values)

        return outputs.logits[0, -1, :]
