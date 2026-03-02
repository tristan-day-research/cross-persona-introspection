"""Hugging Face local model backend."""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class HFBackend:
    """Backend for local HuggingFace causal language models."""

    def __init__(self, model_name: str, device: Optional[str] = None, torch_dtype=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device == "auto" else None,
        )
        if self.device != "auto":
            self.model = self.model.to(self.device)
        self.model.eval()

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
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
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
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
        return outputs.logits[0, -1, :]  # (vocab_size,)

    def get_choice_probs(
        self, messages: list[dict[str, str]], choices: list[str]
    ) -> dict[str, float]:
        """Get probability distribution over constrained choices (e.g. ["A","B","C","D"]).

        Tokenizes each choice label, takes the first token of each, and returns
        softmax probabilities restricted to those tokens.
        """
        logits = self.get_next_token_logits(messages)

        # Map choice labels to their first token id
        choice_token_ids = {}
        for choice in choices:
            tokens = self.tokenizer.encode(choice, add_special_tokens=False)
            if tokens:
                choice_token_ids[choice] = tokens[0]

        # Extract logits for choice tokens and softmax
        choice_logits = torch.tensor(
            [logits[tid].item() for tid in choice_token_ids.values()]
        )
        probs = F.softmax(choice_logits, dim=0)

        return {
            choice: probs[i].item()
            for i, choice in enumerate(choice_token_ids.keys())
        }

    def compute_confidence_metrics(
        self, messages: list[dict[str, str]], choices: Optional[list[str]] = None
    ) -> dict:
        """Compute confidence proxies: entropy, top-1 prob, logit gap.

        If choices are provided, computes over constrained choice tokens.
        Otherwise computes over the full vocabulary.
        """
        logits = self.get_next_token_logits(messages)

        if choices:
            choice_token_ids = []
            for choice in choices:
                tokens = self.tokenizer.encode(choice, add_special_tokens=False)
                if tokens:
                    choice_token_ids.append(tokens[0])
            logits = logits[choice_token_ids]

        probs = F.softmax(logits, dim=0)
        log_probs = F.log_softmax(logits, dim=0)

        entropy = -(probs * log_probs).sum().item()
        top_probs, top_indices = probs.topk(min(2, len(probs)))

        top1_prob = top_probs[0].item()
        logit_gap = (logits[top_indices[0]] - logits[top_indices[1]]).item() if len(top_probs) > 1 else float("inf")

        from cross_persona_introspection.core.config_loader import prob_to_confidence_bin
        confidence_bin = prob_to_confidence_bin(top1_prob)

        return {
            "entropy": entropy,
            "top1_prob": top1_prob,
            "logit_gap": logit_gap,
            "confidence_proxy_bin": confidence_bin,
        }

    # ── Shared-prefix KV cache reuse ──────────────────────────────────────

    def encode_prefix(self, messages: list[dict[str, str]]) -> tuple[torch.Tensor, tuple]:
        """Run a prefix through the model, return (input_ids, past_key_values).

        Use this to build the shared prefix for the Source × Reporter matrix.
        The returned past_key_values can be reused with continue_from_cache().
        """
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

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
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False, return_tensors="pt").to(self.device)

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
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(suffix_ids, past_key_values=past_key_values)

        return outputs.logits[0, -1, :]
