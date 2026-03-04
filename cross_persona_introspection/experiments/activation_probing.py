"""Activation probing experiment: collect hidden states and answer logits
for two opposing personas across a shared question set.

Outputs (all saved to a timestamped subdirectory of results/raw/):
  - activations_{persona}.pt   — [n_questions, n_layers, d_model] float16
  - answer_logits.csv           — per-question per-persona logits & probs
  - paired_answers.csv          — agreement/disagreement table
  - metadata.json               — shapes, token IDs, model info
"""

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cross_persona_introspection.experiments.base import BaseExperiment
from cross_persona_introspection.schemas import PersonaConfig, RunConfig

logger = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).parent.parent.parent / "tasks"

# Shared output-format instruction — NOT embedded in persona system prompts
ANSWER_INSTRUCTION = (
    "Choose the single best answer from this persona's perspective. "
    "Answer with exactly one letter: A, B, C, or D. Do not provide any explanation."
)


def _load_questions(task_file: str, sample_size: Optional[int], seed: int) -> list[dict]:
    """Load opposing-personas questions from JSON, optionally sample."""
    path = TASKS_DIR / task_file
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    with open(path) as f:
        questions = json.load(f)
    if sample_size is not None and sample_size < len(questions):
        rng = random.Random(seed)
        questions = rng.sample(questions, sample_size)
    return questions


def _format_user_message(question: dict) -> str:
    """Format a question dict into a user-facing MCQ string."""
    lines = [question["question_text"], ""]
    for letter in ["A", "B", "C", "D"]:
        lines.append(f"{letter}) {question['options'][letter]}")
    lines.append("")
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines)


class ActivationProbing(BaseExperiment):
    """Collect hidden-state activations and answer logits for opposing personas."""

    def __init__(self, config: RunConfig, personas: dict[str, PersonaConfig]):
        super().__init__(config)
        self.personas = personas
        self.backend = None
        self.questions: list[dict] = []

        # Output directory: one folder per run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = Path(config.output_dir) / f"activation_probing_{timestamp}"

    # ── BaseExperiment interface ──────────────────────────────────────

    def setup(self) -> None:
        from cross_persona_introspection.backends.hf_backend import HFBackend

        logger.info(f"Loading model: {self.config.model_name}")
        self.backend = HFBackend(self.config.model_name)

        task_file = self.config.task_file
        if not task_file:
            raise ValueError("activation_probing requires 'task_file' in config")
        self.questions = _load_questions(task_file, self.config.sample_size, self.config.seed)
        logger.info(f"Loaded {len(self.questions)} questions from {task_file}")

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        assert self.backend is not None
        model = self.backend.model
        tokenizer = self.backend.tokenizer
        device = self.backend.device
        choices = ["A", "B", "C", "D"]

        # Resolve answer token IDs — validate that each maps to a single token
        answer_token_ids = self._resolve_answer_tokens(tokenizer, choices)
        logger.info(f"Answer token IDs: {answer_token_ids}")

        n_questions = len(self.questions)
        n_layers = model.config.num_hidden_layers + 1  # embedding + transformer layers
        d_model = model.config.hidden_size
        question_ids = [q["question_id"] for q in self.questions]

        persona_names = list(self.personas.keys())
        all_activations: dict[str, torch.Tensor] = {}  # persona -> [n_q, n_layers, d_model]
        logit_rows: list[dict] = []

        for persona_name in persona_names:
            persona = self.personas[persona_name]
            logger.info(f"Running persona: {persona_name}")

            acts = torch.zeros(n_questions, n_layers, d_model, dtype=torch.float16)

            for qi, question in enumerate(tqdm(
                self.questions, desc=f"  {persona_name}", leave=True
            )):
                user_msg = _format_user_message(question)
                messages = []
                if persona.system_prompt:
                    messages.append({"role": "system", "content": persona.system_prompt})
                messages.append({"role": "user", "content": user_msg})

                # Apply chat template
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

                # Forward pass with hidden states
                with torch.no_grad():
                    outputs = model(
                        input_ids,
                        output_hidden_states=True,
                    )

                # Hidden states: tuple of (n_layers+1) tensors, each [1, seq_len, d_model]
                # Take the final token position (immediately before generation)
                for layer_idx, hs in enumerate(outputs.hidden_states):
                    acts[qi, layer_idx, :] = hs[0, -1, :].to(torch.float16).cpu()

                # Extract answer logits from final position
                logits = outputs.logits[0, -1, :]  # (vocab_size,)
                answer_logits_raw = {c: logits[tid].item() for c, tid in answer_token_ids.items()}
                answer_logit_tensor = torch.tensor([answer_logits_raw[c] for c in choices])
                answer_probs = F.softmax(answer_logit_tensor, dim=0)
                answer_probs_dict = {c: answer_probs[i].item() for i, c in enumerate(choices)}
                chosen = choices[answer_logit_tensor.argmax().item()]

                logit_rows.append({
                    "question_id": question["question_id"],
                    "persona": persona_name,
                    "logit_A": answer_logits_raw["A"],
                    "logit_B": answer_logits_raw["B"],
                    "logit_C": answer_logits_raw["C"],
                    "logit_D": answer_logits_raw["D"],
                    "prob_A": answer_probs_dict["A"],
                    "prob_B": answer_probs_dict["B"],
                    "prob_C": answer_probs_dict["C"],
                    "prob_D": answer_probs_dict["D"],
                    "chosen_answer": chosen,
                })

            all_activations[persona_name] = acts

        # Save everything
        self._save_outputs(
            all_activations, logit_rows, question_ids, persona_names,
            answer_token_ids, n_questions, n_layers, d_model,
        )

    def evaluate(self) -> dict:
        """Print agreement/disagreement summary. No scoring to do."""
        paired_path = self.out_dir / "paired_answers.csv"
        if not paired_path.exists():
            return {"status": "no paired_answers.csv found"}
        df = pd.read_csv(paired_path)
        n_agree = int(df["is_agreement"].sum())
        n_disagree = int(df["is_disagreement"].sum())
        total = len(df)
        logger.info(f"Agreement: {n_agree}/{total} ({100*n_agree/total:.1f}%)")
        logger.info(f"Disagreement: {n_disagree}/{total} ({100*n_disagree/total:.1f}%)")
        return {
            "n_questions": total,
            "n_agreement": n_agree,
            "n_disagreement": n_disagree,
            "agreement_rate": round(n_agree / total, 4) if total else 0,
        }

    def save_results(self) -> str:
        """Results already saved in run(). Return output dir."""
        return str(self.out_dir)

    # ── Internal helpers ──────────────────────────────────────────────

    def _resolve_answer_tokens(
        self, tokenizer, choices: list[str]
    ) -> dict[str, int]:
        """Map each choice letter to a single token ID.

        Tries space-prefixed variants first (e.g. " A") since chat templates
        often produce contexts where the answer token has a leading space.
        Falls back to bare letter. Fails loudly if neither is a single token.
        """
        token_ids = {}
        for c in choices:
            # Try space-prefixed first (common after generation prompt)
            for variant in [f" {c}", c]:
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(ids) == 1:
                    token_ids[c] = ids[0]
                    break
            if c not in token_ids:
                raise ValueError(
                    f"Cannot find a clean single-token mapping for answer '{c}'. "
                    f"Tried '{c}' -> {tokenizer.encode(c, add_special_tokens=False)} "
                    f"and ' {c}' -> {tokenizer.encode(' ' + c, add_special_tokens=False)}. "
                    f"Check tokenizer compatibility."
                )
        return token_ids

    def _save_outputs(
        self,
        all_activations: dict[str, torch.Tensor],
        logit_rows: list[dict],
        question_ids: list[str],
        persona_names: list[str],
        answer_token_ids: dict[str, int],
        n_questions: int,
        n_layers: int,
        d_model: int,
    ) -> None:
        """Save all collection outputs to self.out_dir."""
        # 1. Activation tensors
        for persona_name, acts in all_activations.items():
            safe_name = persona_name.replace(" ", "_")
            torch.save(acts, self.out_dir / f"activations_{safe_name}.pt")
            logger.info(f"Saved activations_{safe_name}.pt  shape={list(acts.shape)}")

        # 2. Answer logits table
        logits_df = pd.DataFrame(logit_rows)
        logits_df.to_csv(self.out_dir / "answer_logits.csv", index=False)
        logger.info(f"Saved answer_logits.csv  ({len(logits_df)} rows)")

        # 3. Paired answers table
        paired = self._build_paired_answers(logits_df, persona_names)
        paired.to_csv(self.out_dir / "paired_answers.csv", index=False)
        logger.info(f"Saved paired_answers.csv  ({len(paired)} rows)")

        # 4. Metadata
        metadata = {
            "model_name": self.config.model_name,
            "task_file": self.config.task_file,
            "n_questions": n_questions,
            "n_layers": n_layers,
            "d_model": d_model,
            "question_ids": question_ids,
            "persona_names": persona_names,
            "answer_token_ids": {k: int(v) for k, v in answer_token_ids.items()},
            "answer_instruction": ANSWER_INSTRUCTION,
            "seed": self.config.seed,
            "sample_size": self.config.sample_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata.json")

    def _build_paired_answers(
        self, logits_df: pd.DataFrame, persona_names: list[str]
    ) -> pd.DataFrame:
        """Pivot logit_rows into per-question agreement/disagreement table."""
        if len(persona_names) != 2:
            raise ValueError("Paired answers requires exactly 2 personas")
        p1, p2 = persona_names
        df1 = logits_df[logits_df["persona"] == p1][["question_id", "chosen_answer"]].rename(
            columns={"chosen_answer": f"answer_{p1}"}
        )
        df2 = logits_df[logits_df["persona"] == p2][["question_id", "chosen_answer"]].rename(
            columns={"chosen_answer": f"answer_{p2}"}
        )
        paired = df1.merge(df2, on="question_id", how="inner")
        paired["is_agreement"] = paired[f"answer_{p1}"] == paired[f"answer_{p2}"]
        paired["is_disagreement"] = ~paired["is_agreement"]
        return paired
