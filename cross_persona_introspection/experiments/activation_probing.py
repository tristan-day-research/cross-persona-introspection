"""Activation probing experiment: collect hidden states, logit lens projections,
and answer logits for two opposing personas across a shared question set.

Outputs (all saved to a timestamped subdirectory of results/raw/):
  - activations_{persona}.pt    — [n_questions, n_layers, d_model] float16
  - logit_lens_{persona}.pt     — [n_questions, n_layers, 4] float16
                                   raw logits for [A,B,C,D] at every layer
                                   via final_norm(h_l) → lm_head → ABCD slice
  - answer_logits.csv           — per-question per-persona final-layer logits & probs
  - paired_answers.csv          — agreement/disagreement table
  - metadata.json               — shapes, token IDs, model info, choice order
"""

import json
import logging
import random
import time
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
        import torch
        from cross_persona_introspection.backends.hf_backend import HFBackend

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.model_dtype) if self.config.model_dtype else None

        logger.info(f"Loading model: {self.config.model_name}  dtype={self.config.model_dtype or 'auto'}")
        self.backend = HFBackend(self.config.model_name, device="auto", torch_dtype=torch_dtype)

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

        run_errors: list[str] = []
        run_start = time.monotonic()

        # Resolve answer token IDs — validate that each maps to a single token
        answer_token_ids = self._resolve_answer_tokens(tokenizer, choices)
        logger.info(f"Answer token IDs: {answer_token_ids}")

        n_questions = len(self.questions)
        n_layers = model.config.num_hidden_layers + 1  # embedding + transformer layers
        d_model = model.config.hidden_size
        question_ids = [q["question_id"] for q in self.questions]

        # Get logit lens components (final layer norm + LM head)
        final_norm, lm_head = self._get_logit_lens_components(model)
        # Ordered list of answer token IDs for indexing into vocab logits
        token_id_list = [answer_token_ids[c] for c in choices]  # [A_id, B_id, C_id, D_id]

        persona_names = list(self.personas.keys())
        all_activations: dict[str, torch.Tensor] = {}  # persona -> [n_q, n_layers, d_model]
        all_logit_lens: dict[str, torch.Tensor] = {}   # persona -> [n_q, n_layers, 4]
        logit_rows: list[dict] = []
        sample_prompts: dict[str, dict] = {}  # persona -> {question_id, input_text, chosen, logits, probs}
        questions_ok: dict[str, int] = {}     # persona -> count of successful questions

        for persona_name in persona_names:
            persona = self.personas[persona_name]
            logger.info(f"Running persona: {persona_name}")

            acts = torch.zeros(n_questions, n_layers, d_model, dtype=torch.float16)
            lens = torch.zeros(n_questions, n_layers, 4, dtype=torch.float16)
            n_ok = 0

            for qi, question in enumerate(tqdm(
                self.questions, desc=f"  {persona_name}", leave=True
            )):
                try:
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

                    # Forward pass — capture all hidden states in one pass
                    with torch.no_grad():
                        outputs = model(input_ids, output_hidden_states=True)

                    # Hidden states: tuple of n_layers+1 tensors, each [1, seq_len, d_model]
                    # We take the final token position (immediately before generation)
                    with torch.no_grad():
                        for layer_idx, hs in enumerate(outputs.hidden_states):
                            h = hs[0, -1:, :]  # [1, d_model]

                            # Raw activation (for probe training)
                            acts[qi, layer_idx, :] = h[0].to(torch.float16).cpu()

                            # Logit lens: project through final norm + LM head
                            # This is the standard logit lens (nostalgebraist 2020):
                            # "what would the model predict if it stopped at layer l?"
                            h_normed = final_norm(h.float())                 # [1, d_model] float32 for norm stability
                            vocab_logits = lm_head(h_normed.to(h.dtype))[0] # [vocab_size] cast back to model dtype
                            answer_logits_l = vocab_logits[token_id_list]  # [4]
                            lens[qi, layer_idx, :] = answer_logits_l.to(torch.float16).cpu()

                    # Final answer logits — use model output directly (most numerically stable)
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

                    # Capture sample prompt for the first question only
                    if qi == 0:
                        sample_prompts[persona_name] = {
                            "question_id": question["question_id"],
                            "input_text": input_text,
                            "chosen_answer": chosen,
                            "answer_logits": {c: round(v, 4) for c, v in answer_logits_raw.items()},
                            "answer_probs": {c: round(v, 4) for c, v in answer_probs_dict.items()},
                            "n_input_tokens": int(input_ids.shape[1]),
                        }

                    n_ok += 1
                    del outputs  # free GPU memory before next question

                except Exception as e:
                    msg = (
                        f"ERROR — persona={persona_name} qi={qi} "
                        f"question_id={question.get('question_id', '?')}: {type(e).__name__}: {e}"
                    )
                    logger.error(msg)
                    run_errors.append(msg)

            all_activations[persona_name] = acts
            all_logit_lens[persona_name] = lens
            questions_ok[persona_name] = n_ok

        run_elapsed = time.monotonic() - run_start

        # Save everything
        self._save_outputs(
            all_activations, all_logit_lens, logit_rows, question_ids, persona_names,
            answer_token_ids, choices, n_questions, n_layers, d_model,
            sample_prompts=sample_prompts, run_errors=run_errors,
            questions_ok=questions_ok, run_elapsed=run_elapsed,
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

    @staticmethod
    def _get_logit_lens_components(model):
        """Return (final_norm, lm_head) for logit lens computation.

        Supports Llama/Mistral (model.model.norm), GPT-2 (model.transformer.ln_f),
        and models with model.model.final_layer_norm.
        Raises clearly if the architecture is unsupported.
        """
        lm_head = model.lm_head

        if hasattr(model, "model") and hasattr(model.model, "norm"):
            final_norm = model.model.norm                    # Llama, Mistral, Gemma
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            final_norm = model.transformer.ln_f              # GPT-2
        elif hasattr(model, "model") and hasattr(model.model, "final_layer_norm"):
            final_norm = model.model.final_layer_norm        # OPT, Bloom
        else:
            raise ValueError(
                "Cannot find final layer norm for logit lens. "
                "Checked: model.model.norm, model.transformer.ln_f, "
                "model.model.final_layer_norm. "
                "Add support for this architecture in _get_logit_lens_components()."
            )
        return final_norm, lm_head

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
        all_logit_lens: dict[str, torch.Tensor],
        logit_rows: list[dict],
        question_ids: list[str],
        persona_names: list[str],
        answer_token_ids: dict[str, int],
        choices: list[str],
        n_questions: int,
        n_layers: int,
        d_model: int,
        sample_prompts: dict[str, dict] | None = None,
        run_errors: list[str] | None = None,
        questions_ok: dict[str, int] | None = None,
        run_elapsed: float = 0.0,
    ) -> None:
        """Save all collection outputs to self.out_dir."""
        # 1. Activation tensors
        for persona_name, acts in all_activations.items():
            safe_name = persona_name.replace(" ", "_")
            torch.save(acts, self.out_dir / f"activations_{safe_name}.pt")
            logger.info(f"Saved activations_{safe_name}.pt  shape={list(acts.shape)}")

        # 2. Logit lens tensors
        #    Shape: [n_questions, n_layers, 4] float16
        #    Dim 2 order: same as `choices` list, recorded in metadata
        for persona_name, lens in all_logit_lens.items():
            safe_name = persona_name.replace(" ", "_")
            torch.save(lens, self.out_dir / f"logit_lens_{safe_name}.pt")
            logger.info(f"Saved logit_lens_{safe_name}.pt  shape={list(lens.shape)}")

        # 3. Answer logits table (final layer only — full-layer view is in logit_lens)
        logits_df = pd.DataFrame(logit_rows)
        logits_df.to_csv(self.out_dir / "answer_logits.csv", index=False)
        logger.info(f"Saved answer_logits.csv  ({len(logits_df)} rows)")

        # 4. Paired answers table
        paired = self._build_paired_answers(logits_df, persona_names)
        paired.to_csv(self.out_dir / "paired_answers.csv", index=False)
        logger.info(f"Saved paired_answers.csv  ({len(paired)} rows)")

        # 5. Metadata
        bytes_per_act = n_questions * n_layers * d_model * 2  # float16
        bytes_per_lens = n_questions * n_layers * 4 * 2       # float16, 4 choices
        metadata = {
            "model_name": self.config.model_name,
            "task_file": self.config.task_file,
            "n_questions": n_questions,
            "n_layers": n_layers,
            "d_model": d_model,
            "question_ids": question_ids,
            "persona_names": persona_names,
            "answer_token_ids": {k: int(v) for k, v in answer_token_ids.items()},
            "logit_lens_choice_order": choices,  # dim-2 order for logit_lens_*.pt
            "answer_instruction": ANSWER_INSTRUCTION,
            "seed": self.config.seed,
            "sample_size": self.config.sample_size,
            "approx_size_mb": {
                "activations_per_persona": round(bytes_per_act / 1e6, 1),
                "logit_lens_per_persona": round(bytes_per_lens / 1e6, 2),
                "total_all_personas": round(
                    (bytes_per_act + bytes_per_lens) * len(persona_names) / 1e6, 1
                ),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata.json")

        # 6. Human-readable run report
        self._write_run_report(
            metadata, sample_prompts or {}, run_errors or [],
            persona_names, n_questions, n_layers, d_model,
            questions_ok=questions_ok or {}, run_elapsed=run_elapsed,
        )
        logger.info("Saved run_report.txt")

    def _write_run_report(
        self,
        metadata: dict,
        sample_prompts: dict[str, dict],
        run_errors: list[str],
        persona_names: list[str],
        n_questions: int,
        n_layers: int,
        d_model: int,
        questions_ok: dict[str, int] | None = None,
        run_elapsed: float = 0.0,
    ) -> None:
        """Write a human-readable run_report.txt summarising the full run."""
        sep = "=" * 70
        thin = "-" * 70
        lines = []

        h, rem = divmod(int(run_elapsed), 3600)
        m, s = divmod(rem, 60)
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        lines += [
            sep,
            "ACTIVATION PROBING RUN REPORT",
            sep,
            f"Generated  : {datetime.now(timezone.utc).isoformat()}",
            f"Output dir : {self.out_dir}",
            f"Total time : {elapsed_str}  ({run_elapsed:.1f}s)",
            "",
        ]

        # ── Question counts ───────────────────────────────────────────
        lines += [sep, "QUESTION COUNTS", thin]
        lines.append(f"  Questions loaded  : {n_questions}")
        qok = questions_ok or {}
        for pname in persona_names:
            ok = qok.get(pname, "?")
            failed = (n_questions - ok) if isinstance(ok, int) else "?"
            lines.append(f"  {pname:<35s}  {ok}/{n_questions} ok  ({failed} failed)")
        lines.append("")

        # ── Run parameters ────────────────────────────────────────────
        lines += [sep, "RUN PARAMETERS", thin]
        cfg = self.config
        lines += [
            f"experiment_name : {cfg.experiment_name}",
            f"model_name      : {cfg.model_name}",
            f"model_dtype     : {cfg.model_dtype or 'auto (float16 on CUDA)'}",
            f"batch_size      : {cfg.batch_size}",
            f"task_file       : {cfg.task_file}",
            f"sample_size     : {cfg.sample_size}  (None = all questions)",
            f"seed            : {cfg.seed}",
            f"temperature     : {cfg.temperature}",
            f"output_dir      : {cfg.output_dir}",
            "",
        ]

        # ── Dataset / model info ──────────────────────────────────────
        lines += [sep, "DATASET / MODEL INFO", thin]
        lines += [
            f"n_questions     : {n_questions}",
            f"n_layers        : {n_layers}  (embedding layer 0 + {n_layers-1} transformer layers)",
            f"d_model         : {d_model}",
            f"answer_token_ids: {metadata.get('answer_token_ids')}",
            f"logit_lens_order: {metadata.get('logit_lens_choice_order')}",
        ]
        sz = metadata.get("approx_size_mb", {})
        lines += [
            f"approx size/persona (activations): {sz.get('activations_per_persona')} MB",
            f"approx size/persona (logit lens) : {sz.get('logit_lens_per_persona')} MB",
            f"approx total all personas        : {sz.get('total_all_personas')} MB",
            "",
        ]

        # ── Answer instruction ────────────────────────────────────────
        lines += [sep, "ANSWER INSTRUCTION (appended to every user message)", thin]
        lines += [ANSWER_INSTRUCTION, ""]

        # ── Personas ─────────────────────────────────────────────────
        lines += [sep, "PERSONAS", thin]
        for pname in persona_names:
            persona = self.personas[pname]
            lines += [
                f"[{pname}]",
                f"  description  : {persona.description}",
                f"  system_prompt:",
            ]
            if persona.system_prompt:
                for sline in persona.system_prompt.strip().splitlines():
                    lines.append(f"    {sline}")
            else:
                lines.append("    (none — no system prompt)")
            lines.append("")

        # ── Sample prompts ────────────────────────────────────────────
        lines += [sep, "SAMPLE PROMPTS (first question per persona)", thin]
        for pname in persona_names:
            sp = sample_prompts.get(pname)
            lines.append(f"[{pname}]")
            if sp is None:
                lines += ["  (no sample captured — persona may have errored)", ""]
                continue
            lines += [
                f"  question_id    : {sp['question_id']}",
                f"  n_input_tokens : {sp['n_input_tokens']}",
                f"  chosen_answer  : {sp['chosen_answer']}",
                f"  answer_logits  : {sp['answer_logits']}",
                f"  answer_probs   : {sp['answer_probs']}",
                "",
                "  ── full prompt sent to model ──",
            ]
            for pline in sp["input_text"].splitlines():
                lines.append(f"  {pline}")
            lines.append("")

        # ── Errors ────────────────────────────────────────────────────
        lines += [sep, f"ERRORS ({len(run_errors)} total)", thin]
        if run_errors:
            lines += run_errors
        else:
            lines.append("None")
        lines.append("")

        # ── Output files ──────────────────────────────────────────────
        lines += [sep, "OUTPUT FILES", thin]
        for f in sorted(self.out_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            lines.append(f"  {f.name:<40s}  {size_kb:>10.1f} KB")
        lines.append("")
        lines.append(sep)

        report_path = self.out_dir / "run_report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")

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
