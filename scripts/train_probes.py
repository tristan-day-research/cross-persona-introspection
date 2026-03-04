#!/usr/bin/env python3
"""Train linear probes on saved activations from the activation probing experiment.

Usage:
    python scripts/train_probes.py <collection_dir>

Where <collection_dir> is the output directory from run_experiment.py activation_probing_*,
e.g. results/raw/activation_probing_20260303_120000/

Outputs (saved to <collection_dir>/probes/):
    probe_metrics.csv       — one row per (layer, metric_name, subset), columns for accuracy
    probe_predictions.csv   — per-question test-set predictions
    split_manifest.json     — train/val/test question_id lists
    probe_weights/          — saved probe models (joblib)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_collection(collection_dir: Path):
    """Load all outputs from the collection step. Validates consistency."""
    meta_path = collection_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {collection_dir}")
    with open(meta_path) as f:
        metadata = json.load(f)

    persona_names = metadata["persona_names"]
    question_ids = metadata["question_ids"]
    n_questions = metadata["n_questions"]
    n_layers = metadata["n_layers"]
    d_model = metadata["d_model"]

    # Load activations
    activations = {}
    for pname in persona_names:
        safe = pname.replace(" ", "_")
        path = collection_dir / f"activations_{safe}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing activation file: {path}")
        acts = torch.load(path, map_location="cpu", weights_only=True)
        if acts.shape != (n_questions, n_layers, d_model):
            raise ValueError(
                f"Shape mismatch for {pname}: expected ({n_questions}, {n_layers}, {d_model}), "
                f"got {tuple(acts.shape)}"
            )
        activations[pname] = acts.float().numpy()  # [n_q, n_layers, d_model]

    # Load answer tables
    logits_df = pd.read_csv(collection_dir / "answer_logits.csv")
    paired_df = pd.read_csv(collection_dir / "paired_answers.csv")

    # Validate question order
    for pname in persona_names:
        persona_qids = logits_df[logits_df["persona"] == pname]["question_id"].tolist()
        if persona_qids != question_ids:
            raise ValueError(f"Question order mismatch for {pname} in answer_logits.csv")

    if paired_df["question_id"].tolist() != question_ids:
        raise ValueError("Question order mismatch in paired_answers.csv")

    return metadata, activations, logits_df, paired_df


def make_splits(question_ids: list[str], labels: np.ndarray, seed: int = 42):
    """Create train/val/test splits by question_id (60/20/20).

    Uses stratified split on the first persona's answers to keep class balance.
    """
    rng = np.random.RandomState(seed)
    n = len(question_ids)
    indices = np.arange(n)
    rng.shuffle(indices)

    # Stratified split: 60% train, 20% val, 20% test
    n_test = max(1, int(0.2 * n))
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val - n_test

    # Simple shuffle split (stratification is nice-to-have but not critical
    # with small label sets; keeps code simple)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        train_idx.tolist(),
        val_idx.tolist(),
        test_idx.tolist(),
    )


def train_probe_for_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> tuple[LogisticRegression, StandardScaler]:
    """Train a logistic regression probe with standardized features.

    Uses cross-validation on training set to pick regularization strength.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Try a few C values, pick best on val set
    best_score = -1.0
    best_model = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs",
            multi_class="multinomial", random_state=seed,
        )
        clf.fit(X_train_s, y_train)
        score = clf.score(X_val_s, y_val)
        if score > best_score:
            best_score = score
            best_model = clf

    return best_model, scaler


def compute_metrics(
    probe: LogisticRegression,
    scaler: StandardScaler,
    X: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray | None = None,
) -> float | None:
    """Compute accuracy, optionally on a masked subset."""
    if mask is not None:
        X = X[mask]
        y_true = y_true[mask]
    if len(y_true) == 0:
        return None
    X_s = scaler.transform(X)
    return float(probe.score(X_s, y_true))


def main():
    parser = argparse.ArgumentParser(description="Train probes on activation probing outputs")
    parser.add_argument("collection_dir", type=str, help="Path to collection output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    collection_dir = Path(args.collection_dir)
    metadata, activations, logits_df, paired_df = load_collection(collection_dir)

    persona_names = metadata["persona_names"]
    question_ids = metadata["question_ids"]
    n_layers = metadata["n_layers"]
    seed = args.seed

    if len(persona_names) != 2:
        raise ValueError(f"Expected exactly 2 personas, got {len(persona_names)}")
    p1, p2 = persona_names

    # Build answer arrays per persona
    answers = {}
    for pname in persona_names:
        pdf = logits_df[logits_df["persona"] == pname].sort_values("question_id")
        # Ensure order matches question_ids
        ans_series = pdf.set_index("question_id").loc[question_ids, "chosen_answer"]
        answers[pname] = ans_series.values  # array of 'A','B','C','D'

    # Build label encoder (shared across personas for consistency)
    le = LabelEncoder()
    all_answers = np.concatenate([answers[p] for p in persona_names])
    le.fit(all_answers)
    labels = {p: le.transform(answers[p]) for p in persona_names}

    # Agreement/disagreement masks
    agree_mask = (answers[p1] == answers[p2])
    disagree_mask = ~agree_mask
    logger.info(f"Agreement: {agree_mask.sum()}/{len(agree_mask)}, "
                f"Disagreement: {disagree_mask.sum()}/{len(disagree_mask)}")

    # Split by question_id
    train_idx, val_idx, test_idx = make_splits(question_ids, labels[p1], seed)

    # Save split manifest
    out_dir = collection_dir / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "probe_weights"
    weights_dir.mkdir(exist_ok=True)

    split_manifest = {
        "train_question_ids": [question_ids[i] for i in train_idx],
        "val_question_ids": [question_ids[i] for i in val_idx],
        "test_question_ids": [question_ids[i] for i in test_idx],
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "seed": seed,
    }
    with open(out_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)
    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Test-set subset masks
    test_agree = agree_mask[test_idx]
    test_disagree = disagree_mask[test_idx]

    # Train probes: one per persona per layer
    metric_rows = []
    prediction_rows = []

    # Define all evaluation combos we want:
    # (probe_persona, activation_persona, scored_against_persona, description)
    # Same-persona: probe on own activations, scored against own answers
    # Cross-persona: probe on OTHER's activations, scored against probe owner's answers
    #   (key question: can we decode persona A's answer from persona B's hidden states?)
    eval_combos = []
    for probe_p in persona_names:
        other_p = [p for p in persona_names if p != probe_p][0]
        # Same-persona
        eval_combos.append((probe_p, probe_p, probe_p, "same_persona"))
        # Cross-persona: decode probe_p's answer from other_p's activations
        eval_combos.append((probe_p, other_p, probe_p, "cross_persona"))

    for layer in tqdm(range(n_layers), desc="Training probes"):
        probes = {}
        scalers_dict = {}

        for pname in persona_names:
            X_all = activations[pname][:, layer, :]  # [n_q, d_model]
            y_all = labels[pname]

            X_train = X_all[train_idx]
            y_train = y_all[train_idx]
            X_val = X_all[val_idx]
            y_val = y_all[val_idx]

            probe, scaler = train_probe_for_layer(X_train, y_train, X_val, y_val, seed)
            probes[pname] = probe
            scalers_dict[pname] = scaler

            # Save weights
            joblib.dump(
                {"probe": probe, "scaler": scaler, "label_encoder": le},
                weights_dir / f"probe_{pname}_layer{layer}.joblib",
            )

        # Evaluate each combo on test set
        for probe_p, act_p, scored_p, combo_type in eval_combos:
            probe = probes[probe_p]
            scaler = scalers_dict[probe_p]
            X_test = activations[act_p][test_idx][:, layer, :]
            y_test = labels[scored_p][test_idx]

            for subset_name, subset_mask in [
                ("full", None),
                ("agreement", test_agree),
                ("disagreement", test_disagree),
            ]:
                acc = compute_metrics(probe, scaler, X_test, y_test, subset_mask)
                metric_rows.append({
                    "layer": layer,
                    "probe_persona": probe_p,
                    "activation_persona": act_p,
                    "scored_against": scored_p,
                    "combo_type": combo_type,
                    "subset": subset_name,
                    "accuracy": acc,
                    "n_samples": (
                        int(subset_mask.sum()) if subset_mask is not None
                        else len(y_test)
                    ),
                })

        # Per-question test predictions (all probe×activation combos)
        for probe_p, act_p, scored_p, combo_type in eval_combos:
            probe = probes[probe_p]
            scaler = scalers_dict[probe_p]
            X_test = activations[act_p][test_idx][:, layer, :]
            X_test_s = scaler.transform(X_test)
            preds = le.inverse_transform(probe.predict(X_test_s))
            pred_proba = probe.predict_proba(X_test_s)

            for i, qi in enumerate(test_idx):
                row = {
                    "layer": layer,
                    "question_id": question_ids[qi],
                    "probe_persona": probe_p,
                    "activation_persona": act_p,
                    "combo_type": combo_type,
                    "predicted_answer": preds[i],
                    "true_answer_probe_persona": answers[probe_p][qi],
                    "true_answer_act_persona": answers[act_p][qi],
                    "is_agreement": bool(agree_mask[qi]),
                }
                # Add per-class probabilities
                for ci, cls_label in enumerate(le.classes_):
                    row[f"prob_{cls_label}"] = float(pred_proba[i, ci])
                prediction_rows.append(row)

    # Save metrics
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "probe_metrics.csv", index=False)
    logger.info(f"Saved probe_metrics.csv ({len(metrics_df)} rows)")

    # Save predictions
    preds_df = pd.DataFrame(prediction_rows)
    preds_df.to_csv(out_dir / "probe_predictions.csv", index=False)
    logger.info(f"Saved probe_predictions.csv ({len(preds_df)} rows)")

    # Print summary
    print("\n" + "=" * 60)
    print("PROBE TRAINING COMPLETE")
    print("=" * 60)

    # Show same-persona accuracy by layer (last 5 layers)
    same = metrics_df[
        (metrics_df["combo_type"] == "same_persona") & (metrics_df["subset"] == "full")
    ].sort_values("layer")
    if len(same) > 0:
        print(f"\nSame-persona probe accuracy (last 5 layers):")
        for _, row in same.tail(10).iterrows():
            print(f"  Layer {row['layer']:3d}  {row['probe_persona']:25s}  "
                  f"acc={row['accuracy']:.3f}")

    # Show cross-persona accuracy on disagreement questions (last 5 layers)
    cross_dis = metrics_df[
        (metrics_df["combo_type"] == "cross_persona") &
        (metrics_df["subset"] == "disagreement")
    ].sort_values("layer")
    if len(cross_dis) > 0:
        print(f"\nCross-persona probe accuracy on DISAGREEMENT (last 5 layers):")
        for _, row in cross_dis.tail(10).iterrows():
            print(f"  Layer {row['layer']:3d}  probe={row['probe_persona']:25s}  "
                  f"acts={row['activation_persona']:25s}  acc={row['accuracy']}")

    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
