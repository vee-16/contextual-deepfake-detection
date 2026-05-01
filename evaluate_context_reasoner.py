"""
Evaluate the CLIP context reasoner on the OpenFake test set.

Usage:
    python evaluate_context_reasoner.py \
        --test_path data/openfake/test \
        --output_dir results/context_eval \
        --threshold 0.5
"""

import argparse
import csv
import os
import time

import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from context_reasoning import ContextReasoner
from datasets import load_from_disk


LABEL_MAP = {"real": 0, "fake": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP context reasoner on OpenFake test set."
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/openfake/test",
        help="Path to the test split saved with load_all_chunks-compatible structure.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save results. Defaults to results/context_eval_<timestamp>.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on context_score. Higher = more atypical = fake.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of test samples (for quick smoke tests).",
    )
    parser.add_argument(
        "--save_to_run_dir",
        type=str,
        default=None,
        help=(
            "Optional path to an existing training run directory. If provided, "
            "context score arrays are also saved into <run_dir>/predictions/ "
            "so evaluate_run.py can include them in its analysis."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results", f"context_eval_{timestamp}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving results to: {output_dir}")
    print(f"Threshold: {args.threshold}")

    # Load test set
    print(f"Loading test set from: {args.test_path}")
    if not os.path.isdir(args.test_path):
        raise FileNotFoundError(f"Test path does not exist: {args.test_path}")
    test_dataset = load_from_disk(args.test_path)

    n_total = len(test_dataset) if args.limit is None else min(args.limit, len(test_dataset))
    print(f"Evaluating on {n_total} samples")

    # Load reasoner once — text embeddings cached internally after first call
    reasoner = ContextReasoner(model_name=args.model_name, threshold=args.threshold)
    print(f"Reasoner loaded on device: {reasoner.device}")

    # Collect per-sample results
    y_true = []
    typical_scores = []
    atypical_scores = []
    context_scores = []

    failed_indices = []

    start_time = time.time()
    for idx in tqdm(range(n_total), desc="Scoring images"):
        sample = test_dataset[idx]

        # Handle both raw-PIL and {bytes, path} dict forms
        raw = sample["image"]
        try:
            if isinstance(raw, dict):
                from PIL import Image
                import io
                pil_image = Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
            else:
                pil_image = raw.convert("RGB")
        except Exception as e:
            print(f"Skipping idx={idx} due to image load error: {e}")
            failed_indices.append(idx)
            continue

        label_str = sample["label"]
        if label_str not in LABEL_MAP:
            print(f"Skipping idx={idx} with unexpected label: {label_str}")
            failed_indices.append(idx)
            continue

        try:
            result = reasoner.score_image(pil_image)
        except Exception as e:
            print(f"Skipping idx={idx} due to scoring error: {e}")
            failed_indices.append(idx)
            continue

        y_true.append(LABEL_MAP[label_str])
        typical_scores.append(result["typical_score"])
        atypical_scores.append(result["atypical_score"])
        context_scores.append(result["context_score"])

    elapsed = time.time() - start_time
    n_scored = len(y_true)
    print(f"\nScored {n_scored}/{n_total} images in {elapsed:.1f}s "
          f"({n_scored / max(elapsed, 1e-9):.2f} img/s)")
    if failed_indices:
        print(f"{len(failed_indices)} images failed and were skipped.")

    if n_scored == 0:
        raise RuntimeError("No images were successfully scored. Aborting.")

    y_true = np.array(y_true)
    typical_scores = np.array(typical_scores)
    atypical_scores = np.array(atypical_scores)
    context_scores = np.array(context_scores)

    # Predictions at threshold: context_score >= threshold -> atypical -> fake (1)
    y_pred = (context_scores >= args.threshold).astype(int)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, context_scores)
    except ValueError:
        roc = float("nan")
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Context Reasoner Metrics ===")
    print(f"Threshold     : {args.threshold}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"F1-score      : {f1:.4f}")
    print(f"ROC-AUC       : {roc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Per-class score means — useful sanity check
    real_mask = y_true == 0
    fake_mask = y_true == 1
    if real_mask.any():
        print(f"\nMean context_score (real) : {context_scores[real_mask].mean():.4f}")
    if fake_mask.any():
        print(f"Mean context_score (fake) : {context_scores[fake_mask].mean():.4f}")

    # Save raw arrays
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    np.save(os.path.join(pred_dir, "context_y_true.npy"), y_true)
    np.save(os.path.join(pred_dir, "context_y_pred.npy"), y_pred)
    np.save(os.path.join(pred_dir, "context_score.npy"), context_scores)
    np.save(os.path.join(pred_dir, "context_typical_score.npy"), typical_scores)
    np.save(os.path.join(pred_dir, "context_atypical_score.npy"), atypical_scores)

    # Optionally also write into an existing run's predictions/ folder so
    # evaluate_run.py can pick them up alongside the attention scores.
    if args.save_to_run_dir is not None:
        run_pred_dir = os.path.join(args.save_to_run_dir, "predictions")
        if not os.path.isdir(run_pred_dir):
            print(f"Warning: {run_pred_dir} does not exist; skipping run-dir copy.")
        else:
            # Sanity check — refuse to write if test_y_true differs in size.
            run_y_true_path = os.path.join(run_pred_dir, "test_y_true.npy")
            if os.path.isfile(run_y_true_path):
                run_y_true = np.load(run_y_true_path)
                if len(run_y_true) != len(y_true):
                    print(
                        f"Warning: run_dir test_y_true has {len(run_y_true)} samples "
                        f"but CLIP eval scored {len(y_true)}. "
                        "These were probably evaluated on different subsets — "
                        "NOT saving to run_dir to avoid mixing arrays."
                    )
                else:
                    np.save(os.path.join(run_pred_dir, "test_context_score.npy"), context_scores)
                    np.save(os.path.join(run_pred_dir, "test_context_typical_score.npy"), typical_scores)
                    np.save(os.path.join(run_pred_dir, "test_context_atypical_score.npy"), atypical_scores)
                    print(f"Also saved CLIP score arrays to: {run_pred_dir}")
            else:
                print(f"Warning: {run_y_true_path} missing; skipping run-dir copy.")

    # Aggregate metrics CSV
    metrics_csv = os.path.join(output_dir, "context_metrics.csv")
    cm_flat = cm.flatten() if cm.shape == (2, 2) else np.zeros(4, dtype=int)
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_samples", "threshold",
            "accuracy", "precision", "recall", "f1_score", "roc_auc",
            "cm_00", "cm_01", "cm_10", "cm_11",
            "mean_context_score_real", "mean_context_score_fake",
        ])
        writer.writerow([
            n_scored, args.threshold,
            f"{acc:.6f}", f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}", f"{roc:.6f}",
            int(cm_flat[0]), int(cm_flat[1]), int(cm_flat[2]), int(cm_flat[3]),
            f"{context_scores[real_mask].mean():.6f}" if real_mask.any() else "nan",
            f"{context_scores[fake_mask].mean():.6f}" if fake_mask.any() else "nan",
        ])
    print(f"\nSaved aggregate metrics to: {metrics_csv}")

    # Per-sample CSV
    per_sample_csv = os.path.join(output_dir, "context_per_sample.csv")
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_idx", "true_label", "pred_label",
            "typical_score", "atypical_score", "context_score",
        ])
        for i in range(n_scored):
            writer.writerow([
                i,
                int(y_true[i]),
                int(y_pred[i]),
                f"{typical_scores[i]:.6f}",
                f"{atypical_scores[i]:.6f}",
                f"{context_scores[i]:.6f}",
            ])
    print(f"Saved per-sample scores to: {per_sample_csv}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()