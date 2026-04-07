"""
Evaluation script for a single training run.

Usage (from project root):

    python evaluate_run.py --run_dir results/epochs5_bs32_lr0p0001_161523

This will:
- Load test predictions and ground-truth labels from the run directory.
- Compute accuracy, precision, recall, F1, ROC-AUC (if probabilities exist).
- Save a CSV with detailed metrics for easy inclusion in reports.
"""

import argparse
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_run(run_dir: str) -> None:
    """Compute evaluation metrics for a given run directory."""
    run_dir = os.path.abspath(run_dir)
    print(f"Evaluating run at: {run_dir}")

    # --------------------------
    # Load predictions
    # --------------------------
    pred_dir = os.path.join(run_dir, "predictions")
    path_y_true = os.path.join(pred_dir, "test_y_true.npy")
    path_y_pred = os.path.join(pred_dir, "test_y_pred.npy")
    path_y_prob = os.path.join(pred_dir, "test_y_prob.npy")

    if not os.path.isfile(path_y_true):
        raise FileNotFoundError(
            f"Ground truth not found: {path_y_true}. "
            "Make sure the training script saved test_y_true.npy."
        )
    if not os.path.isfile(path_y_pred):
        raise FileNotFoundError(
            f"Predictions not found: {path_y_pred}. "
            "Make sure the training script saved test_y_pred.npy."
        )

    y_true = np.load(path_y_true)
    y_pred = np.load(path_y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    # --------------------------
    # Handle probabilities
    # --------------------------
    has_probs = os.path.isfile(path_y_prob)
    if has_probs:
        y_prob = np.load(path_y_prob)

        # If shape is [N,2], take fake class
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
    else:
        y_prob = None
        print("Note: test_y_prob.npy not found; ROC-AUC will be skipped.")

    # --------------------------
    # Metrics
    # --------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_prob is not None:
        try:
            roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            # In case only one class is present in y_true
            roc = float("nan")
    else:
        roc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    # --------------------------
    # Print results
    # --------------------------
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    # --------------------------
    # Save CSV
    # --------------------------
    metrics_csv = os.path.join(run_dir, "detailed_metrics.csv")
    header = (
        "accuracy,precision,recall,f1_score,roc_auc,"
        "cm_00,cm_01,cm_10,cm_11\n"
    )

    # Flatten confusion matrix (2x2 for binary classification).
    if cm.shape == (2, 2):
        cm_flat = cm.flatten()
    else:
        # Fallback: pad / truncate to 4 entries to keep CSV shape consistent.
        flat = cm.flatten()
        cm_flat = np.zeros(4, dtype=int)
        cm_flat[: min(4, flat.size)] = flat[:4]

    row = (
        f"{acc:.6f},{prec:.6f},{rec:.6f},{f1:.6f},{roc:.6f},"
        f"{cm_flat[0]},{cm_flat[1]},{cm_flat[2]},{cm_flat[3]}\n"
    )

    file_exists = os.path.isfile(metrics_csv)
    with open(metrics_csv, "w", encoding="utf-8") as f:
        if not file_exists:
            f.write(header)
        f.write(row)

    print(f"\nSaved detailed metrics to: {metrics_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline deepfake classifier run."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the results directory (e.g. results/epochs5_bs32_lr0p0001_161523).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_run(args.run_dir)

