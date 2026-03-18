"""
evaluate_experiments.py

Evaluate one or multiple experiment runs and generate clean, reusable
evaluation artifacts for reports.

Usage:

Single run:
    python evaluate_experiments.py --run_dir results/<run_name>

All runs:
    python evaluate_experiments.py --all_runs --results_root results
"""

import argparse
import os
import json
import numpy as np
import csv

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)


# --------------------------
# Core evaluation function
# --------------------------
def evaluate_single_run(run_dir):
    print(f"\nEvaluating: {run_dir}")

    y_true_path = os.path.join(run_dir, "test_y_true.npy")
    y_pred_path = os.path.join(run_dir, "test_y_pred.npy")
    y_prob_path = os.path.join(run_dir, "test_y_prob.npy")

    if not os.path.exists(y_true_path) or not os.path.exists(y_pred_path):
        print("Missing required prediction files. Skipping.")
        return None

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)

    has_probs = os.path.exists(y_prob_path)
    y_prob = np.load(y_prob_path) if has_probs else None

    # --------------------------
    # Metrics
    # --------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if has_probs:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except:
            roc_auc = float("nan")
    else:
        roc_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    # PR AUC
    if has_probs:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
    else:
        pr_auc = float("nan")

    # --------------------------
    # Save ROC Curve
    # --------------------------
    if has_probs:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_csv_path = os.path.join(run_dir, "roc_curve.csv")

        with open(roc_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fpr", "tpr", "threshold"])
            for i in range(len(fpr)):
                writer.writerow([fpr[i], tpr[i], thresholds[i]])

    # --------------------------
    # Save PR Curve
    # --------------------------
    if has_probs:
        pr_csv_path = os.path.join(run_dir, "pr_curve.csv")

        with open(pr_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["precision", "recall"])
            for i in range(len(precision_curve)):
                writer.writerow([precision_curve[i], recall_curve[i]])

    # --------------------------
    # Save summary JSON
    # --------------------------
    summary = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }

    json_path = os.path.join(run_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    # --------------------------
    # Save summary CSV
    # --------------------------
    csv_path = os.path.join(run_dir, "metrics_summary.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "accuracy", "precision", "recall",
            "f1_score", "roc_auc", "pr_auc"
        ])
        writer.writerow([
            acc, prec, rec, f1, roc_auc, pr_auc
        ])

    print("Saved metrics + curves")

    return summary


# --------------------------
# Evaluate all runs
# --------------------------
def evaluate_all_runs(results_root):
    print(f"\nEvaluating all runs in: {results_root}")

    all_results = []

    for run_name in os.listdir(results_root):
        run_dir = os.path.join(results_root, run_name)

        if not os.path.isdir(run_dir):
            continue

        result = evaluate_single_run(run_dir)
        if result:
            result["run_name"] = run_name
            all_results.append(result)

    # Save combined CSV
    if all_results:
        summary_path = os.path.join(results_root, "all_runs_summary.csv")

        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "run_name",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "pr_auc"
            ])

            for r in all_results:
                writer.writerow([
                    r["run_name"],
                    r["accuracy"],
                    r["precision"],
                    r["recall"],
                    r["f1_score"],
                    r["roc_auc"],
                    r["pr_auc"]
                ])

        print(f"\nSaved combined results: {summary_path}")


# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", type=str, help="Single run directory")
    parser.add_argument("--all_runs", action="store_true", help="Evaluate all runs")
    parser.add_argument("--results_root", type=str, default="results")

    args = parser.parse_args()

    if args.all_runs:
        evaluate_all_runs(args.results_root)
    elif args.run_dir:
        evaluate_single_run(args.run_dir)
    else:
        print("Please provide --run_dir or --all_runs")


if __name__ == "__main__":
    main()