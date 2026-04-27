"""
Evaluation script for a single training run.

Usage (from project root):

    python evaluate_run.py --run_dir results/epochs5_bs32_lr0p0001_161523

This will:
- Load test predictions and ground-truth labels from the run directory.
- Compute accuracy, precision, recall, F1, ROC-AUC.
- Load and analyze three attention-based anomaly scores:
    1. Perplexity       — per-patch local anomaly (last layer attention entropy)
    2. CLS entropy      — global scene incoherence (CLS token attention spread)
    3. Attention rollout — object-relationship signal across all 12 layers
- Compute a combined weighted anomaly score per image.
- Save per-sample scores and aggregate metrics to CSV.
- Save patch heatmap visualizations for perplexity and rollout.
"""

import argparse
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from initialize_dataset import load_all_chunks
from torchvision import transforms
from torch.utils.data import Dataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# --------------------------
# Weights for combined score
# Tunable — these reflect equal contribution by default.
# Perplexity (mean aggregated) and rollout (std aggregated) over 196 patches
# --------------------------
ALPHA = 0.33   # perplexity weight
BETA  = 0.33   # CLS entropy weight
GAMMA = 0.34   # rollout weight

def evaluate_run(run_dir: str) -> dict:
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

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc,
        "confusion_matrix": cm,
    }


# --------------------------
# Load attention scores
# --------------------------

def load_attention_scores(run_dir: str) -> dict:
    """
    Load the three attention-based score arrays saved by the training script.

    Expected files in predictions/:
        test_perplexity.npy  — shape [N, 196]  per-patch perplexity per image
        test_cls_entropy.npy — shape [N]        CLS entropy scalar per image
        test_rollout.npy     — shape [N, 196]  rollout patch score per image

    Returns dict with arrays, or None for missing files.
    """
    pred_dir = os.path.join(run_dir, "predictions")

    scores = {}
    for key, fname in [
        ("perplexity", "test_perplexity.npy"),
        ("cls_entropy", "test_cls_entropy.npy"),
        ("rollout",     "test_rollout.npy"),
    ]:
        path = os.path.join(pred_dir, fname)
        if os.path.isfile(path):
            scores[key] = np.load(path)
            print(f"Loaded {fname}: shape {scores[key].shape}")
        else:
            scores[key] = None
            print(f"Warning: {fname} not found — {key} analysis will be skipped.")

    return scores


# --------------------------
# Normalize scores
# --------------------------

def min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D array to [0, 1] range.
    Used so that perplexity, CLS entropy, and rollout are on comparable scales
    before computing the combined anomaly score.
    """
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# --------------------------
# Combined anomaly score
# --------------------------

def compute_combined_anomaly_score(
    perplexity: np.ndarray,   # [N, 196]
    cls_entropy: np.ndarray,  # [N]
    rollout: np.ndarray,      # [N, 196]
) -> np.ndarray:
    """
    Combine the three scores into a single per-image anomaly score.

    Each component is reduced to a scalar per image first:
    - Perplexity: mean over patches [N, 196] -> [N]
    - CLS entropy: already scalar [N]
    - Rollout: std over patches [N, 196] -> [N]
      (high std = uneven patch influence = anomalous object relationships)

    Then each is normalized to [0,1] and combined with weights ALPHA/BETA/GAMMA.

    Higher combined score = more likely fake.

    Args:
        perplexity: per-patch perplexity [N, 196]
        cls_entropy: per-image CLS entropy [N]
        rollout: per-patch rollout score [N, 196]

    Returns:
        combined: [N] anomaly score per image
    """
    mean_perplexity = perplexity.mean(axis=1)   # [N]
    rollout_std     = rollout.std(axis=1)        # [N] — uneven rollout = suspicious

    # Negate CLS entropy: real images have higher raw entropy, so we flip
    # so that higher score = more likely fake, consistent with other scores
    neg_cls_entropy = -cls_entropy

    norm_perp = min_max_normalize(mean_perplexity)
    norm_cls  = min_max_normalize(neg_cls_entropy)
    norm_roll = min_max_normalize(rollout_std)

    combined = ALPHA * norm_perp + BETA * norm_cls + GAMMA * norm_roll
    return combined


# --------------------------
# Attention score analysis
# --------------------------

def analyze_attention_scores(
    scores: dict,
    y_true: np.ndarray,
    run_dir: str,
) -> dict:
    """
    Analyze the three attention scores against ground-truth labels.

    For each score, compute:
    - Mean score for real images (label=0) vs fake images (label=1)
    - How well the score separates real from fake (ROC-AUC of score vs label)

    Also computes the combined anomaly score and its ROC-AUC.

    Args:
        scores: dict from load_attention_scores
        y_true: ground truth labels [N]
        run_dir: path to save results

    Returns:
        analysis: dict of per-score stats
    """
    analysis = {}

    # --- Perplexity ---
    if scores["perplexity"] is not None:
        mean_perp = scores["perplexity"].mean(axis=1)    # [N]
        real_mean = mean_perp[y_true == 0].mean()
        fake_mean = mean_perp[y_true == 1].mean()

        try:
            perp_auc = roc_auc_score(y_true, mean_perp)
        except ValueError:
            perp_auc = float("nan")

        analysis["perplexity"] = {
            "real_mean": real_mean,
            "fake_mean": fake_mean,
            "auc": perp_auc,
        }

        print(f"\n=== Perplexity Scores ===")
        print(f"Mean perplexity (real images): {real_mean:.4f}")
        print(f"Mean perplexity (fake images): {fake_mean:.4f}")
        print(f"Perplexity ROC-AUC vs fake label: {perp_auc:.4f}")
        print("(Higher AUC = perplexity better separates real vs fake)")

    # --- CLS Entropy ---
    if scores["cls_entropy"] is not None:
        cls = scores["cls_entropy"]
        real_mean = cls[y_true == 0].mean()
        fake_mean = cls[y_true == 1].mean()

        # Real images have higher CLS entropy than fake — the model is more
        # uncertain about real scenes because they are more complex/varied.
        # Fake images tend to have lower CLS entropy (more focused attention).
        # We negate the score so that higher value = more likely fake,
        # consistent with perplexity and rollout directions.
        cls_for_auc = -cls
        try:
            cls_auc = roc_auc_score(y_true, cls_for_auc)
        except ValueError:
            cls_auc = float("nan")

        analysis["cls_entropy"] = {
            "real_mean": real_mean,
            "fake_mean": fake_mean,
            "auc": cls_auc,
            "inverted": True,   # flag so combined score knows to negate
        }

        print(f"\n=== CLS Entropy Scores ===")
        print(f"Mean CLS entropy (real images): {real_mean:.4f}")
        print(f"Mean CLS entropy (fake images): {fake_mean:.4f}")
        print(f"CLS entropy ROC-AUC vs fake label: {cls_auc:.4f}")
        print("(Note: CLS entropy is negated — real images have higher raw entropy)")
        print("(Higher AUC = negated CLS entropy better separates real vs fake)")

    # --- Rollout ---
    if scores["rollout"] is not None:
        rollout_std = scores["rollout"].std(axis=1)       # [N]
        real_mean = rollout_std[y_true == 0].mean()
        fake_mean = rollout_std[y_true == 1].mean()

        try:
            roll_auc = roc_auc_score(y_true, rollout_std)
        except ValueError:
            roll_auc = float("nan")

        analysis["rollout"] = {
            "real_mean": real_mean,
            "fake_mean": fake_mean,
            "auc": roll_auc,
        }

        print(f"\n=== Rollout Scores ===")
        print(f"Mean rollout std (real images): {real_mean:.4f}")
        print(f"Mean rollout std (fake images): {fake_mean:.4f}")
        print(f"Rollout ROC-AUC vs fake label: {roll_auc:.4f}")
        print("(Higher std = uneven patch influence = more anomalous scene structure)")

    # --- Combined score ---
    if all(scores[k] is not None for k in ["perplexity", "cls_entropy", "rollout"]):
        combined = compute_combined_anomaly_score(
            scores["perplexity"],
            scores["cls_entropy"],
            scores["rollout"],
        )
        real_mean = combined[y_true == 0].mean()
        fake_mean = combined[y_true == 1].mean()

        try:
            comb_auc = roc_auc_score(y_true, combined)
        except ValueError:
            comb_auc = float("nan")

        analysis["combined"] = {
            "scores": combined,
            "real_mean": real_mean,
            "fake_mean": fake_mean,
            "auc": comb_auc,
        }

        print(f"\n=== Combined Anomaly Score (α={ALPHA} · perp + β={BETA} · cls + γ={GAMMA} · roll) ===")
        print(f"Mean combined score (real images): {real_mean:.4f}")
        print(f"Mean combined score (fake images): {fake_mean:.4f}")
        print(f"Combined ROC-AUC vs fake label: {comb_auc:.4f}")

    return analysis


# --------------------------
# Heatmap visualization
# --------------------------

def heatmap_to_image(patch_scores: np.ndarray, grid_size: int = 14) -> np.ndarray:
    """
    Reshape [196] patch scores into a [14, 14] heatmap and upsample to [224, 224]
    so it can be overlaid on the original 224x224 image.

    Args:
        patch_scores: [196] scores, one per patch
        grid_size: 14 for ViT_B_16 (224 / 16 patch size = 14)

    Returns:
        heatmap: [224, 224] upsampled heatmap
    """
    grid = patch_scores.reshape(grid_size, grid_size)

    # Upsample from 14x14 to 224x224 using nearest-neighbor repeat
    scale = 224 // grid_size   # = 16
    heatmap = np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)
    return heatmap


def save_attention_heatmaps(
    scores: dict,
    loader_dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run_dir: str,
    num_samples: int = 5,
) -> None:
    """
    Save side-by-side visualizations for a random sample of test images showing:
    - Original image
    - Perplexity heatmap (local patch anomalies)
    - Rollout heatmap (object-relationship influence map)

    Each saved as a PNG in results/<run>/attention_heatmaps/

    Args:
        scores: dict from load_attention_scores
        loader_dataset: the test dataset (supports index access)
        y_true: ground truth labels [N]
        y_pred: predicted labels [N]
        run_dir: path to run directory
        num_samples: how many images to visualize
    """
    if scores["perplexity"] is None and scores["rollout"] is None:
        print("Skipping heatmap visualization — no attention scores available.")
        return

    heatmap_dir = os.path.join(run_dir, "attention_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    class_names = {0: "real", 1: "fake"}
    N = len(y_true)
    selected = np.random.choice(N, size=min(num_samples, N), replace=False)

    for idx in selected:
        image_tensor, _ = loader_dataset[idx]

        # Undo normalization for display
        image = image_tensor.permute(1, 2, 0).numpy()
        image = (image * 0.5) + 0.5
        image = np.clip(image, 0, 1)

        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        correct = "✓" if y_true[idx] == y_pred[idx] else "✗"

        # Determine subplot count based on available scores
        has_perp = scores["perplexity"] is not None
        has_roll = scores["rollout"] is not None
        num_plots = 1 + int(has_perp) + int(has_roll)

        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
        if num_plots == 1:
            axes = [axes]

        col = 0

        # Original image
        axes[col].imshow(image)
        axes[col].set_title(
            f"Original\nTrue: {true_label} | Pred: {pred_label} {correct}",
            fontsize=9
        )
        axes[col].axis("off")
        col += 1

        # Perplexity heatmap
        if has_perp:
            perp_scores = scores["perplexity"][idx]          # [196]
            perp_norm = (perp_scores - perp_scores.min()) / (perp_scores.max() - perp_scores.min() + 1e-9)
            perp_heatmap = heatmap_to_image(perp_norm)

            axes[col].imshow(image)
            axes[col].imshow(perp_heatmap, cmap="hot", alpha=0.5)
            axes[col].set_title(
                f"Perplexity Heatmap\nmean={perp_scores.mean():.3f}",
                fontsize=9
            )
            axes[col].axis("off")
            col += 1

        # Rollout heatmap
        if has_roll:
            roll_scores = scores["rollout"][idx]             # [196]
            roll_norm = (roll_scores - roll_scores.min()) / (roll_scores.max() - roll_scores.min() + 1e-9)
            roll_heatmap = heatmap_to_image(roll_norm)

            axes[col].imshow(image)
            axes[col].imshow(roll_heatmap, cmap="Blues", alpha=0.5)
            axes[col].set_title(
                f"Rollout Heatmap\nstd={roll_scores.std():.3f}",
                fontsize=9
            )
            axes[col].axis("off")
            col += 1

        plt.suptitle(
            f"Sample {idx} — Attention Analysis",
            fontsize=10, fontweight="bold"
        )
        plt.tight_layout()

        fname = f"sample_{idx}_true_{true_label}_pred_{pred_label}.png"
        save_path = os.path.join(heatmap_dir, fname)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

    print(f"Saved {len(selected)} attention heatmap(s) to: {heatmap_dir}")


# --------------------------
# Save CSV
# --------------------------

def save_results_csv(
    run_dir: str,
    standard_metrics: dict,
    analysis: dict,
    scores: dict,
) -> None:
    """
    Save two CSV files:

    1. detailed_metrics.csv — one row with all aggregate metrics
       (standard classification + attention score AUCs + combined score AUC)

    2. per_sample_scores.csv — one row per test image with:
       sample index, true label, predicted label, prediction probability,
       mean perplexity, CLS entropy, rollout std, combined anomaly score

    Args:
        run_dir: path to run directory
        standard_metrics: dict from evaluate_run()
        analysis: dict from analyze_attention_scores()
        scores: dict from load_attention_scores()
    """
    # --- Aggregate metrics CSV ---
    metrics_csv = os.path.join(run_dir, "detailed_metrics.csv")

    cm = standard_metrics["confusion_matrix"]
    if cm.shape == (2, 2):
        cm_flat = cm.flatten()
    else:
        flat = cm.flatten()
        cm_flat = np.zeros(4, dtype=int)
        cm_flat[: min(4, flat.size)] = flat[:4]

    perp_auc = analysis.get("perplexity", {}).get("auc", float("nan"))
    cls_auc  = analysis.get("cls_entropy", {}).get("auc", float("nan"))
    roll_auc = analysis.get("rollout", {}).get("auc", float("nan"))
    comb_auc = analysis.get("combined", {}).get("auc", float("nan"))

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "accuracy", "precision", "recall", "f1_score", "roc_auc",
            "cm_00", "cm_01", "cm_10", "cm_11",
            "perplexity_auc", "cls_entropy_auc", "rollout_auc", "combined_anomaly_auc",
        ])
        writer.writerow([
            f"{standard_metrics['accuracy']:.6f}",
            f"{standard_metrics['precision']:.6f}",
            f"{standard_metrics['recall']:.6f}",
            f"{standard_metrics['f1_score']:.6f}",
            f"{standard_metrics['roc_auc']:.6f}",
            cm_flat[0], cm_flat[1], cm_flat[2], cm_flat[3],
            f"{perp_auc:.6f}",
            f"{cls_auc:.6f}",
            f"{roll_auc:.6f}",
            f"{comb_auc:.6f}",
        ])

    print(f"\nSaved aggregate metrics to: {metrics_csv}")

    y_true = standard_metrics["y_true"]
    y_pred = standard_metrics["y_pred"]
    y_prob = standard_metrics["y_prob"]
    N = len(y_true)

    mean_perp   = scores["perplexity"].mean(axis=1) if scores["perplexity"] is not None else np.full(N, float("nan"))
    cls_entropy = scores["cls_entropy"]             if scores["cls_entropy"] is not None else np.full(N, float("nan"))
    roll_std    = scores["rollout"].std(axis=1)     if scores["rollout"] is not None     else np.full(N, float("nan"))
    combined    = analysis.get("combined", {}).get("scores", np.full(N, float("nan")))
    prob_fake   = y_prob if y_prob is not None else np.full(N, float("nan"))

    per_sample_csv = os.path.join(run_dir, "per_sample_scores.csv")
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_idx", "true_label", "pred_label", "prob_fake",
            "mean_perplexity", "cls_entropy", "rollout_std", "combined_anomaly_score",
        ])
        for i in range(N):
            writer.writerow([
                i,
                int(y_true[i]),
                int(y_pred[i]),
                f"{prob_fake[i]:.6f}",
                f"{mean_perp[i]:.6f}",
                f"{cls_entropy[i]:.6f}",
                f"{roll_std[i]:.6f}",
                f"{combined[i]:.6f}" if not np.isnan(combined[i]) else "nan",
            ])

    print(f"Saved per-sample scores to: {per_sample_csv}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline deepfake classifier run with attention-based anomaly scoring."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the results directory (e.g. results/vit_e5_bs32_lr0p0001_...).",
    )
    parser.add_argument(
        "--num_heatmaps",
        type=int,
        default=5,
        help="Number of attention heatmap visualizations to save.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/openfake/test",
        help="Path to test dataset (needed to load images for heatmap visualization).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)

    # Step 1: Standard metrics
    standard_metrics = evaluate_run(run_dir)

    # Step 2: Load attention scores saved by training script
    scores = load_attention_scores(run_dir)

    # Step 3: Analyze scores vs ground truth
    analysis = analyze_attention_scores(
        scores=scores,
        y_true=standard_metrics["y_true"],
        run_dir=run_dir,
    )

    # Step 4: Save heatmap visualizations
    # Load test dataset to access raw images
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        class _TestDataset(Dataset):
            def __init__(self, hf_dataset, transform):
                self.dataset = hf_dataset
                self.transform = transform
                self.label_map = {"real": 0, "fake": 1}

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                sample = self.dataset[idx]
                raw = sample["image"]

                # load_from_disk returns image as a dict {"bytes": ..., "path": ...}
                # or as a PIL Image depending on dataset version — handle both
                if isinstance(raw, dict):
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
                else:
                    image = raw.convert("RGB")

                label = self.label_map[sample["label"]]
                if self.transform:
                    image = self.transform(image)
                return image, label

        hf_test = load_all_chunks(args.dataset_path)
        test_dataset = _TestDataset(hf_test, transform)

        save_attention_heatmaps(
            scores=scores,
            loader_dataset=test_dataset,
            y_true=standard_metrics["y_true"],
            y_pred=standard_metrics["y_pred"],
            run_dir=run_dir,
            num_samples=args.num_heatmaps,
        )
    except Exception as e:
        print(f"Warning: Could not load dataset for heatmaps — {e}")
        print("Skipping heatmap visualization.")

    # Step 5: Save all results to CSV
    save_results_csv(
        run_dir=run_dir,
        standard_metrics=standard_metrics,
        analysis=analysis,
        scores=scores,
    )

    print("\nEvaluation complete.")