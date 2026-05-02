import argparse
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from Contextual_Deepfake_Detector import (
    build_model,
    transform,
    denormalize_image,
    generate_saliency_map,
    generate_occlusion_map_fixed,
    extract_all_attention_weights,
    compute_patch_perplexity,
    compute_cls_entropy,
    compute_attention_rollout,
)

from context_reasoning import ContextReasoner


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "real", 1: "fake"}


def load_model(checkpoint_path):
    model = build_model()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def load_image(image_path):
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = transform(pil_image)
    return pil_image, image_tensor


def predict_image(model, image_tensor):
    model.eval()

    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))

    return pred_class, probs


def normalize_map(arr):
    arr = arr.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return arr


def patch_scores_to_heatmap(patch_scores, grid_size=14):
    patch_scores = normalize_map(patch_scores)
    grid = patch_scores.reshape(grid_size, grid_size)

    scale = 224 // grid_size
    heatmap = np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)

    return heatmap


def save_explanation_figure(
    image_tensor,
    saliency_map,
    occlusion_map,
    rollout_heatmap,
    best_box,
    pred_class,
    probs,
    context_result,
    output_path,
):
    image = denormalize_image(image_tensor)

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title(
        f"Original\nPred: {CLASS_NAMES[pred_class]} | "
        f"P(real)={probs[0]:.4f}, P(fake)={probs[1]:.4f}"
    )
    plt.axis("off")

    if best_box is not None:
        left, top, right, bottom = best_box
        rect = plt.Rectangle(
            (left, top),
            right - left,
            bottom - top,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
        plt.gca().add_patch(rect)

    plt.subplot(2, 2, 2)
    plt.imshow(saliency_map, cmap="hot")
    plt.title("Saliency Map")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(image)
    plt.imshow(occlusion_map, cmap="hot", alpha=0.5)
    plt.title("Occlusion Map")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(image)
    plt.imshow(rollout_heatmap, cmap="hot", alpha=0.5)
    plt.title("Attention Rollout Heatmap")
    plt.axis("off")

    plt.suptitle(
        "CLIP Context: "
        f"{context_result['predicted_context_label']} | "
        f"context_score={context_result['context_score']:.4f}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def explain_image(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint)
    pil_image, image_tensor = load_image(args.image)

    pred_class, probs = predict_image(model, image_tensor)

    saliency_map, _, _ = generate_saliency_map(
        model,
        image_tensor,
        target_class=pred_class,
    )

    occlusion_map, _, _, best_box = generate_occlusion_map_fixed(
        model,
        image_tensor,
        patch_size=args.patch_size,
        stride=args.stride,
        target_class=pred_class,
    )

    all_attn = extract_all_attention_weights(model, image_tensor)
    last_layer_attn = all_attn[-1]

    perplexity = compute_patch_perplexity(last_layer_attn)
    cls_entropy = compute_cls_entropy(last_layer_attn)
    rollout = compute_attention_rollout(all_attn)

    rollout_heatmap = patch_scores_to_heatmap(rollout)

    reasoner = ContextReasoner(threshold=args.context_threshold)
    context_result = reasoner.score_image(pil_image)

    figure_path = os.path.join(args.output_dir, "explanation_report.png")
    save_explanation_figure(
        image_tensor=image_tensor,
        saliency_map=saliency_map,
        occlusion_map=occlusion_map,
        rollout_heatmap=rollout_heatmap,
        best_box=best_box,
        pred_class=pred_class,
        probs=probs,
        context_result=context_result,
        output_path=figure_path,
    )

    report = {
        "image_path": args.image,
        "predicted_class": pred_class,
        "predicted_label": CLASS_NAMES[pred_class],
        "prob_real": float(probs[0]),
        "prob_fake": float(probs[1]),
        "confidence": float(max(probs)),
        "best_occlusion_box": best_box,
        "mean_patch_perplexity": float(np.mean(perplexity)),
        "cls_entropy": float(cls_entropy),
        "rollout_std": float(np.std(rollout)),
        "context_reasoning": context_result,
        "figure_path": figure_path,
    }

    json_path = os.path.join(args.output_dir, "explanation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n=== Explanation Report ===")
    print(f"Prediction: {CLASS_NAMES[pred_class]}")
    print(f"P(real): {probs[0]:.4f}")
    print(f"P(fake): {probs[1]:.4f}")
    print(f"Confidence: {max(probs):.4f}")
    print(f"Context label: {context_result['predicted_context_label']}")
    print(f"Context score: {context_result['context_score']:.4f}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved JSON: {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a confidence and explanation report for one image."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image to explain.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained ViT checkpoint.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/single_image_explanation",
        help="Directory where report files will be saved.",
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=32,
        help="Occlusion patch size.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Occlusion stride.",
    )

    parser.add_argument(
        "--context_threshold",
        type=float,
        default=0.5,
        help="CLIP context threshold.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    explain_image(args)