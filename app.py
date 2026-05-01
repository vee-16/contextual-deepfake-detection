"""
Streamlit demo for the deepfake detection pipeline.

Run with:
    streamlit run app.py -- --checkpoint results/<run>/best_model.pth

The app lets a user upload an image and shows:
  - ViT classifier prediction (real vs fake) with probability
  - Saliency map
  - Occlusion map + most-impactful occlusion box
  - CLIP-based contextual reasoning score (typical vs atypical)
"""

import argparse
import io
import sys

import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Reuse the existing functions from your training script so we stay consistent.
from Contextual_Deepfake_Detector import (
    generate_saliency_map,
    generate_occlusion_map_fixed,
    generate_occlusion_map_auto,
)

# CLIP-based context reasoner from your stage-3 module.
# Adjust the import path if the file is named differently.
try:
    from context_reasoning import ContextReasoner
except ImportError:
    ContextReasoner = None


# --------------------------
# CLI args (passed after `--` when running streamlit)
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/best_model.pth",
        help="Path to a trained ViT checkpoint (best_model.pth).",
    )
    # streamlit injects its own args; ignore unknown ones
    args, _ = parser.parse_known_args()
    return args


# --------------------------
# Constants
# --------------------------
CLASS_NAMES = {0: "real", 1: "fake"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# --------------------------
# Cached loaders
# --------------------------
@st.cache_resource(show_spinner="Loading ViT classifier...")
def load_classifier(checkpoint_path: str):
    """Load the ViT_B_16 classifier with the binary head and trained weights."""
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)

    loaded = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        model.load_state_dict(loaded["model_state_dict"])
    else:
        model.load_state_dict(loaded)

    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading CLIP context reasoner...")
def load_context_reasoner():
    if ContextReasoner is None:
        return None
    return ContextReasoner()


# --------------------------
# Helpers
# --------------------------
def denormalize_for_display(image_tensor: torch.Tensor) -> np.ndarray:
    """Undo the mean=0.5, std=0.5 normalization so the image displays correctly."""
    img = image_tensor.detach().cpu().clone().permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5
    return np.clip(img, 0, 1)


def classify(model, image_tensor: torch.Tensor):
    """Run the classifier and return (predicted_class, probs[real, fake])."""
    model.eval()
    with torch.no_grad():
        out = model(image_tensor.unsqueeze(0).to(DEVICE))
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
    return pred, probs


def render_saliency_figure(image_tensor, saliency_map, pred, probs):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(denormalize_for_display(image_tensor))
    axes[0].set_title(f"Original — Pred: {CLASS_NAMES[pred]}")
    axes[0].axis("off")

    axes[1].imshow(saliency_map, cmap="hot")
    axes[1].set_title(
        f"Saliency\nP(real)={probs[0]:.3f}, P(fake)={probs[1]:.3f}"
    )
    axes[1].axis("off")
    fig.tight_layout()
    return fig


def render_occlusion_figure(image_tensor, occlusion_map, best_box, patch_size, probs, pred):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    img = denormalize_for_display(image_tensor)

    axes[0].imshow(img)
    if best_box is not None:
        left, top, right, bottom = best_box
        rect = plt.Rectangle(
            (left, top), right - left, bottom - top,
            fill=False, edgecolor="red", linewidth=2,
        )
        axes[0].add_patch(rect)
    axes[0].set_title(f"Most-impactful occlusion box\nPred: {CLASS_NAMES[pred]}")
    axes[0].axis("off")

    axes[1].imshow(occlusion_map, cmap="hot")
    axes[1].set_title(
        f"Occlusion map (patch={patch_size})\nP(real)={probs[0]:.3f}, P(fake)={probs[1]:.3f}"
    )
    axes[1].axis("off")
    fig.tight_layout()
    return fig


# --------------------------
# Streamlit UI
# --------------------------
def main():
    args = parse_args()

    st.set_page_config(page_title="Deepfake Detector", layout="wide")
    st.title("Deepfake Detector")
    st.caption(f"Device: `{DEVICE}` · Checkpoint: `{args.checkpoint}`")

    # Sidebar controls
    st.sidebar.header("Settings")
    show_saliency = st.sidebar.checkbox("Saliency map", value=True)
    show_occlusion = st.sidebar.checkbox("Occlusion map", value=True)
    use_auto_patch = st.sidebar.checkbox("Auto-select occlusion patch size", value=False)
    patch_size = st.sidebar.slider("Occlusion patch size", 8, 64, 32, step=8,
                                   disabled=use_auto_patch)
    stride = st.sidebar.slider("Occlusion stride", 4, 32, 16, step=4,
                               disabled=use_auto_patch)
    show_context = st.sidebar.checkbox("CLIP context reasoning", value=True)

    # Load models
    try:
        model = load_classifier(args.checkpoint)
    except Exception as e:
        st.error(f"Failed to load classifier from `{args.checkpoint}`: {e}")
        st.stop()

    reasoner = load_context_reasoner() if show_context else None
    if show_context and reasoner is None:
        st.warning("CLIP context reasoner not available — check that `context_reasoner.py` is importable.")

    # Upload
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("Upload an image to run the pipeline.")
        return

    pil_image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    image_tensor = TRANSFORM(pil_image)

    # --- Classifier ---
    pred, probs = classify(model, image_tensor)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(pil_image, caption="Uploaded image", use_container_width=True)
    with col2:
        verdict = CLASS_NAMES[pred].upper()
        if pred == 1:
            st.error(f"Prediction: **{verdict}**")
        else:
            st.success(f"Prediction: **{verdict}**")
        st.metric("P(real)", f"{probs[0]:.4f}")
        st.metric("P(fake)", f"{probs[1]:.4f}")

    # --- Saliency ---
    if show_saliency:
        st.subheader("Saliency map")
        with st.spinner("Computing saliency..."):
            saliency_map, sal_pred, sal_probs = generate_saliency_map(model, image_tensor)
        st.pyplot(render_saliency_figure(image_tensor, saliency_map, sal_pred, sal_probs))

    # --- Occlusion ---
    if show_occlusion:
        st.subheader("Occlusion map")
        with st.spinner("Computing occlusion map (this is the slow one)..."):
            if use_auto_patch:
                occ_map, occ_pred, occ_probs, used_patch, best_box = generate_occlusion_map_auto(
                    model, image_tensor
                )
            else:
                occ_map, occ_pred, occ_probs, best_box = generate_occlusion_map_fixed(
                    model, image_tensor, patch_size=patch_size, stride=stride
                )
                used_patch = patch_size
        st.pyplot(render_occlusion_figure(
            image_tensor, occ_map, best_box, used_patch, occ_probs, occ_pred
        ))

    # --- CLIP context reasoning ---
    if show_context and reasoner is not None:
        st.subheader("Contextual reasoning (CLIP)")
        with st.spinner("Scoring context..."):
            ctx = reasoner.score_image(pil_image)

            # Recompute per-prompt similarities here so we don't have to
            # modify the shared context_reasoning.py module. Same math the
            # reasoner does internally, just kept around this time.
            image_emb = reasoner.encode_image(pil_image)
            typical_emb, atypical_emb = reasoner._get_prompt_embeddings()
            typical_sim = reasoner.compute_similarity(image_emb, typical_emb).squeeze(0).cpu().tolist()
            atypical_sim = reasoner.compute_similarity(image_emb, atypical_emb).squeeze(0).cpu().tolist()

            typical_per_prompt = [
                {"prompt": p, "similarity": float(s)}
                for p, s in zip(reasoner.typical_prompts, typical_sim)
            ]
            atypical_per_prompt = [
                {"prompt": p, "similarity": float(s)}
                for p, s in zip(reasoner.atypical_prompts, atypical_sim)
            ]
            top_typical = sorted(typical_per_prompt, key=lambda d: d["similarity"], reverse=True)[:3]
            top_atypical = sorted(atypical_per_prompt, key=lambda d: d["similarity"], reverse=True)[:3]

        c1, c2, c3 = st.columns(3)
        c1.metric("Typical similarity", f"{ctx['typical_score']:.4f}")
        c2.metric("Atypical similarity", f"{ctx['atypical_score']:.4f}")
        c3.metric("Context score", f"{ctx['context_score']:.4f}",
                  help="0 = typical, 1 = atypical")

        label = ctx["predicted_context_label"]
        if label == "atypical":
            st.warning(f"Context label: **{label}**")
        else:
            st.info(f"Context label: **{label}**")

        st.markdown("**Why?** Top-firing prompts (cosine similarity to image):")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("*Looks typical of:*")
            for entry in top_typical:
                st.markdown(f"- `{entry['similarity']:.3f}` — {entry['prompt']}")

        with col_right:
            st.markdown("*Looks atypical because:*")
            for entry in top_atypical:
                st.markdown(f"- `{entry['similarity']:.3f}` — {entry['prompt']}")

        with st.expander("Full per-prompt breakdown"):
            st.markdown("**All typical prompts:**")
            for entry in sorted(typical_per_prompt, key=lambda d: d["similarity"], reverse=True):
                st.markdown(f"- `{entry['similarity']:.3f}` — {entry['prompt']}")
            st.markdown("**All atypical prompts:**")
            for entry in sorted(atypical_per_prompt, key=lambda d: d["similarity"], reverse=True):
                st.markdown(f"- `{entry['similarity']:.3f}` — {entry['prompt']}")


if __name__ == "__main__":
    main()