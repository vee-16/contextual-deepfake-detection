import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import subprocess
import random

from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import random_split, DataLoader, Dataset
from datasets import load_from_disk
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO

# --------------------------
# Config
# --------------------------
# Paths and directories
TRAIN_PATH = "data/openfake/train"
TEST_PATH = "data/openfake/test"
RESULTS_DIR = "results"

# Create folders if they don't already exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# These will be set from command line arguments later
BATCH_SIZE = None
LEARNING_RATE = None
EPOCHS = None


# --------------------------
# Device
# --------------------------
# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# Transforms
# --------------------------
# Standard ViT preprocessing: resize to 224x224, convert to tensor, and normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# --------------------------
# Dataset wrapper
# --------------------------
# Converts HuggingFace dataset format into a PyTorch Dataset
# so it works nicely with DataLoader
class OpenFakeTorchDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

        # Map string labels to integers
        self.label_map = {
            "real": 0,
            "fake": 1
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # load_from_disk may return image as a dict {"bytes": ..., "path": ...}
        # or as a PIL Image depending on dataset version — handle both safely
        img_data = sample["image"]
        try:
            if isinstance(img_data, dict) and "bytes" in img_data:
                image = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
            else:
                image = img_data.convert("RGB")
        except Exception:
            # Fallback for corrupted images
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        label_str = sample["label"]

        # Basic sanity check
        if label_str not in self.label_map:
            raise ValueError(f"Unexpected label: {label_str}")

        label = self.label_map[label_str]

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------------
# Data loading
# --------------------------
# Loads dataset from disk and builds PyTorch dataloaders
def build_dataloaders():
    hf_train_dataset = load_from_disk(TRAIN_PATH)
    hf_test_dataset = load_from_disk(TEST_PATH)

    # Print dataset info so we know things loaded correctly
    print("Train set size:", len(hf_train_dataset))
    print("Test set size:", len(hf_test_dataset))
    print("Train columns:", hf_train_dataset.column_names)
    print("Test columns:", hf_test_dataset.column_names)

    unique_train_labels = set(hf_train_dataset["label"])
    unique_test_labels = set(hf_test_dataset["label"])

    print("Unique train labels:", unique_train_labels)
    print("Unique test labels:", unique_test_labels)
    print("Train label counts:", Counter(hf_train_dataset["label"]))
    print("Test label counts:", Counter(hf_test_dataset["label"]))

    # Print example sample just to check structure
    sample = hf_train_dataset[0]
    print("\nExample sample:")
    print("Label:", sample["label"])
    print("Prompt:", sample.get("prompt", None))
    print("Model:", sample.get("model", None))
    print("Image type:", type(sample["image"]))

    # Wrap datasets so they work with PyTorch
    full_train_dataset = OpenFakeTorchDataset(hf_train_dataset, transform=transform)
    full_test_dataset = OpenFakeTorchDataset(hf_test_dataset, transform=transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Split training data into train + validation
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    test_dataset = full_test_dataset

    # Dataloaders handle batching and shuffling
    use_pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=use_pin_memory)

    print("\nAfter split:")
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    return train_loader, val_loader, test_loader


# --------------------------
# Model
# --------------------------
# Load pretrained Vision Transformer and adapt it for binary classification
def build_model():
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    # Replace classification head for binary classification
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)

    return model.to(device)


# --------------------------
# Train / Evaluate
# --------------------------
# Runs one full training epoch
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    # tqdm just shows a progress bar
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# Evaluate model performance
def evaluate(model, loader):
    # Switch model to evaluation mode
    model.eval()

    # Prediction and true labels
    all_preds = []
    all_labels = []

    # Disable gradient tracking since we are not training
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):

            # Move data to GPU or CPU depending on device
            images = images.to(device)
            labels = labels.to(device)

            # Run forward pass through the model
            outputs = model(images)

            # outputs contains raw class scores
            # torch.max selects the class with the highest score
            _, predicted = torch.max(outputs, 1)

            # Move predictions back to CPU and store them
            all_preds.extend(predicted.cpu().numpy())

            # Store true labels
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    # Accuracy = how many predictions were correct
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    # Precision = of the samples predicted fake, how many were actually fake
    precision = precision_score(all_labels, all_preds, zero_division=0)

    # Recall = of all real fake images, how many did we correctly detect
    recall = recall_score(all_labels, all_preds, zero_division=0)

    # F1 = harmonic mean of precision and recall
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # [[true_real predicted_real, true_real predicted_fake]
    #  [true_fake predicted_real, true_fake predicted_fake]]
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    return metrics


# --------------------------
# Collect predictions for evaluation
# --------------------------
def collect_predictions(model, loader):
    """
    Run the model on a dataloader and collect:
    - y_true: ground-truth labels
    - y_pred: predicted labels
    - y_prob: probability of the 'fake' class (class index 1)
    These are saved for later analysis in evaluation scripts.
    """
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting predictions", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Convert raw logits to probabilities for the fake class (index 1)
            probs_fake = torch.softmax(outputs, dim=1)[:, 1]

            _, predicted = torch.max(outputs, 1)

            all_probs.extend(probs_fake.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return y_true, y_pred, y_prob


# --------------------------
# Attention extraction
# --------------------------
def extract_all_attention_weights(model, image_tensor):
    """
    Extract attention weights from ALL 12 transformer blocks of ViT_B_16.

    torchvision's MHA internally calls F.multi_head_attention_forward with
    need_weights=False, so forward hooks always receive None for attn_weights.
    The fix is to manually run each encoder layer step-by-step, calling the
    self-attention module directly with need_weights=True.

    We replicate the ViT encoder forward pass:
      1. Patch embedding + CLS token + positional embedding  (model.encoder)
      2. For each of the 12 EncoderBlock layers:
           a. LayerNorm
           b. Self-attention (called directly with need_weights=True)
           c. Residual add
           d. LayerNorm + MLP + residual

    Returns:
        all_attn: list of 12 numpy arrays, each [num_heads, num_tokens, num_tokens]
    """
    model.eval()
    input_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]

    all_attn = []

    with torch.no_grad():
        # Step 1: patch embedding -> [1, 197, hidden_dim]
        x = model._process_input(input_tensor)

        # Prepend CLS token
        n = x.shape[0]
        cls_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)   # [1, 197, 768]

        # Add positional embedding
        x = model.encoder.dropout(x + model.encoder.pos_embedding)

        # Step 2: manually forward through each EncoderBlock
        for layer in model.encoder.layers:

            # --- Self-attention sub-block ---
            # LayerNorm before attention
            x_norm = layer.ln_1(x)

            # Call MHA directly with need_weights=True
            # torchvision ViT uses batch_first=True so we pass as-is
            # average_attn_weights=False keeps per-head weights intact
            attn_out, attn_weights = layer.self_attention(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False,
            )
            # attn_weights: [batch, num_heads, num_tokens, num_tokens]
            all_attn.append(attn_weights[0].cpu().numpy())  # [num_heads, 197, 197]

            # Residual connection after attention
            x = x + layer.dropout(attn_out)

            # --- MLP sub-block ---
            x = x + layer.mlp(layer.ln_2(x))

        # Final layernorm
        x = model.encoder.ln(x)

    if len(all_attn) == 0:
        raise RuntimeError("No attention weights extracted — check model architecture.")

    return all_attn  # list of 12 arrays, each [num_heads, 197, 197]


# --------------------------
# Perplexity scoring
# --------------------------
def compute_patch_perplexity(attn_last_layer: np.ndarray) -> np.ndarray:
    """
    Compute per-patch perplexity from the last transformer layer's attention.

    For each patch token, we look at the column of attention it receives from
    all other tokens. The entropy of that distribution measures how 'surprising'
    or uncertain the model is about that patch.

    Perplexity = exp(entropy). Higher = more anomalous patch.

    Steps:
    1. Average attention over all heads -> [num_tokens, num_tokens]
    2. Take columns (attention received by each token)
    3. Renormalize each column to a valid distribution
    4. Compute entropy per token
    5. Exponentiate to get perplexity
    6. Drop CLS token, return patch tokens only [196]

    Args:
        attn_last_layer: [num_heads, num_tokens, num_tokens]

    Returns:
        perplexity: [196] per-patch perplexity scores
    """
    # Average over heads -> [num_tokens, num_tokens]
    mean_attn = attn_last_layer.mean(axis=0)

    # Transpose: attn_cols[i] = distribution of attention received by token i
    attn_cols = mean_attn.T                                       # [num_tokens, num_tokens]
    attn_cols = np.clip(attn_cols, 1e-9, 1.0)
    attn_cols = attn_cols / attn_cols.sum(axis=1, keepdims=True)  # renormalize

    # Entropy per token: H = -sum(p * log(p))
    entropy = -np.sum(attn_cols * np.log(attn_cols), axis=1)      # [num_tokens]

    # Perplexity = exp(entropy)
    perplexity = np.exp(entropy)

    # Drop CLS token (index 0), return patch tokens [196]
    return perplexity[1:]


# --------------------------
# CLS entropy scoring
# --------------------------
def compute_cls_entropy(attn_last_layer: np.ndarray) -> float:
    """
    Compute the entropy of the CLS token's attention distribution.

    The CLS token aggregates information from all patches to make the
    final real/fake decision. Its attention row tells us which patches
    it focuses on.

    Real images tend to have higher CLS entropy (spread attention, complex scenes).
    Fake images tend to have lower CLS entropy (more focused, less coherent scenes).

    Note: this score is negated in evaluate_run.py so that higher = more likely fake,
    consistent with perplexity and rollout directions.

    Args:
        attn_last_layer: [num_heads, num_tokens, num_tokens]

    Returns:
        cls_entropy: scalar float (raw, un-negated)
    """
    # Average over heads -> [num_tokens, num_tokens]
    mean_attn = attn_last_layer.mean(axis=0)

    # CLS token is index 0 — its attention row over all other tokens
    cls_attn = mean_attn[0]                  # [num_tokens]
    cls_attn = np.clip(cls_attn, 1e-9, 1.0)
    cls_attn = cls_attn / cls_attn.sum()     # normalize

    # Shannon entropy
    entropy = -np.sum(cls_attn * np.log(cls_attn))
    return float(entropy)


# --------------------------
# Attention rollout
# --------------------------
def compute_attention_rollout(all_attn_layers: list) -> np.ndarray:
    """
    Compute attention rollout across all 12 transformer layers.

    Rollout (Abnar & Zuidema, 2020) propagates attention through the network
    by multiplying attention matrices layer by layer, with a residual connection
    (identity matrix added at each step). This captures how information flows
    from patches all the way to the CLS token through the full network depth.

    This tells us which patches ultimately influence the final decision,
    accounting for object-level relationships built up across all layers —
    not just the last layer's attention.

    Steps per layer:
    1. Average attention over heads
    2. Add identity matrix (residual connection)
    3. Renormalize rows
    4. Matrix multiply with running rollout

    Finally: extract CLS row -> how much each patch contributes to the
    final classification through the entire network.

    Args:
        all_attn_layers: list of 12 arrays, each [num_heads, num_tokens, num_tokens]

    Returns:
        rollout_patches: [196] — rollout score per patch token
    """
    num_tokens = all_attn_layers[0].shape[-1]

    # Start with identity (perfect information flow before any attention)
    rollout = np.eye(num_tokens)

    for attn in all_attn_layers:
        # Average over heads -> [num_tokens, num_tokens]
        mean_attn = attn.mean(axis=0)

        # Add residual connection: each token also attends to itself
        attn_with_residual = mean_attn + np.eye(num_tokens)

        # Renormalize rows so they sum to 1
        attn_with_residual = attn_with_residual / attn_with_residual.sum(axis=-1, keepdims=True)

        # Propagate: multiply running rollout by this layer's attention
        rollout = attn_with_residual @ rollout

    # CLS token is index 0 — its row tells us patch influence on final decision
    cls_rollout = rollout[0]          # [num_tokens]

    # Drop CLS self-attention (index 0), keep patch tokens [196]
    rollout_patches = cls_rollout[1:]
    return rollout_patches


# --------------------------
# Collect attention scores across test set
# --------------------------
def collect_attention_scores(model, loader):
    """
    Run all three attention-based scores over the entire test loader.

    For each image:
    - Extract attention from all 12 layers
    - Compute perplexity [196], CLS entropy [scalar], rollout [196]

    Results saved as flat arrays:
    - perplexity_scores: [N, 196]  — per-patch perplexity per image
    - cls_entropy_scores: [N]      — global entropy per image
    - rollout_scores: [N, 196]     — rollout patch influence per image

    Args:
        model: trained ViT model
        loader: test DataLoader

    Returns:
        perplexity_scores, cls_entropy_scores, rollout_scores
    """
    all_perplexity = []
    all_cls_entropy = []
    all_rollout = []

    dataset = loader.dataset
    print(f"\nCollecting attention scores for {len(dataset)} test samples...")

    for idx in tqdm(range(len(dataset)), desc="Attention scoring"):
        image_tensor, _ = dataset[idx]

        try:
            all_attn = extract_all_attention_weights(model, image_tensor)
        except RuntimeError as e:
            print(f"Warning: could not extract attention for sample {idx}: {e}")
            # Fill with zeros if extraction fails so arrays stay aligned
            all_perplexity.append(np.zeros(196))
            all_cls_entropy.append(0.0)
            all_rollout.append(np.zeros(196))
            continue

        last_layer_attn = all_attn[-1]

        perplexity = compute_patch_perplexity(last_layer_attn)
        cls_entropy = compute_cls_entropy(last_layer_attn)
        rollout = compute_attention_rollout(all_attn)

        all_perplexity.append(perplexity)
        all_cls_entropy.append(cls_entropy)
        all_rollout.append(rollout)

    return (
        np.array(all_perplexity),      # [N, 196]
        np.array(all_cls_entropy),     # [N]
        np.array(all_rollout),         # [N, 196]
    )


# --------------------------
# Saliency map
# --------------------------
def generate_saliency_map(model, image_tensor, target_class=None):
    """
    Generate a saliency map for a single image tensor.

    Args:
        model: trained PyTorch model
        image_tensor: single image tensor of shape [3, H, W]
        target_class: optional class index to explain.
                      If None, uses the model's predicted class.

    Returns:
        saliency: 2D numpy array of shape [H, W]
        predicted_class: int
        predicted_probs: numpy array of class probabilities
    """
    model.eval()

    # Add batch dimension and move to device
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # We need gradients with respect to the input image
    input_tensor.requires_grad_()

    # Forward pass
    output = model(input_tensor)

    # Predicted class
    predicted_class = output.argmax(dim=1).item()

    # Use predicted class unless user specifies a target
    if target_class is None:
        target_class = predicted_class

    # Zero existing gradients
    model.zero_grad()

    # Backprop only the score for the target class
    score = output[0, target_class]
    score.backward()

    # Gradient of output w.r.t. input image
    gradients = input_tensor.grad.detach().cpu()[0]   # [3, H, W]

    # Standard saliency: max absolute gradient across color channels
    saliency, _ = torch.max(torch.abs(gradients), dim=0)  # [H, W]

    # Normalize saliency map to [0, 1]
    saliency -= saliency.min()
    if saliency.max() > 0:
        saliency /= saliency.max()

    # Probabilities for reference
    probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]

    return saliency.numpy(), predicted_class, probs


# --------------------------
# Visualization
# --------------------------
def show_saliency_map(image_tensor, saliency_map, true_label=None, predicted_class=None, class_names=None):
    """
    Display original image and its saliency map side by side.

    Args:
        image_tensor: tensor of shape [3, H, W]
        saliency_map: numpy array of shape [H, W]
        true_label: optional ground truth label
        predicted_class: optional predicted class
        class_names: optional dict like {0: "real", 1: "fake"}
    """
    if class_names is None:
        class_names = {0: "real", 1: "fake"}

    # Undo normalization for display
    image = image_tensor.detach().cpu().clone()
    image = image.permute(1, 2, 0).numpy()  # [H, W, C]

    # Your transform used mean=0.5, std=0.5
    image = (image * 0.5) + 0.5
    image = np.clip(image, 0, 1)

    title_parts = []
    if true_label is not None:
        title_parts.append(f"True: {class_names[true_label]}")
    if predicted_class is not None:
        title_parts.append(f"Pred: {class_names[predicted_class]}")
    title_text = " | ".join(title_parts)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image" + (f"\n{title_text}" if title_text else ""))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(saliency_map, cmap="hot")
    plt.title("Saliency Map")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def demo_saliency_on_sample(model, loader, sample_index=0):
    """
    Grab one sample from a dataloader and display its saliency map.
    """
    images, labels = next(iter(loader))

    image_tensor = images[sample_index].cpu()
    true_label = labels[sample_index].item()

    saliency_map, predicted_class, probs = generate_saliency_map(model, image_tensor)

    print("True label:", true_label)
    print("Predicted class:", predicted_class)
    print("Class probabilities:", probs)

    show_saliency_map(
        image_tensor=image_tensor,
        saliency_map=saliency_map,
        true_label=true_label,
        predicted_class=predicted_class,
        class_names={0: "real", 1: "fake"}
    )

def save_saliency_maps(model, loader, run_dir, num_samples=5):
    """
    Save saliency map visualizations for a few samples from the dataloader.

    Args:
        model: trained or loaded model
        loader: dataloader to draw samples from
        run_dir: directory where images should be saved
        num_samples: number of saliency figures to save
    """
    saliency_dir = os.path.join(run_dir, "saliency_maps")
    os.makedirs(saliency_dir, exist_ok=True)

    # Get random indices from dataset
    dataset_size = len(loader.dataset)
    selected_indices = random.sample(range(dataset_size), num_samples)

    class_names = {0: "real", 1: "fake"}

    model.eval()

    for idx in selected_indices:
        image_tensor, true_label = loader.dataset[idx]
        image_tensor = image_tensor.cpu()

        saliency_map, predicted_class, probs = generate_saliency_map(model, image_tensor)

        # Undo normalization for display
        image = image_tensor.detach().cpu().clone()
        image = image.permute(1, 2, 0).numpy()
        image = (image * 0.5) + 0.5
        image = np.clip(image, 0, 1)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(
            f"Original Image\nTrue: {class_names[true_label]} | Pred: {class_names[predicted_class]}"
        )
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(saliency_map, cmap="hot")
        plt.title(
            f"Saliency Map\nP(real)={probs[0]:.4f}, P(fake)={probs[1]:.4f}"
        )
        plt.axis("off")

        plt.tight_layout()

        filename = f"sample_{idx}_true_{class_names[true_label]}_pred_{class_names[predicted_class]}.png"
        save_path = os.path.join(saliency_dir, filename)

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Saved saliency map: {save_path}")

    print(f"Saved {num_samples} saliency map(s) to: {saliency_dir}")


# --------------------------
# Main
# --------------------------
def main():
    # Parse command line arguments for experiment settings
    # EX: python Contextual_Deepfake_Detector.py --epochs 10 --batch_size 64 --lr 0.0001
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--num_saliency", type=int, default=5)

    args = parser.parse_args()

    # Make arguments globally accessible
    global EPOCHS, BATCH_SIZE, LEARNING_RATE

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr

    # Show whether we are running on GPU or CPU
    print("Using device:", device)

    # --------------------------
    # Run folder
    # --------------------------
    # Create unique results folder named by config (epochs, batch_size, lr) + time
    time_suffix = time.strftime("%Y%m%d_%H%M%S")
    lr_str = str(LEARNING_RATE).replace(".", "p")
    if args.load_model is not None:
        run_name = f"inference_{time_suffix}"
    else:
        run_name = f"vit_e{EPOCHS}_bs{BATCH_SIZE}_lr{lr_str}_{time_suffix}"
        
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("Saving results to:", run_dir)

    # --------------------------
    # Data
    # --------------------------
    # Load dataset and build dataloaders
    # dataloaders handle batching, shuffling, and parallel loading
    train_loader, val_loader, test_loader = build_dataloaders()

    # Get dataset sizes for logging later
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    # --------------------------
    # Model
    # --------------------------
    # Build the model (Vision Transformer)
    model = build_model()
    training_time = None
    best_val_acc = 0.0
    best_model_path = os.path.join(run_dir, "best_model.pth")

    # --------------------------
    # LOAD MODEL (if provided)
    # --------------------------
    if args.load_model is not None:
        print(f"Loading model from {args.load_model}")

        loaded_obj = torch.load(args.load_model, map_location=device)

        if isinstance(loaded_obj, dict) and "model_state_dict" in loaded_obj:
            model.load_state_dict(loaded_obj["model_state_dict"])
        else:
            model.load_state_dict(loaded_obj)

        model.eval()
    else:
        print("Starting training from pretrained ViT weights")

        # Track how long training takes
        start_time = time.time()

        # Loss function used for classification
        criterion = nn.CrossEntropyLoss()

        # Adam optimizer updates model weights during training
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # --------------------------
        # TRAINING LOOP
        # --------------------------
        for epoch in range(EPOCHS):

            # Train model for one full pass through the dataset
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)

            # Evaluate on validation set to see how model generalizes
            val_metrics = evaluate(model, val_loader)

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]

                torch.save(model.state_dict(), best_model_path)
                print(f"Saved BEST model -> {best_model_path}")

            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            print("Loss:", train_loss)
            print("Validation Accuracy:", val_metrics["accuracy"])
            print("Validation Precision:", val_metrics["precision"])
            print("Validation Recall:", val_metrics["recall"])
            print("Validation F1:", val_metrics["f1_score"])
        
        # Calculate total training time
        training_time = time.time() - start_time
        print("Training time (seconds):", training_time)

        # IMPORTANT: Load BEST model before evaluation
        print("Loading best model for final evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

    # --------------------------
    # SAVE PREDICTIONS
    # --------------------------
    # Collect and save per-example predictions for evaluation
    print("\nCollecting test set predictions for evaluation...")
    y_true, y_pred, y_prob = collect_predictions(model, test_loader)

    pred_dir = os.path.join(run_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    np.save(os.path.join(pred_dir, "test_y_true.npy"), y_true)
    np.save(os.path.join(pred_dir, "test_y_pred.npy"), y_pred)
    np.save(os.path.join(pred_dir, "test_y_prob.npy"), y_prob)

    print("Saved predictions to predictions/ folder")

    # --------------------------
    # SAVE ATTENTION SCORES
    # Perplexity, CLS entropy, and rollout are computed here and saved
    # as .npy files. evaluate_run.py loads these for analysis and heatmaps.
    # --------------------------
    perplexity_scores, cls_entropy_scores, rollout_scores = collect_attention_scores(
        model, test_loader
    )

    np.save(os.path.join(pred_dir, "test_perplexity.npy"), perplexity_scores)
    np.save(os.path.join(pred_dir, "test_cls_entropy.npy"), cls_entropy_scores)
    np.save(os.path.join(pred_dir, "test_rollout.npy"), rollout_scores)

    print("Saved attention scores to predictions/ folder")

    # --------------------------
    # SALIENCY
    # --------------------------
    #demo_saliency_on_sample(model, test_loader, sample_index=0)
    save_saliency_maps(model, test_loader, run_dir, num_samples=args.num_saliency)

    # --------------------------
    # FINAL EVALUATION
    # --------------------------
    subprocess.run([
    "python", "evaluate_run.py",
    "--run_dir", run_dir
])

if __name__ == "__main__":
    main()