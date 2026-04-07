import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import random_split, DataLoader, Dataset
from datasets import load_from_disk
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

import matplotlib.pyplot as plt

# --------------------------
# Config
# --------------------------
# Paths and directories
TRAIN_PATH = "data/openfake/train"
TEST_PATH = "data/openfake/test"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Create folders if they don't already exist
os.makedirs(MODELS_DIR, exist_ok=True)
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

        # Convert image to RGB
        image = sample["image"].convert("RGB")
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=use_pin_memory)

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
    accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)

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

    class_names = {0: "real", 1: "fake"}

    saved_count = 0
    sample_global_index = 0

    model.eval()

    for images, labels in loader:
        batch_size = images.size(0)

        for i in range(batch_size):
            if saved_count >= num_samples:
                print(f"Saved {saved_count} saliency map(s) to: {saliency_dir}")
                return

            image_tensor = images[i].cpu()
            true_label = labels[i].item()

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

            filename = (
                f"sample_{sample_global_index}_true_{class_names[true_label]}"
                f"_pred_{class_names[predicted_class]}.png"
            )
            save_path = os.path.join(saliency_dir, filename)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

            print(f"Saved saliency map: {save_path}")

            saved_count += 1
            sample_global_index += 1

    print(f"Saved {saved_count} saliency map(s) to: {saliency_dir}")

# --------------------------
# Occlusion map
# --------------------------

def generate_occlusion_map_fixed(model, image_tensor, patch_size=32, stride=16, target_class=None):
    """
    Generate an occlusion map for a single image tensor using a fixed patch size.
    Also returns the highest-impact occlusion box.
    """
    model.eval()

    input_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

        if target_class is None:
            target_class = predicted_class

        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        base_score = probs[target_class]

    _, _, height, width = input_tensor.shape
    occlusion_map = np.zeros((height, width), dtype=np.float32)

    best_score_drop = -float("inf")
    best_box = None

    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            bottom = top + patch_size
            right = left + patch_size

            occluded = input_tensor.clone()
            occluded[:, :, top:bottom, left:right] = 0.0

            with torch.no_grad():
                occluded_output = model(occluded)
                occluded_probs = torch.softmax(occluded_output, dim=1)[0].cpu().numpy()
                score_drop = base_score - occluded_probs[target_class]

            occlusion_map[top:bottom, left:right] += score_drop

            if score_drop > best_score_drop:
                best_score_drop = score_drop
                best_box = (left, top, right, bottom)

    occlusion_map -= occlusion_map.min()
    if occlusion_map.max() > 0:
        occlusion_map /= occlusion_map.max()

    return occlusion_map, predicted_class, probs, best_box

def generate_occlusion_map_auto(model, image_tensor, patch_sizes=(16, 24, 32, 48), target_class=None):
    """
    Generate an occlusion map and automatically choose the patch size.
    Also returns the highest-impact occlusion box.
    """
    best_map = None
    best_patch_size = None
    best_score = -1.0
    best_predicted_class = None
    best_probs = None
    best_box = None

    for patch_size in patch_sizes:
        stride = max(8, patch_size // 2)

        occlusion_map, predicted_class, probs, box = generate_occlusion_map_fixed(
            model,
            image_tensor,
            patch_size=patch_size,
            stride=stride,
            target_class=target_class
        )

        mean_value = occlusion_map.mean()
        max_value = occlusion_map.max()

        if mean_value > 0:
            score = max_value / mean_value
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_map = occlusion_map
            best_patch_size = patch_size
            best_predicted_class = predicted_class
            best_probs = probs
            best_box = box

    return best_map, best_predicted_class, best_probs, best_patch_size, best_box

def save_occlusion_box_overlay(image_tensor,
                               best_box,
                               save_path,
                               true_label=None,
                               predicted_class=None,
                               probs=None,
                               box_color="black"):
    """
    Save the original image with the most important occlusion box outlined.
    """
    class_names = {0: "real", 1: "fake"}

    image = image_tensor.detach().cpu().clone()
    image = image.permute(1, 2, 0).numpy()
    image = (image * 0.5) + 0.5
    image = np.clip(image, 0, 1)

    plt.figure(figsize=(5, 5))
    plt.imshow(image)

    if best_box is not None:
        left, top, right, bottom = best_box
        width = right - left
        height = bottom - top

        rect = plt.Rectangle(
            (left, top),
            width,
            height,
            fill=False,
            edgecolor=box_color,
            linewidth=2
            )
        plt.gca().add_patch(rect)

    title_parts = []
    if true_label is not None:
        title_parts.append(f"True: {class_names[true_label]}")
    if predicted_class is not None:
        title_parts.append(f"Pred: {class_names[predicted_class]}")
    if probs is not None:
        title_parts.append(f"P(real)={probs[0]:.4f}, P(fake)={probs[1]:.4f}")

    plt.title("\n".join(title_parts))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def save_occlusion_maps(
    model,
    loader,
    run_dir,
    num_samples=5,
    use_auto_patch_size=False,
    patch_size=32,
    stride=16,
    box_color="black"
):
    """
    Save occlusion map visualizations and best-box overlays for a few samples.
    """
    occlusion_dir = os.path.join(run_dir, "occlusion_maps")
    occlusion_box_dir = os.path.join(run_dir, "occlusion_boxes")

    os.makedirs(occlusion_dir, exist_ok=True)
    os.makedirs(occlusion_box_dir, exist_ok=True)

    class_names = {0: "real", 1: "fake"}

    saved_count = 0
    sample_index = 0

    for images, labels in loader:
        for i in range(images.size(0)):
            if saved_count >= num_samples:
                print(f"Saved {saved_count} occlusion map(s) to: {occlusion_dir}")
                print(f"Saved {saved_count} occlusion box image(s) to: {occlusion_box_dir}")
                return

            image_tensor = images[i].cpu()
            true_label = labels[i].item()

            if use_auto_patch_size:
                occlusion_map, predicted_class, probs, used_patch_size, best_box = generate_occlusion_map_auto(
                    model,
                    image_tensor
                )
            else:
                occlusion_map, predicted_class, probs, best_box = generate_occlusion_map_fixed(
                    model,
                    image_tensor,
                    patch_size=patch_size,
                    stride=stride
                )
                used_patch_size = patch_size

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
            plt.imshow(occlusion_map, cmap="hot")
            plt.title(
                f"Occlusion Map\nPatch={used_patch_size} | P(real)={probs[0]:.4f}, P(fake)={probs[1]:.4f}"
            )
            plt.axis("off")

            plt.tight_layout()

            base_filename = (
                f"sample_{sample_index}_true_{class_names[true_label]}"
                f"_pred_{class_names[predicted_class]}"
            )

            map_save_path = os.path.join(occlusion_dir, f"{base_filename}.png")
            plt.savefig(map_save_path, bbox_inches="tight")
            plt.close()

            print(f"Saved occlusion map: {map_save_path}")

            box_save_path = os.path.join(occlusion_box_dir, f"{base_filename}_box.png")
            save_occlusion_box_overlay(
                image_tensor=image_tensor,
                best_box=best_box,
                save_path=box_save_path,
                true_label=true_label,
                predicted_class=predicted_class,
                probs=probs,
                box_color=box_color
            )

            print(f"Saved occlusion box image: {box_save_path}")

            saved_count += 1
            sample_index += 1

    print(f"Saved {saved_count} occlusion map(s) to: {occlusion_dir}")
    print(f"Saved {saved_count} occlusion box image(s) to: {occlusion_box_dir}")

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
    parser.add_argument("--num_occlusion", type=int, default=5)
    parser.add_argument("--occlusion_patch", type=int, default=32)
    parser.add_argument("--occlusion_stride", type=int, default=16)
    parser.add_argument("--use_auto_patch_size", action="store_true")
    parser.add_argument("--occlusion_box_color", type=str, default="black")

    args = parser.parse_args()

    # Make arguments globally accessible
    global EPOCHS, BATCH_SIZE, LEARNING_RATE

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr

    # Show whether we are running on GPU or CPU
    print("Using device:", device)

    # Create unique results folder named by config (epochs, batch_size, lr) + time
    time_suffix = time.strftime("%H%M%S")
    lr_str = str(LEARNING_RATE).replace(".", "p")
    if args.load_model is not None:
        run_name = f"inference_{time_suffix}"
    else:
        run_name = f"epochs{EPOCHS}_bs{BATCH_SIZE}_lr{lr_str}_{time_suffix}"
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("Saving results to:", run_dir)

    # Load dataset and build dataloaders
    # dataloaders handle batching, shuffling, and parallel loading
    train_loader, val_loader, test_loader = build_dataloaders()

    # Get dataset sizes for logging later
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    # Build the model (Vision Transformer)
    model = build_model()
    training_time = None

    if args.load_model is not None:

        loaded_obj = torch.load(args.load_model, map_location=device, weights_only=False)
        if "model_state_dict" in loaded_obj:
            model.load_state_dict(loaded_obj["model_state_dict"])
        else:
            model.load_state_dict(loaded_obj)
        print(f"Loaded model from {args.load_model}")
    else:

        print("Starting training from pretrained ViT weights")

        # Track how long training takes
        start_time = time.time()

        # Loss function used for classification
        criterion = nn.CrossEntropyLoss()

        # Adam optimizer updates model weights during training
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        for epoch in range(EPOCHS):

            # Train model for one full pass through the dataset
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)

            # Evaluate on validation set to see how model generalizes
            val_metrics = evaluate(model, val_loader)

            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            print("Loss:", train_loss)
            print("Validation Accuracy:", val_metrics["accuracy"])
            print("Validation Precision:", val_metrics["precision"])
            print("Validation Recall:", val_metrics["recall"])
            print("Validation F1:", val_metrics["f1_score"])

            # Save checkpoint after each epoch
            # This allows us to resume training if something crashes
            checkpoint_path = os.path.join(MODELS_DIR, f"checkpoint_epoch_{epoch+1}.pth")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Calculate total training time
        training_time = time.time() - start_time
        print("Training time (seconds):", training_time)

    # Post-training, do a final evaluation on test set
    test_metrics = evaluate(model, test_loader)

    print("\nFinal Test Metrics")
    print("Accuracy:", test_metrics["accuracy"])
    print("Precision:", test_metrics["precision"])
    print("Recall:", test_metrics["recall"])
    print("F1 Score:", test_metrics["f1_score"])
    print("Confusion Matrix:\n", test_metrics["confusion_matrix"])

    # Collect and save per-example predictions for evaluation
    print("\nCollecting test set predictions for evaluation...")
    y_true, y_pred, y_prob = collect_predictions(model, test_loader)

    np.save(os.path.join(run_dir, "test_y_true.npy"), y_true)
    np.save(os.path.join(run_dir, "test_y_pred.npy"), y_pred)
    np.save(os.path.join(run_dir, "test_y_prob.npy"), y_prob)
    print("Saved test_y_true.npy, test_y_pred.npy, and test_y_prob.npy")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save results to CSV so we can compare experiments later
    results_file = os.path.join(run_dir, "baseline_results.csv")

    file_exists = os.path.isfile(results_file)

    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header if file doesn't exist yet
        if not file_exists:
            writer.writerow([
                "timestamp",
                "device",
                "train_size",
                "val_size",
                "test_size",
                "epochs",
                "batch_size",
                "learning_rate",
                "training_time_seconds",
                "accuracy",
                "precision",
                "recall",
                "f1_score"
            ])

        # Write experiment results
        writer.writerow([
            timestamp,
            device.type,
            train_size,
            val_size,
            test_size,
            EPOCHS,
            BATCH_SIZE,
            LEARNING_RATE,
            training_time if training_time is not None else "N/A",
            test_metrics["accuracy"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1_score"]
        ])
    
    # Save confusion matrix separately for visualization later
    cm_path = os.path.join(run_dir, "confusion_matrix.npy")
    np.save(cm_path, test_metrics["confusion_matrix"])

    if args.load_model is None:
        # Save the trained model, can be reloaded later without retraining
        model_save_path = os.path.join(run_dir, "baseline_vit.pth")
        torch.save(model.state_dict(), model_save_path)
        print("Model saved to:", model_save_path)

    #demo_saliency_on_sample(model, test_loader, sample_index=0)
    save_saliency_maps(model, test_loader, run_dir, num_samples=args.num_saliency)

    save_occlusion_maps(
        model,
        test_loader,
        run_dir,
        num_samples=args.num_occlusion,
        use_auto_patch_size=args.use_auto_patch_size,
        patch_size=args.occlusion_patch,
        stride=args.occlusion_stride,
        box_color=args.occlusion_box_color
    )

if __name__ == "__main__":
    main()