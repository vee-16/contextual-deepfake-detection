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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

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
# Main
# --------------------------
def main():
    # Parse command line arguments for experiment settings
    # EX: python Contextual_Deepfake_Detector.py --epochs 10 --batch_size 64 --lr 0.0001
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

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

    # Calculate total training time
    training_time = time.time() - start_time
    print("Training time (seconds):", training_time)

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
            training_time,
            test_metrics["accuracy"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1_score"]
        ])
    
    # Save confusion matrix separately for visualization later
    cm_path = os.path.join(run_dir, "confusion_matrix.npy")
    np.save(cm_path, test_metrics["confusion_matrix"])

    # Save the trained model, can be reloaded later without retraining
    model_save_path = os.path.join(run_dir, "baseline_vit.pth")
    torch.save(model.state_dict(), model_save_path)
    print("Model saved to:", model_save_path)


if __name__ == "__main__":
    main()