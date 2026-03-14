import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import random_split, DataLoader, Dataset
from datasets import load_from_disk
from collections import Counter


# --------------------------
# Config
# --------------------------
TRAIN_PATH = "data/openfake/train"
TEST_PATH = "data/openfake/test"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 5


# --------------------------
# Device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# Transforms
# --------------------------
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
class OpenFakeTorchDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.label_map = {
            "real": 0,
            "fake": 1
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["image"].convert("RGB")
        label_str = sample["label"]

        if label_str not in self.label_map:
            raise ValueError(f"Unexpected label: {label_str}")

        label = self.label_map[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------------
# Data loading
# --------------------------
def build_dataloaders():
    hf_train_dataset = load_from_disk(TRAIN_PATH)
    hf_test_dataset = load_from_disk(TEST_PATH)

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

    sample = hf_train_dataset[0]
    print("\nExample sample:")
    print("Label:", sample["label"])
    print("Prompt:", sample.get("prompt", None))
    print("Model:", sample.get("model", None))
    print("Image type:", type(sample["image"]))

    full_train_dataset = OpenFakeTorchDataset(hf_train_dataset, transform=transform)
    full_test_dataset = OpenFakeTorchDataset(hf_test_dataset, transform=transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    test_dataset = full_test_dataset

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\nAfter split:")
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    return train_loader, val_loader, test_loader


# --------------------------
# Model
# --------------------------
def build_model():
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    # Replace classification head for binary classification
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)

    return model.to(device)


# --------------------------
# Train / Evaluate
# --------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# --------------------------
# Main
# --------------------------
def main():
    print("Using device:", device)

    train_loader, val_loader, test_loader = build_dataloaders()

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc = evaluate(model, val_loader)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("Loss:", train_loss)
        print("Validation Accuracy:", val_acc)

    test_acc = evaluate(model, test_loader)
    print("\nFinal Test Accuracy:", test_acc)


if __name__ == "__main__":
    main()