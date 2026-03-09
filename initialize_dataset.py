from datasets import load_dataset, Dataset, load_from_disk
from pathlib import Path
import argparse


"""
# OpenFake dataset size:
#   Train: 1.87M rows
#   Test:  59.7K rows
#
# For local development, load a small subset, e.g., using `--sample_size 100`, instead of the full dataset.
"""

DATA_PATH = Path("data/openfake")

def create_subset(sample_size: int):
    dataset = load_dataset("ComplexDataLab/OpenFake", streaming=True)

    train_samples = list(dataset["train"].take(sample_size))
    test_samples = list(dataset["test"].take(sample_size))

    train_ds = Dataset.from_list(train_samples)
    test_ds = Dataset.from_list(test_samples)

    (DATA_PATH / "train").mkdir(parents=True, exist_ok=True)
    (DATA_PATH / "test").mkdir(parents=True, exist_ok=True)

    train_ds.save_to_disk(DATA_PATH / "train")
    test_ds.save_to_disk(DATA_PATH / "test")

    return train_ds, test_ds


def load_subset():
    train = load_from_disk(DATA_PATH / "train")
    test = load_from_disk(DATA_PATH / "test")
    return train, test


def get_dataset(sample_size: int):
    """Load local subset if available, otherwise create it."""
    if (DATA_PATH / "train").exists() and (DATA_PATH / "test").exists():
        return load_subset()
    return create_subset(sample_size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of samples per split for the local subset"
    )

    args = parser.parse_args()

    train, test = get_dataset(sample_size=args.sample_size)

    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")

    sample = train[0]

    print("\nExample:")
    print("Label:", sample["label"])
    print("Prompt:", sample["prompt"])
    print("Model:", sample["model"])

    # Show image
    # sample["image"].show()