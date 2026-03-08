from datasets import load_dataset, load_from_disk
from pathlib import Path

"""
# OpenFake dataset size:
#   Train: 1.87M rows
#   Test:  59.7K roes
#
# For local development, load a small subset (e.g., 1000 samples) using `sample_size`, instead of the full dataset.
"""

DATA_PATH = Path("data/openfake")

def get_dataset(sample_size=None):
    if DATA_PATH.exists():
        dataset = load_from_disk(DATA_PATH)
    else:
        dataset = load_dataset("ComplexDataLab/OpenFake")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(DATA_PATH)

    train = dataset["train"]
    test = dataset["test"]

    if sample_size:
        train = train.select(range(min(sample_size, len(train))))
        test = test.select(range(min(sample_size, len(test))))

    return train, test


if __name__ == "__main__":
    train, test = get_dataset(sample_size=1000)

    print("Train:", len(train))
    print("Test:", len(test))