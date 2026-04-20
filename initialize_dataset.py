from datasets import load_dataset, Dataset, load_from_disk, Image as HFImage, concatenate_datasets
from pathlib import Path

import os
import argparse
import time

"""
# OpenFake dataset size:
#   Train: 1.87M rows
#   Test:  59.7K rows
#
# For local development, load a small subset, e.g., using `--sample_size 100`, instead of the full dataset.
"""

DATA_PATH = Path("data/openfake")

def safe_collect(stream, sample_size, save_path, split_name="train"):

    chunk_size = 5000
    total = 0
    chunk = []

    save_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for example in stream:
        chunk.append(example)
        total += 1

        # progress
        if total % 1000 == 0:
            progress = (total / sample_size) * 100
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0

            print(f"{split_name}: {total}/{sample_size} ({progress:.2f}%) | {rate:.1f} samples/s")

        # save chunk
        if len(chunk) >= chunk_size:
            progress = (total / sample_size) * 100
            print(f"{split_name}: Saving chunk at {total} ({progress:.2f}%)")

            Dataset.from_list(chunk).save_to_disk(save_path / f"chunk_{total}")
            chunk = []

        if total >= sample_size:
            break

    # save remaining
    if chunk:
        Dataset.from_list(chunk).save_to_disk(save_path / f"chunk_{total}")

    print(f"{split_name} DONE: {total}")

def create_subset(sample_size: int):
    dataset = load_dataset("ComplexDataLab/OpenFake", streaming=True)

    # prevent crash from bad images
    dataset = dataset.cast_column("image", HFImage(decode=False))

    train_stream = dataset["train"].shuffle(seed=42, buffer_size=10000)
    test_stream = dataset["test"].shuffle(seed=42, buffer_size=10000)

    train_path = DATA_PATH / "train"
    test_path = DATA_PATH / "test"

    safe_collect(train_stream, sample_size, train_path, "train")
    safe_collect(test_stream, sample_size, test_path, "test")

    return load_all_chunks(train_path), load_all_chunks(test_path)

def load_all_chunks(path):
    chunk_paths = sorted(path.glob("chunk_*"))

    if not chunk_paths:
        print(f"No chunks found in {path}")
        return None

    datasets = [load_from_disk(p) for p in chunk_paths]
    return concatenate_datasets(datasets)

def get_dataset(sample_size: int):
    train_path = DATA_PATH / "train"
    test_path = DATA_PATH / "test"

    # if dataset already exists, just load it
    if train_path.exists() and test_path.exists():
        print("Loading existing dataset...")
        train = load_all_chunks(train_path)
        test = load_all_chunks(test_path)

        if train is not None and test is not None:
            print(f"Loaded: {len(train)} train, {len(test)} test")
            return train, test

    # otherwise create fresh
    print("Creating dataset from scratch...")
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