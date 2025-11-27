import os
import random
import shutil
from math import floor

random.seed(42)  # For reproducibility

RAW_DATA_DIR = "data/raw"
SPLIT_DATA_DIR = "data/split"

CLASSES = ["NSFW", "SFW"]
SPLITS = ["train", "val", "test"]
SPLIT_RATIOS = {"train": 0.75, "val": 0.15, "test": 0.10}

def make_dirs():
    for split in SPLITS:
        for cls in CLASSES:
            dir_path = os.path.join(SPLIT_DATA_DIR, split, cls)
            os.makedirs(dir_path, exist_ok=True)

def split_and_copy():
    for cls in CLASSES:
        src_dir = os.path.join(RAW_DATA_DIR, cls)
        images = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        print(f"Found {len(images)} images for class '{cls}'")

        random.shuffle(images)

        n_total = len(images)
        n_train = floor(SPLIT_RATIOS["train"] * n_total)
        n_val = floor(SPLIT_RATIOS["val"] * n_total)
        n_test = n_total - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, split_files in splits.items():
            for fname in split_files:
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(SPLIT_DATA_DIR, split, cls, fname)
                shutil.copy2(src_path, dst_path)

        print(f"Split for class '{cls}': train={n_train}, val={n_val}, test={n_test}")

def main():
    print("Creating split directories...")
    make_dirs()
    print("Splitting and copying images...")
    split_and_copy()
    print("Done.")

if __name__ == "__main__":
    main()
