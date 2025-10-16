import os
import shutil
import random
from tqdm import tqdm

def split_dataset(
    source_dir: str,
    dest_dir: str,
    split_ratio: float = 0.8,
    seed: int = 42,
):
    """
    Split dataset from source_dir into train/val folders (80/20 by default).
    Assumes structure: source_dir/class_name/*.jpg
    """

    random.seed(seed)
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"Found {len(classes)} classes in '{source_dir}'")

    # Create destination dirs
    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for cls in tqdm(classes, desc="Splitting classes"):
        src_class_dir = os.path.join(source_dir, cls)
        train_class_dir = os.path.join(train_dir, cls)
        val_class_dir = os.path.join(val_dir, cls)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Get all files
        images = [f for f in os.listdir(src_class_dir)
                  if os.path.isfile(os.path.join(src_class_dir, f))]
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)

        # Split
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Copy files
        for img in train_imgs:
            shutil.copy2(os.path.join(src_class_dir, img),
                         os.path.join(train_class_dir, img))
        for img in val_imgs:
            shutil.copy2(os.path.join(src_class_dir, img),
                         os.path.join(val_class_dir, img))

    print(f"\n✅ Done! Split complete:")
    print(f"   Train folder: {train_dir}")
    print(f"   Val folder:   {val_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset into train/val folders (80/20)")
    parser.add_argument("--source", required=True, help="Path to source dataset folder")
    parser.add_argument("--dest", required=True, help="Path to destination output folder")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train split ratio (default 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()

    split_dataset(args.source, args.dest, args.ratio, args.seed)
