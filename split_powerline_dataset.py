import os
import shutil
import random
from pathlib import Path
import yaml
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def split_dataset(
    augmented_dir: str = "data/two_stage/powerline/augmented",
    output_dir: str = "data/two_stage/powerline/datasets",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    verify_data: bool = True,
    visualize_splits: bool = True
):
    """
    Split the augmented dataset into train, validation, and test sets.

    Args:
        augmented_dir: Directory containing augmented images and labels
        output_dir: Directory to save the split dataset
        train_ratio: Ratio of images for training
        val_ratio: Ratio of images for validation
        test_ratio: Ratio of images for testing
        seed: Random seed for reproducibility
        verify_data: Whether to verify data integrity before splitting
        visualize_splits: Whether to generate visualizations of the splits
    """
    # Set random seeds for reproducibility in multiple libraries
    random.seed(seed)
    np.random.seed(seed)

    print(f"Preparing dataset split with seed {seed}")
    print(
        f"Ratios - Train: {train_ratio:.1f}, Val: {val_ratio:.1f}, Test: {test_ratio:.1f}")

    # Create paths
    aug_path = Path(augmented_dir)
    out_path = Path(output_dir)

    # Verify ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Split ratios must sum to 1.0")

    # Create output directories
    for split in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            os.makedirs(out_path / split / subdir, exist_ok=True)

    # Get all image files (both original and augmented)
    image_files = list(Path(aug_path / "images").glob("*.jpg"))
    print(f"Found {len(image_files)} images in the augmented directory")

    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {aug_path / 'images'}")

    # Data verification
    if verify_data:
        verify_dataset_integrity(aug_path, image_files)

    # Group images by their base name (without _augN suffix)
    # This ensures original and augmented versions of the same image stay together
    image_groups = {}
    for img_path in image_files:
        base_name = img_path.stem
        # Remove augmentation suffix if present
        if "_aug" in base_name:
            base_name = base_name.rsplit("_aug", 1)[0]

        if base_name not in image_groups:
            image_groups[base_name] = []
        image_groups[base_name].append(img_path)

    print(f"Found {len(image_groups)} unique base images (before augmentation)")

    # Shuffle the keys (Fisher-Yates algorithm via random.shuffle)
    base_names = list(image_groups.keys())
    random.shuffle(base_names)

    # Calculate split indices
    total = len(base_names)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split the dataset
    splits = {
        "train": base_names[:train_end],
        "val": base_names[train_end:val_end],
        "test": base_names[val_end:]
    }

    # Stats tracking
    stats = {
        "train": {"images": 0, "labels": 0, "image_sizes": []},
        "val": {"images": 0, "labels": 0, "image_sizes": []},
        "test": {"images": 0, "labels": 0, "image_sizes": []}
    }

    bbox_counts = {
        "train": [],
        "val": [],
        "test": []
    }

    # Copy files to respective directories
    for split, base_names_list in splits.items():
        print(
            f"\nProcessing {split} split ({len(base_names_list)} base images)...")

        for base_name in base_names_list:
            for img_path in image_groups[base_name]:
                # Get corresponding label file
                label_path = aug_path / "labels" / f"{img_path.stem}.txt"

                # Copy image and label
                if label_path.exists():
                    shutil.copy(img_path, out_path / split / "images")
                    shutil.copy(label_path, out_path / split / "labels")

                    stats[split]["images"] += 1

                    # Count bounding boxes in this image
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        num_bboxes = len(lines)
                        stats[split]["labels"] += num_bboxes
                        bbox_counts[split].append(num_bboxes)
                else:
                    print(f"Warning: Label file not found for {img_path.name}")

    # Generate and print dataset statistics
    print("\n=== Dataset Split Statistics ===")
    for split in ["train", "val", "test"]:
        image_count = stats[split]["images"]
        bbox_count = stats[split]["labels"]

        print(f"\n{split.upper()} Set:")
        print(f"  • Images: {image_count}")
        print(f"  • Bounding boxes: {bbox_count}")
        if image_count > 0:
            print(
                f"  • Average bboxes per image: {bbox_count / image_count:.2f}")
            if bbox_counts[split]:
                print(f"  • Min bboxes per image: {min(bbox_counts[split])}")
                print(f"  • Max bboxes per image: {max(bbox_counts[split])}")

    # Create dataset.yaml
    dataset_config = {
        "path": str(out_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["powerline"]
    }

    with open(out_path / "dataset.yaml", "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    # Copy classes.txt
    classes_src = aug_path / "classes.txt"
    classes_dst = out_path / "classes.txt"
    if classes_src.exists() and not classes_dst.exists():
        shutil.copy(classes_src, classes_dst)

    print(f"\nCreated dataset configuration at {out_path / 'dataset.yaml'}")

    if visualize_splits and bbox_counts["train"] and bbox_counts["val"] and bbox_counts["test"]:
        visualize_dataset_splits(stats, bbox_counts, out_path)

    return str(out_path)


def verify_dataset_integrity(dataset_path: Path, image_files: list) -> None:
    """Verify dataset integrity by checking images and labels"""
    print("\nVerifying dataset integrity...")

    # Check if images are valid
    corrupt_images = 0
    missing_labels = 0
    empty_labels = 0

    for img_path in image_files:
        # Check for corresponding label
        label_path = dataset_path / "labels" / f"{img_path.stem}.txt"

        if not label_path.exists():
            missing_labels += 1
            print(f"  Missing label for {img_path.name}")
            continue

        # Check if label file is empty
        if label_path.stat().st_size == 0:
            empty_labels += 1
            print(f"  Empty label file for {img_path.name}")

    if corrupt_images == 0 and missing_labels == 0 and empty_labels == 0:
        print("✓ All checks passed!")
    else:
        print(f"\nFound issues:")
        if corrupt_images > 0:
            print(f"  - {corrupt_images} corrupt images")
        if missing_labels > 0:
            print(f"  - {missing_labels} missing label files")
        if empty_labels > 0:
            print(f"  - {empty_labels} empty label files")


def visualize_dataset_splits(stats: dict, bbox_counts: dict, output_dir: Path) -> None:
    """Generate visualizations of the dataset splits"""
    plt.figure(figsize=(15, 10))

    # Plot 1: Image distribution
    plt.subplot(2, 2, 1)
    splits = ["train", "val", "test"]
    image_counts = [stats[split]["images"] for split in splits]
    plt.bar(splits, image_counts, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.title('Image Distribution Across Splits')
    plt.ylabel('Number of Images')
    for i, count in enumerate(image_counts):
        plt.text(i, count + 5, str(count), ha='center')

    # Plot 2: Bbox distribution
    plt.subplot(2, 2, 2)
    bbox_counts_list = [stats[split]["labels"] for split in splits]
    plt.bar(splits, bbox_counts_list, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.title('Bounding Box Distribution Across Splits')
    plt.ylabel('Number of Bounding Boxes')
    for i, count in enumerate(bbox_counts_list):
        plt.text(i, count + 5, str(count), ha='center')

    # Plot 3: Bbox per image histogram
    plt.subplot(2, 2, 3)
    for split, counts in bbox_counts.items():
        if counts:  # Check if counts is not empty
            plt.hist(counts, alpha=0.5, label=split)
    plt.title('Bounding Boxes per Image')
    plt.xlabel('Number of Bounding Boxes')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot 4: Average bboxes per image
    plt.subplot(2, 2, 4)
    avg_bboxes = []
    for split in splits:
        if stats[split]["images"] > 0:
            avg_bboxes.append(stats[split]["labels"] / stats[split]["images"])
        else:
            avg_bboxes.append(0)
    plt.bar(splits, avg_bboxes, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.title('Average Bounding Boxes per Image')
    plt.ylabel('Average Count')
    for i, avg in enumerate(avg_bboxes):
        plt.text(i, avg + 0.1, f"{avg:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_statistics.png")
    print(
        f"Dataset visualization saved to {output_dir / 'dataset_statistics.png'}")


if __name__ == "__main__":
    split_dataset()
