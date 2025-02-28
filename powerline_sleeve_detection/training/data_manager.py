import os
import shutil
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import yaml
import numpy as np
from sklearn.model_selection import train_test_split

from powerline_sleeve_detection.system.config import Config

logger = logging.getLogger(__name__)


class DataManager:
    """Manages dataset preparation for training powerline sleeve detection models."""

    def __init__(self, config: Config):
        """Initialize the data manager with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.dataset_dir = Path(config.get('training.dataset_dir', 'dataset'))
        self.labels_dir = None
        self.images_dir = None

    def prepare_dataset(self,
                        raw_data_path: str,
                        splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                        single_class: bool = True,
                        class_names: List[str] = None) -> str:
        """Prepare dataset from raw data for YOLO training.

        Args:
            raw_data_path: Path to directory with raw images and labels
            splits: Train, validation, test splits (must sum to 1.0)
            single_class: Whether to convert all classes to a single 'sleeve' class
            class_names: List of class names (if not single_class)

        Returns:
            Path to the created dataset.yaml file
        """
        if sum(splits) != 1.0:
            raise ValueError(
                f"Dataset splits must sum to 1.0, got {sum(splits)}")

        raw_data_path = Path(raw_data_path)
        if not raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw data path does not exist: {raw_data_path}")

        # Create dataset directory structure
        dataset_dir = self.dataset_dir
        dataset_dir.mkdir(exist_ok=True, parents=True)

        # Identify images and labels
        images = list(raw_data_path.glob("**/*.jpg")) + \
            list(raw_data_path.glob("**/*.png"))
        labels = list(raw_data_path.glob("**/*.txt"))

        logger.info(
            f"Found {len(images)} images and {len(labels)} labels in {raw_data_path}")

        # Create train/val/test directories
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                (dataset_dir / split / subdir).mkdir(exist_ok=True, parents=True)

        # Get image-label pairs
        image_label_pairs = self._match_images_with_labels(images, labels)
        logger.info(f"Matched {len(image_label_pairs)} image-label pairs")

        # Split the dataset
        train_pairs, val_test_pairs = train_test_split(
            image_label_pairs, train_size=splits[0], random_state=42
        )
        val_pairs, test_pairs = train_test_split(
            val_test_pairs,
            train_size=splits[1] / (splits[1] + splits[2]),
            random_state=42
        )

        logger.info(
            f"Split dataset: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")

        # Copy files to respective directories
        self._copy_files(train_pairs, dataset_dir / 'train', single_class)
        self._copy_files(val_pairs, dataset_dir / 'val', single_class)
        self._copy_files(test_pairs, dataset_dir / 'test', single_class)

        # Create dataset.yaml file
        if single_class:
            class_names = ['sleeve']
        elif class_names is None:
            class_names = self._detect_class_names(labels)

        dataset_yaml_path = self._create_dataset_yaml(class_names)

        # Update config with dataset path
        self.config.set('training.dataset_yaml', str(dataset_yaml_path))
        self.config.set('training.class_names', class_names)
        self.config.save()

        return str(dataset_yaml_path)

    def _match_images_with_labels(self,
                                  images: List[Path],
                                  labels: List[Path]) -> List[Tuple[Path, Path]]:
        """Match images with their corresponding labels.

        Args:
            images: List of image paths
            labels: List of label paths

        Returns:
            List of (image_path, label_path) tuples
        """
        pairs = []

        # Create dictionary of labels by stem (filename without extension)
        label_dict = {label.stem: label for label in labels}

        for image in images:
            if image.stem in label_dict:
                pairs.append((image, label_dict[image.stem]))

        return pairs

    def _copy_files(self,
                    pairs: List[Tuple[Path, Path]],
                    target_dir: Path,
                    single_class: bool = True) -> None:
        """Copy image-label pairs to target directory.

        Args:
            pairs: List of (image_path, label_path) tuples
            target_dir: Target directory (e.g., train, val, test)
            single_class: Whether to convert all classes to a single class
        """
        images_dir = target_dir / 'images'
        labels_dir = target_dir / 'labels'

        for image_path, label_path in pairs:
            # Copy image
            shutil.copy(image_path, images_dir / image_path.name)

            # Copy and possibly modify label
            if single_class:
                # Convert all class IDs to 0 (single class)
                self._convert_to_single_class(
                    label_path, labels_dir / label_path.name)
            else:
                shutil.copy(label_path, labels_dir / label_path.name)

    def _convert_to_single_class(self,
                                 source_label: Path,
                                 target_label: Path) -> None:
        """Convert a YOLO label file to use a single class (0).

        Args:
            source_label: Source label file
            target_label: Target label file
        """
        with open(source_label, 'r') as f:
            lines = f.readlines()

        modified_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # class_id x y w h
                # Replace class ID with 0
                parts[0] = '0'
                modified_lines.append(' '.join(parts) + '\n')

        with open(target_label, 'w') as f:
            f.writelines(modified_lines)

    def _detect_class_names(self, label_files: List[Path]) -> List[str]:
        """Detect class names from label files.

        Args:
            label_files: List of label files

        Returns:
            List of class names (default names if detection fails)
        """
        # Try to detect the maximum class ID in the labels
        max_class_id = -1

        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            max_class_id = max(max_class_id, class_id)
            except Exception as e:
                logger.warning(f"Error reading label file {label_file}: {e}")

        if max_class_id >= 0:
            # Generate default class names
            return [f'class_{i}' for i in range(max_class_id + 1)]
        else:
            # Fallback to a single class
            return ['sleeve']

    def _create_dataset_yaml(self, class_names: List[str]) -> Path:
        """Create a YOLO dataset.yaml file.

        Args:
            class_names: List of class names

        Returns:
            Path to the created dataset.yaml file
        """
        dataset_yaml = self.dataset_dir / 'dataset.yaml'

        dataset_config = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }

        with open(dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        logger.info(f"Created dataset config at {dataset_yaml}")
        return dataset_yaml

    def create_empty_dataset(self, class_names: List[str] = None) -> str:
        """Create an empty dataset structure for manual data collection.

        Args:
            class_names: List of class names

        Returns:
            Path to the created dataset directory
        """
        if class_names is None:
            class_names = ['sleeve']

        dataset_dir = self.dataset_dir
        dataset_dir.mkdir(exist_ok=True, parents=True)

        # Create raw images and labels directories
        raw_dir = dataset_dir / 'raw'
        (raw_dir / 'images').mkdir(exist_ok=True, parents=True)
        (raw_dir / 'labels').mkdir(exist_ok=True, parents=True)

        # Create a template for dataset.yaml
        template_yaml = dataset_dir / 'template.yaml'
        template_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }

        with open(template_yaml, 'w') as f:
            yaml.dump(template_config, f, default_flow_style=False)

        # Create a README file with instructions
        readme = dataset_dir / 'README.md'
        with open(readme, 'w') as f:
            f.write("# Powerline Sleeve Detection Dataset\n\n")
            f.write("## Directory Structure\n")
            f.write("- `raw/images`: Place your raw unlabeled images here\n")
            f.write(
                "- `raw/labels`: Place your YOLO format labels here (after labeling)\n\n")
            f.write("## Class Names\n")
            for i, name in enumerate(class_names):
                f.write(f"- {i}: {name}\n")
            f.write("\n## Labeling Format\n")
            f.write("YOLO format: `class_id x_center y_center width height`\n")
            f.write("All values are normalized between 0 and 1\n")

        logger.info(f"Created empty dataset at {dataset_dir}")
        return str(dataset_dir)
