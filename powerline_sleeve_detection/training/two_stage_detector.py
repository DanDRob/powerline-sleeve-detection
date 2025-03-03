import os
import logging
import numpy as np
import cv2
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import yaml
import albumentations as A
from tqdm import tqdm
import torch

logger = logging.getLogger(__name__)


class TwoStageDetector:
    """
    Implements a two-stage approach for powerline sleeve detection:
    1. First detect powerlines
    2. Then detect sleeves on the powerlines
    """

    def __init__(self, config_path: str):
        """
        Initialize the two-stage detector with configuration.

        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.powerline_model_path = None
        self.sleeve_model_path = None

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def prepare_directory_structure(self, base_dir: str = "data/two_stage") -> Dict[str, str]:
        """
        Create directory structure for the two-stage detection workflow.

        Args:
            base_dir: Base directory for the two-stage detection data

        Returns:
            Dictionary with paths to all created directories
        """
        base_path = Path(base_dir)

        # Create main directories
        dirs = {
            "base": str(base_path),
            "powerline": {
                "raw": str(base_path / "powerline" / "raw"),
                "labeled": str(base_path / "powerline" / "labeled"),
                "augmented": str(base_path / "powerline" / "augmented"),
                "datasets": str(base_path / "powerline" / "datasets"),
                "models": str(base_path / "powerline" / "models"),
                "results": str(base_path / "powerline" / "results"),
            },
            "sleeve": {
                "raw": str(base_path / "sleeve" / "raw"),
                "labeled": str(base_path / "sleeve" / "labeled"),
                "augmented": str(base_path / "sleeve" / "augmented"),
                "datasets": str(base_path / "sleeve" / "datasets"),
                "models": str(base_path / "sleeve" / "models"),
                "results": str(base_path / "sleeve" / "results"),
            }
        }

        # Create all directories
        for category in ["powerline", "sleeve"]:
            for dir_name, dir_path in dirs[category].items():
                os.makedirs(dir_path, exist_ok=True)

        logger.info(f"Created directory structure in {base_path}")
        return dirs

    def setup_labeling_environment(self, images_dir: str, output_dir: str, target: str = "powerline"):
        """
        Set up environment for manual labeling of images.

        Args:
            images_dir: Directory containing images to label
            output_dir: Directory to save labeled data
            target: What to label - 'powerline' or 'sleeve'
        """
        os.makedirs(output_dir, exist_ok=True)
        images_path = Path(images_dir)
        output_path = Path(output_dir)

        # Create images and labels directories in the output directory
        images_output = output_path / "images"
        labels_output = output_path / "labels"
        os.makedirs(images_output, exist_ok=True)
        os.makedirs(labels_output, exist_ok=True)

        # Copy images to the labeling directory
        image_files = list(images_path.glob("*.jpg")) + \
            list(images_path.glob("*.png"))

        if not image_files:
            logger.error(f"No images found in {images_dir}")
            return

        for img_file in tqdm(image_files, desc="Copying images for labeling"):
            shutil.copy(img_file, images_output / img_file.name)

            # Create empty label file to be filled during labeling
            label_file = labels_output / f"{img_file.stem}.txt"
            if not label_file.exists():
                with open(label_file, 'w') as f:
                    pass  # Create empty file

        logger.info(
            f"Prepared {len(image_files)} images for {target} labeling")
        logger.info(f"Images directory: {images_output}")
        logger.info(f"Labels directory: {labels_output}")

        # Generate classes.txt file with appropriate class names
        with open(output_path / "classes.txt", 'w') as f:
            if target == "powerline":
                f.write("powerline\n")
            else:
                f.write("sleeve\n")

        # Create dataset.yaml file for YOLOv5
        dataset_yaml = {
            "path": str(output_path),
            "train": "images",
            "val": "images",
            "test": "images",
            "nc": 1,
            "names": ["powerline"] if target == "powerline" else ["sleeve"]
        }

        with open(output_path / "dataset.yaml", 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)

        logger.info(f"Created dataset configuration for {target} labeling")

        # Print instructions for manual labeling using LabelImg
        print("\n" + "="*80)
        print(f"INSTRUCTIONS FOR MANUAL {target.upper()} LABELING")
        print("="*80)
        print("1. Install LabelImg: pip install labelImg")
        print(
            f"2. Run: labelImg {images_output} {labels_output} {output_path}/classes.txt")
        print("3. Use the following keyboard shortcuts for faster labeling:")
        print("   - W: Create a rectangle box")
        print("   - D: Next image")
        print("   - A: Previous image")
        print("   - Ctrl+S: Save")
        print("   - Ctrl+R: Change default save directory")
        print(f"4. Label all {target}s in the images")
        print("="*80 + "\n")

    def augment_dataset(self,
                        dataset_dir: str,
                        output_dir: str,
                        num_augmentations: int = 5,
                        severity: str = 'medium'):
        """
        Augment a labeled dataset to create additional training examples.

        Args:
            dataset_dir: Directory containing labeled dataset (with images/ and labels/ subdirs)
            output_dir: Directory to save augmented dataset
            num_augmentations: Number of augmented copies to create per original image
            severity: Augmentation severity ('light', 'medium', 'heavy')

        Returns:
            Tuple of (original_count, augmented_count)
        """
        # Set up paths
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)

        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"

        augmented_images_dir = output_path / "images"
        augmented_labels_dir = output_path / "labels"

        os.makedirs(augmented_images_dir, exist_ok=True)
        os.makedirs(augmented_labels_dir, exist_ok=True)

        # Copy classes.txt and dataset.yaml if they exist
        if (dataset_path / "classes.txt").exists():
            shutil.copy(dataset_path / "classes.txt",
                        output_path / "classes.txt")

        if (dataset_path / "dataset.yaml").exists():
            with open(dataset_path / "dataset.yaml", 'r') as f:
                dataset_config = yaml.safe_load(f)

            dataset_config["path"] = str(output_path)

            with open(output_path / "dataset.yaml", 'w') as f:
                yaml.dump(dataset_config, f, default_flow_style=False)

        # Create augmentation transforms based on severity
        transform = self._create_augmentation_transforms(severity)

        # Get all image files
        image_files = list(images_dir.glob("*.jpg")) + \
            list(images_dir.glob("*.png"))
        original_count = len(image_files)
        augmented_count = 0

        # First, copy original files
        for img_file in tqdm(image_files, desc="Copying original files"):
            # Copy image
            shutil.copy(img_file, augmented_images_dir / img_file.name)

            # Copy corresponding label if it exists
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, augmented_labels_dir / label_file.name)
            else:
                # Create empty label file
                with open(augmented_labels_dir / f"{img_file.stem}.txt", 'w') as f:
                    pass

        # Now create augmented versions
        for img_idx, img_file in enumerate(tqdm(image_files, desc="Augmenting images")):
            # Read image
            image = cv2.imread(str(img_file))
            if image is None:
                logger.warning(f"Could not read image: {img_file}")
                continue

            # Read corresponding bboxes from label file
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                bboxes, class_labels = self._read_yolo_labels(label_file)
            else:
                bboxes, class_labels = [], []

            # Generate augmented versions
            for aug_idx in range(num_augmentations):
                # Apply augmentation with bounding box transforms
                if bboxes:
                    try:
                        # Validate bboxes before transformation
                        validated_bboxes = self._validate_bboxes(bboxes)
                        transformed = transform(
                            image=image, bboxes=validated_bboxes, class_labels=class_labels)
                        aug_image = transformed["image"]
                        aug_bboxes = transformed["bboxes"]
                        aug_class_labels = transformed["class_labels"]
                    except Exception as e:
                        logger.warning(
                            f"Augmentation failed for {img_file.name}, aug #{aug_idx+1}: {str(e)}")
                        continue
                else:
                    # Always pass class_labels even if empty to match the BboxParams configuration
                    transformed = transform(
                        image=image, bboxes=[], class_labels=[])
                    aug_image = transformed["image"]
                    aug_bboxes = []
                    aug_class_labels = []

                # Save augmented image
                aug_filename = f"{img_file.stem}_aug{aug_idx+1}{img_file.suffix}"
                cv2.imwrite(
                    str(augmented_images_dir / aug_filename), aug_image)

                # Save augmented labels
                aug_label_filename = f"{img_file.stem}_aug{aug_idx+1}.txt"
                self._write_yolo_labels(
                    augmented_labels_dir / aug_label_filename,
                    aug_bboxes,
                    aug_class_labels
                )

                augmented_count += 1

        logger.info(
            f"Augmentation complete: {original_count} original images + {augmented_count} augmented images")
        return original_count, augmented_count

    def _create_augmentation_transforms(self, severity: str) -> A.Compose:
        """
        Create augmentation transforms based on severity level.

        Args:
            severity: Augmentation severity ('light', 'medium', 'heavy')

        Returns:
            Albumentations Compose object with transforms
        """
        # Base transforms (always applied)
        transforms = [
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.2),

            # Color/intensity transforms
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        ]

        # Add more transforms based on severity
        if severity == 'light':
            transforms.extend([
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.RandomShadow(p=0.2),
                A.CLAHE(clip_limit=2, p=0.3),
            ])
        elif severity == 'medium':
            transforms.extend([
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
                A.CLAHE(clip_limit=4, p=0.4),
                A.ImageCompression(quality_lower=85, quality_upper=100, p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
            ])
        elif severity == 'heavy':
            transforms.extend([
                A.GaussianBlur(blur_limit=(3, 9), p=0.4),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.4),
                A.CLAHE(clip_limit=6, p=0.5),
                A.ImageCompression(quality_lower=70, quality_upper=90, p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.4),
                A.RandomSnow(snow_point_lower=0.1,
                             snow_point_upper=0.2, brightness_coeff=2.5, p=0.1),
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20,
                             drop_width=1, drop_color=(200, 200, 200), p=0.1),
            ])

        # Create the composition with bounding box compatibility
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3,
            )
        )

    def _read_yolo_labels(self, label_path: Path) -> Tuple[List, List]:
        """
        Read bounding boxes and class labels from a YOLO format label file.

        Args:
            label_path: Path to the label file

        Returns:
            Tuple of (bboxes, class_labels)
        """
        bboxes = []
        class_labels = []

        if not label_path.exists():
            return bboxes, class_labels

        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)

        return bboxes, class_labels

    def _validate_bboxes(self, bboxes: List) -> List:
        """
        Ensures all bounding box coordinates are within the valid range [0,1].

        Args:
            bboxes: List of bounding boxes in YOLO format

        Returns:
            List of validated bounding boxes
        """
        validated_bboxes = []
        for bbox in bboxes:
            # For YOLO format: [x_center, y_center, width, height]
            x_center = max(0.001, min(0.999, bbox[0]))
            y_center = max(0.001, min(0.999, bbox[1]))
            width = max(0.001, min(0.999, bbox[2]))
            height = max(0.001, min(0.999, bbox[3]))

            # Ensure x_center +/- width/2 stays in [0,1]
            if x_center - width/2 < 0:
                width = 2 * x_center
            if x_center + width/2 > 1:
                width = 2 * (1 - x_center)

            # Ensure y_center +/- height/2 stays in [0,1]
            if y_center - height/2 < 0:
                height = 2 * y_center
            if y_center + height/2 > 1:
                height = 2 * (1 - y_center)

            validated_bboxes.append([x_center, y_center, width, height])
        return validated_bboxes

    def _write_yolo_labels(self, label_path: Path, bboxes: List, class_labels: List) -> None:
        """
        Write bounding boxes and class labels to a YOLO format label file.

        Args:
            label_path: Path to the output label file
            bboxes: List of bounding boxes in YOLO format [x_center, y_center, width, height]
            class_labels: List of class IDs for each bounding box
        """
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                if len(bbox) >= 4:
                    f.write(
                        f"{int(class_id)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    def train_model(self,
                    dataset_dir: str,
                    output_dir: str,
                    target: str = "powerline",
                    epochs: int = 100,
                    batch_size: int = 16,
                    model_size: str = "l",
                    hyperparameter_tuning: bool = False):
        """
        Train a YOLOv5 model for the given target (powerline or sleeve).
        Optionally perform hyperparameter tuning to find the optimal model.

        Args:
            dataset_dir: Directory containing dataset with images/, labels/ and dataset.yaml
            output_dir: Directory to save trained model
            target: What to detect - 'powerline' or 'sleeve'
            epochs: Number of training epochs (used if hyperparameter_tuning is False)
            batch_size: Batch size for training (used if hyperparameter_tuning is False)
            model_size: YOLOv5 model size ('n', 's', 'm', 'l', 'x')
            hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Path to the trained model weights
        """
        # Set up paths
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)

        # Check if dataset.yaml exists
        dataset_yaml = dataset_path / "dataset.yaml"
        if not dataset_yaml.exists():
            logger.error(f"dataset.yaml not found in {dataset_dir}")
            return None

        # Clone YOLOv5 if needed
        yolov5_dir = Path("yolov5")
        if not yolov5_dir.exists():
            logger.info("Cloning YOLOv5 repository...")
            os.system("git clone https://github.com/ultralytics/yolov5.git")
            os.system("pip install -r yolov5/requirements.txt")

        # If not performing hyperparameter tuning, train a single model with the provided parameters
        if not hyperparameter_tuning:
            model_name = f"yolov5{model_size}"
            project_dir = output_path.parent if output_path.name == "models" else output_path
            run_name = f"{target}_detector_{model_size}"

            # Construct the training command
            cmd = (
                f"python yolov5/train.py "
                f"--img 640 "
                f"--batch {batch_size} "
                f"--epochs {epochs} "
                f"--data {dataset_yaml} "
                f"--weights {model_name}.pt "
                f"--project {project_dir} "
                f"--name {run_name} "
                f"--cache"
            )

            logger.info(f"Starting {target} model training with {model_name}")
            logger.info(f"Training command: {cmd}")

            # Run the training
            os.system(cmd)

            # Find the trained model path
            model_path = project_dir / run_name / "weights" / "best.pt"

            if model_path.exists():
                logger.info(f"Training complete. Model saved to {model_path}")

                # Update model path
                if target == "powerline":
                    self.powerline_model_path = str(model_path)
                else:
                    self.sleeve_model_path = str(model_path)

                return str(model_path)
            else:
                logger.error(
                    f"Model training failed. Model not found at {model_path}")
                return None

        # Hyperparameter tuning
        else:
            logger.info(
                f"Starting hyperparameter tuning for {target} detection")

            # Define hyperparameter configurations to try
            hyperparams = [
                # Configuration 1: Default parameters
                {
                    "img_size": 640,
                    "batch_size": 16,
                    "epochs": epochs,
                    "lr0": 0.01,
                    "optimizer": "SGD",
                    "weight_decay": 0.0005,
                    "momentum": 0.937,
                    "warmup_epochs": 3.0,
                    "description": "Default parameters"
                },
                # Configuration 2: Higher learning rate
                {
                    "img_size": 640,
                    "batch_size": 16,
                    "epochs": epochs,
                    "lr0": 0.02,
                    "optimizer": "SGD",
                    "weight_decay": 0.0005,
                    "momentum": 0.937,
                    "warmup_epochs": 3.0,
                    "description": "Higher learning rate"
                },
                # Configuration 3: AdamW optimizer
                {
                    "img_size": 640,
                    "batch_size": 16,
                    "epochs": epochs,
                    "lr0": 0.001,
                    "optimizer": "AdamW",
                    "weight_decay": 0.0005,
                    "warmup_epochs": 3.0,
                    "description": "AdamW optimizer"
                },
                # Configuration 4: Larger image size
                {
                    "img_size": 640,
                    "batch_size": 32,  # Smaller batch size for larger images
                    "epochs": epochs,
                    "lr0": 0.01,
                    "optimizer": "SGD",
                    "weight_decay": 0.0005,
                    "momentum": 0.937,
                    "warmup_epochs": 3.0,
                    "description": "Larger image size"
                },
                # Configuration 5: More aggressive weight decay
                {
                    "img_size": 640,
                    "batch_size": 16,
                    "epochs": epochs,
                    "lr0": 0.01,
                    "optimizer": "SGD",
                    "weight_decay": 0.001,  # More aggressive
                    "momentum": 0.937,
                    "warmup_epochs": 3.0,
                    "description": "More aggressive weight decay"
                }
            ]

            # Train models with each hyperparameter configuration
            best_map = 0.0
            best_model_path = None
            best_config = None

            for i, config in enumerate(hyperparams):
                run_name = f"{target}_detector_{model_size}_config{i+1}"
                model_name = f"yolov5{model_size}"
                project_dir = output_path.parent if output_path.name == "models" else output_path

                logger.info(
                    f"Training configuration {i+1}/{len(hyperparams)}: {config['description']}")

                # Construct the training command with specific hyperparameters
                cmd = (
                    f"python yolov5/train.py "
                    f"--img {config['img_size']} "
                    f"--batch {config['batch_size']} "
                    f"--epochs {config['epochs']} "
                    f"--data {dataset_yaml} "
                    f"--weights {model_name}.pt "
                    f"--project {project_dir} "
                    f"--name {run_name} "
                    f"--hyp.lr0 {config['lr0']} "
                    f"--hyp.weight_decay {config['weight_decay']} "
                    f"--hyp.warmup_epochs {config['warmup_epochs']} "
                    f"--optimizer {config['optimizer']} "
                )

                if config['optimizer'] == 'SGD':
                    cmd += f" --hyp.momentum {config['momentum']} "

                logger.info(f"Training command: {cmd}")

                # Run the training
                os.system(cmd)

                # Check the results
                results_csv = project_dir / run_name / "results.csv"
                model_path = project_dir / run_name / "weights" / "best.pt"

                if model_path.exists() and results_csv.exists():
                    try:
                        # Read the results.csv to get validation mAP
                        import pandas as pd
                        results = pd.read_csv(results_csv)

                        # Get the best mAP value (usually the last row's metrics/mAP_0.5 column)
                        if 'metrics/mAP_0.5' in results.columns:
                            current_map = results['metrics/mAP_0.5'].max()
                            logger.info(
                                f"Configuration {i+1} achieved mAP@0.5 of {current_map:.4f}")

                            if current_map > best_map:
                                best_map = current_map
                                best_model_path = model_path
                                best_config = config
                                logger.info(
                                    f"New best model found! mAP@0.5: {best_map:.4f}")
                    except Exception as e:
                        logger.error(
                            f"Error reading results for configuration {i+1}: {str(e)}")

            # Report the best model
            if best_model_path is not None:
                logger.info("=" * 80)
                logger.info(f"Hyperparameter tuning complete!")
                logger.info(
                    f"Best configuration: {best_config['description']}")
                logger.info(f"Best mAP@0.5: {best_map:.4f}")
                logger.info(f"Best model path: {best_model_path}")
                logger.info("=" * 80)

                # Copy the best model to the expected location
                final_model_path = project_dir / \
                    f"{target}_detector_{model_size}" / "weights" / "best.pt"
                os.makedirs(final_model_path.parent, exist_ok=True)
                shutil.copy(best_model_path, final_model_path)

                # Update model path
                if target == "powerline":
                    self.powerline_model_path = str(final_model_path)
                else:
                    self.sleeve_model_path = str(final_model_path)

                return str(final_model_path)
            else:
                logger.error(
                    "Hyperparameter tuning failed to find a suitable model")
                return None

    def detect_powerlines(self,
                          images_dir: str,
                          output_dir: str,
                          conf_threshold: float = 0.25):
        """
        Detect powerlines in images using the trained powerline model.

        Args:
            images_dir: Directory containing images to process
            output_dir: Directory to save results
            conf_threshold: Confidence threshold for detections

        Returns:
            Path to the directory with processed images
        """
        if not self.powerline_model_path:
            logger.error(
                "Powerline model path not set. Train or load a model first.")
            return None

        # Set up paths
        images_path = Path(images_dir)
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)

        # Create subdirectories for results
        images_output = output_path / "images"
        labels_output = output_path / "labels"
        os.makedirs(images_output, exist_ok=True)
        os.makedirs(labels_output, exist_ok=True)

        # Prepare YOLOv5 detection command
        cmd = (
            f"python yolov5/detect.py "
            f"--weights {self.powerline_model_path} "
            f"--source {images_dir} "
            f"--conf {conf_threshold} "
            f"--save-txt "
            f"--save-conf "
            f"--project {output_path.parent} "
            f"--name {output_path.name} "
            f"--exist-ok"
        )

        logger.info(f"Running powerline detection")
        logger.info(f"Detection command: {cmd}")

        # Run the detection
        os.system(cmd)

        logger.info(
            f"Powerline detection complete. Results saved to {output_path}")
        return str(output_path)

    def extract_powerline_regions(self,
                                  detection_dir: str,
                                  output_dir: str,
                                  padding: float = 0.1):
        """
        Extract regions containing powerlines from detection results for sleeve labeling.

        Args:
            detection_dir: Directory containing powerline detection results
            output_dir: Directory to save extracted regions
            padding: Percentage of padding to add around detected powerlines

        Returns:
            Path to the directory with extracted powerline regions
        """
        # Set up paths
        detection_path = Path(detection_dir)
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)

        # Find detection results
        images_dir = detection_path / "images"
        labels_dir = detection_path / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            logger.error(f"Detection results not found in {detection_dir}")
            return None

        # Get all image files
        image_files = list(images_dir.glob("*.jpg")) + \
            list(images_dir.glob("*.png"))

        if not image_files:
            logger.error(f"No images found in {images_dir}")
            return None

        extracted_count = 0

        for img_file in tqdm(image_files, desc="Extracting powerline regions"):
            # Read image
            image = cv2.imread(str(img_file))
            if image is None:
                logger.warning(f"Could not read image: {img_file}")
                continue

            height, width = image.shape[:2]

            # Read corresponding label file
            label_file = labels_dir / f"{img_file.stem}.txt"

            if not label_file.exists():
                logger.warning(f"No labels found for image: {img_file.name}")
                continue

            # Read powerline bounding boxes
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Extract each powerline region
            region_idx = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Parse YOLO format (class_id, x_center, y_center, w, h)
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    # Convert to pixel coordinates with padding
                    x1 = max(0, int((x_center - w/2 - padding * w) * width))
                    y1 = max(0, int((y_center - h/2 - padding * h) * height))
                    x2 = min(width, int((x_center + w/2 + padding * w) * width))
                    y2 = min(height, int(
                        (y_center + h/2 + padding * h) * height))

                    # Extract the region
                    if x2 > x1 and y2 > y1:
                        region = image[y1:y2, x1:x2]

                        # Save the region
                        region_filename = f"{img_file.stem}_region{region_idx}{img_file.suffix}"
                        cv2.imwrite(str(output_path / region_filename), region)

                        extracted_count += 1
                        region_idx += 1

        logger.info(
            f"Extracted {extracted_count} powerline regions for sleeve labeling")
        return str(output_path)

    def run_complete_pipeline(self,
                              raw_images_dir: str,
                              output_base_dir: str,
                              powerline_model_path: str = None,
                              sleeve_model_path: str = None):
        """
        Run the complete two-stage detection pipeline if models are available.

        Args:
            raw_images_dir: Directory containing raw images to process
            output_base_dir: Base directory for outputs
            powerline_model_path: Path to pretrained powerline model (optional)
            sleeve_model_path: Path to pretrained sleeve model (optional)

        Returns:
            Dictionary with detection results
        """
        # Update model paths if provided
        if powerline_model_path:
            self.powerline_model_path = powerline_model_path

        if sleeve_model_path:
            self.sleeve_model_path = sleeve_model_path

        # Check if models are available
        if not self.powerline_model_path:
            logger.error(
                "Powerline model not available. Train or provide a model first.")
            return None

        if not self.sleeve_model_path:
            logger.warning(
                "Sleeve model not available. Will only detect powerlines.")

        # Set up output directories
        output_path = Path(output_base_dir)
        powerline_results = output_path / "powerline_results"
        sleeve_results = output_path / "sleeve_results" if self.sleeve_model_path else None

        # Step 1: Detect powerlines
        powerline_detection = self.detect_powerlines(
            images_dir=raw_images_dir,
            output_dir=str(powerline_results),
            conf_threshold=0.25
        )

        results = {
            "powerline_detection": powerline_detection
        }

        # Step 2: Detect sleeves if model is available
        if self.sleeve_model_path:
            # Process each detected powerline region
            extracted_regions = output_path / "powerline_regions"
            self.extract_powerline_regions(
                detection_dir=powerline_detection,
                output_dir=str(extracted_regions),
                padding=0.1
            )

            # Detect sleeves in the extracted regions
            sleeve_detection = self.detect_sleeves(
                images_dir=str(extracted_regions),
                output_dir=str(sleeve_results),
                conf_threshold=0.25
            )

            results["sleeve_detection"] = sleeve_detection

        return results

    def detect_sleeves(self,
                       images_dir: str,
                       output_dir: str,
                       conf_threshold: float = 0.25):
        """
        Detect sleeves in images using the trained sleeve model.

        Args:
            images_dir: Directory containing images to process
            output_dir: Directory to save results
            conf_threshold: Confidence threshold for detections

        Returns:
            Path to the directory with processed images
        """
        if not self.sleeve_model_path:
            logger.error(
                "Sleeve model path not set. Train or load a model first.")
            return None

        # Set up paths
        images_path = Path(images_dir)
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)

        # Prepare YOLOv5 detection command
        cmd = (
            f"python yolov5/detect.py "
            f"--weights {self.sleeve_model_path} "
            f"--source {images_dir} "
            f"--conf {conf_threshold} "
            f"--save-txt "
            f"--save-conf "
            f"--project {output_path.parent} "
            f"--name {output_path.name} "
            f"--exist-ok"
        )

        logger.info(f"Running sleeve detection")
        logger.info(f"Detection command: {cmd}")

        # Run the detection
        os.system(cmd)

        logger.info(
            f"Sleeve detection complete. Results saved to {output_path}")
        return str(output_path)
