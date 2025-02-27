import os
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AugmentationPipeline:
    """Provides data augmentation for powerline sleeve detection training."""

    def __init__(self, severity: str = 'medium'):
        """Initialize the augmentation pipeline.

        Args:
            severity: Augmentation severity level ('light', 'medium', 'heavy')
        """
        self.severity = severity
        self.transform = self._create_transform(severity)

    def _create_transform(self, severity: str) -> A.Compose:
        """Create augmentation transform pipeline based on severity.

        Args:
            severity: Augmentation severity level

        Returns:
            Albumentations transform pipeline
        """
        # Base transforms (always applied)
        base_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ]

        # Add more transforms based on severity
        if severity == 'light':
            additional_transforms = [
                A.RandomShadow(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ]
        elif severity == 'medium':
            additional_transforms = [
                A.RandomShadow(p=0.4),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.CLAHE(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.RandomScale(scale_limit=0.1, p=0.5),
            ]
        elif severity == 'heavy':
            additional_transforms = [
                A.RandomShadow(p=0.5),
                A.GaussianBlur(blur_limit=(3, 9), p=0.4),
                A.CLAHE(p=0.4),
                A.Rotate(limit=30, p=0.6),
                A.RandomScale(scale_limit=0.15, p=0.6),
                A.Perspective(p=0.3),
                A.GridDistortion(p=0.3),
                A.RGBShift(p=0.3),
            ]
        else:
            logger.warning(f"Unknown severity '{severity}', using 'medium'")
            additional_transforms = [
                A.RandomShadow(p=0.4),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.CLAHE(p=0.3),
                A.Rotate(limit=15, p=0.5),
            ]

        # Combine transforms
        transforms = base_transforms + additional_transforms

        # Create the composition with bounding box format
        return A.Compose(transforms, bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))

    def augment_dataset(self,
                        dataset_path: Union[str, Path],
                        output_path: Optional[Union[str, Path]] = None,
                        multiplier: int = 3) -> Tuple[int, int]:
        """Augment a YOLO dataset.

        Args:
            dataset_path: Path to the dataset (containing images and labels subdirs)
            output_path: Path to save augmented data (if None, appends to original)
            multiplier: Number of augmentations per original image

        Returns:
            Tuple of (original_count, augmented_count)
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {dataset_path}")

        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'

        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(
                f"Images or labels directory not found in {dataset_path}")

        # Set up output directory
        if output_path:
            output_path = Path(output_path)
            output_images_dir = output_path / 'images'
            output_labels_dir = output_path / 'labels'
            output_images_dir.mkdir(exist_ok=True, parents=True)
            output_labels_dir.mkdir(exist_ok=True, parents=True)
        else:
            output_images_dir = images_dir
            output_labels_dir = labels_dir

        # Get image-label pairs
        image_files = list(images_dir.glob('*.jpg')) + \
            list(images_dir.glob('*.png'))
        original_count = len(image_files)
        augmented_count = 0

        logger.info(
            f"Augmenting {original_count} images with multiplier {multiplier}...")

        for img_path in tqdm(image_files, desc="Augmenting images"):
            # Find corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                logger.warning(
                    f"No label file found for {img_path.name}, skipping")
                continue

            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Failed to read image {img_path}, skipping")
                continue

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read bounding boxes from label file
            bboxes, class_labels = self._read_yolo_labels(label_path)

            # Skip if no bounding boxes
            if not bboxes:
                logger.warning(
                    f"No valid bounding boxes in {label_path}, skipping")
                continue

            # Perform augmentation multiple times
            for i in range(multiplier):
                # Apply augmentations
                try:
                    transformed = self.transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )

                    augmented_image = transformed['image']
                    augmented_bboxes = transformed['bboxes']
                    augmented_labels = transformed['class_labels']

                    if not augmented_bboxes:
                        continue  # Skip if bounding boxes were lost

                    # Save augmented image
                    aug_img_filename = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                    aug_img_path = output_images_dir / aug_img_filename

                    # Convert RGB back to BGR for OpenCV
                    augmented_image = cv2.cvtColor(
                        augmented_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_img_path), augmented_image)

                    # Save augmented labels
                    aug_label_filename = f"{img_path.stem}_aug_{i}.txt"
                    aug_label_path = output_labels_dir / aug_label_filename
                    self._write_yolo_labels(
                        aug_label_path, augmented_bboxes, augmented_labels)

                    augmented_count += 1
                except Exception as e:
                    logger.error(f"Error during augmentation: {e}")
                    continue

        logger.info(
            f"Augmentation complete. Original: {original_count}, Augmented: {augmented_count}")
        return original_count, augmented_count

    def _read_yolo_labels(self, label_path: Path) -> Tuple[List, List]:
        """Read bounding boxes from a YOLO label file.

        Args:
            label_path: Path to the label file

        Returns:
            Tuple of (bboxes, class_labels)
        """
        bboxes = []
        class_labels = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # Validate bounding box coordinates
                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)

        return bboxes, class_labels

    def _write_yolo_labels(self,
                           label_path: Path,
                           bboxes: List,
                           class_labels: List) -> None:
        """Write bounding boxes to a YOLO label file.

        Args:
            label_path: Path to the output label file
            bboxes: List of bounding boxes in YOLO format
            class_labels: List of class labels
        """
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                if len(bbox) >= 4:
                    # Ensure values are in range [0,1]
                    x_center = max(0, min(1, bbox[0]))
                    y_center = max(0, min(1, bbox[1]))
                    width = max(0.001, min(1, bbox[2]))
                    height = max(0.001, min(1, bbox[3]))

                    f.write(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    @staticmethod
    def preview_augmentations(image_path: str,
                              label_path: str,
                              severity: str = 'medium',
                              num_samples: int = 5,
                              output_dir: Optional[str] = None) -> None:
        """Generate preview of augmentations for a single image.

        Args:
            image_path: Path to the image
            label_path: Path to the YOLO label file
            severity: Augmentation severity level
            num_samples: Number of augmented samples to generate
            output_dir: Output directory (if None, uses current directory)
        """
        try:
            # Create augmentation pipeline
            augmenter = AugmentationPipeline(severity=severity)

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image {image_path}")
                return

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load labels
            bboxes, class_labels = augmenter._read_yolo_labels(
                Path(label_path))
            if not bboxes:
                logger.error(f"No valid bounding boxes in {label_path}")
                return

            # Create output directory
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True, parents=True)
            else:
                output_path = Path('.')

            # Draw original image with bounding boxes
            original_vis = image.copy()
            original_vis = augmenter._draw_bboxes(
                original_vis, bboxes, class_labels)

            # Save original visualization
            img_basename = Path(image_path).stem
            original_out_path = output_path / f"{img_basename}_original.jpg"
            cv2.imwrite(str(original_out_path), cv2.cvtColor(
                original_vis, cv2.COLOR_RGB2BGR))

            # Generate augmented samples
            for i in range(num_samples):
                transformed = augmenter.transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                aug_image = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_labels = transformed['class_labels']

                # Draw bounding boxes
                aug_vis = augmenter._draw_bboxes(
                    aug_image, aug_bboxes, aug_labels)

                # Save visualization
                aug_out_path = output_path / f"{img_basename}_aug_{i}.jpg"
                cv2.imwrite(str(aug_out_path), cv2.cvtColor(
                    aug_vis, cv2.COLOR_RGB2BGR))

            logger.info(
                f"Saved {num_samples} augmentation previews to {output_path}")

        except Exception as e:
            logger.error(f"Error generating augmentation previews: {e}")

    def _draw_bboxes(self,
                     image: np.ndarray,
                     bboxes: List,
                     class_labels: List,
                     color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw bounding boxes on an image.

        Args:
            image: The image to draw on
            bboxes: List of bounding boxes in YOLO format
            class_labels: List of class labels
            color: RGB color tuple for boxes

        Returns:
            Image with drawn bounding boxes
        """
        height, width = image.shape[:2]
        result = image.copy()

        for bbox, class_id in zip(bboxes, class_labels):
            if len(bbox) >= 4:
                x_center, y_center, w, h = bbox

                # Convert from normalized coordinates to pixel coordinates
                x1 = int((x_center - w/2) * width)
                y1 = int((y_center - h/2) * height)
                x2 = int((x_center + w/2) * width)
                y2 = int((y_center + h/2) * height)

                # Ensure coordinates are within image boundaries
                x1 = max(0, min(width-1, x1))
                y1 = max(0, min(height-1, y1))
                x2 = max(0, min(width-1, x2))
                y2 = max(0, min(height-1, y2))

                # Draw rectangle
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

                # Add label
                label_text = f"Class {class_id}"
                cv2.putText(result, label_text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result
