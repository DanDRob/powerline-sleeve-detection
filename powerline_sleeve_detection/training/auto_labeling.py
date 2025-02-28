import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

from powerline_sleeve_detection.system.config import Config

logger = logging.getLogger(__name__)


class AutoLabeler:
    """Automated labeling for powerline sleeve detection."""

    def __init__(self, config: Config):
        """Initialize the auto labeler.

        Args:
            config: Application configuration
        """
        self.config = config
        self.model = None
        self.confidence_threshold = config.get(
            'auto_labeling.confidence_threshold', 0.5)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path: str) -> None:
        """Load a detection model.

        Args:
            model_path: Path to the model file
        """
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def label_images(self,
                     images_dir: Union[str, Path],
                     output_dir: Optional[Union[str, Path]] = None,
                     confidence_threshold: Optional[float] = None,
                     class_mapping: Optional[Dict[int, int]] = None) -> Tuple[int, int]:
        """Label images using a pre-trained model.

        Args:
            images_dir: Directory containing images to label
            output_dir: Directory to save labeled images and annotations
            confidence_threshold: Confidence threshold for detections
            class_mapping: Optional mapping of model class IDs to new class IDs

        Returns:
            Tuple of (total_images, labeled_images)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")

        confidence_threshold = confidence_threshold or self.confidence_threshold
        images_dir = Path(images_dir)

        if output_dir:
            output_dir = Path(output_dir)
            labels_dir = output_dir / 'labels'
            labels_dir.mkdir(exist_ok=True, parents=True)

            # Create images directory in output
            images_output_dir = output_dir / 'images'
            images_output_dir.mkdir(exist_ok=True, parents=True)
        else:
            output_dir = images_dir.parent
            labels_dir = output_dir / 'labels'
            labels_dir.mkdir(exist_ok=True, parents=True)
            images_output_dir = None

        # Find all images
        image_files = list(images_dir.glob('*.jpg')) + \
            list(images_dir.glob('*.png'))
        total_images = len(image_files)
        labeled_images = 0

        logger.info(f"Found {total_images} images to label")

        # Process each image
        for img_path in tqdm(image_files, desc="Labeling images"):
            try:
                # Run inference
                results = self.model(img_path, conf=confidence_threshold)

                # Skip if no detections
                if len(results) == 0 or len(results[0].boxes) == 0:
                    logger.debug(f"No detections in {img_path}")
                    continue

                # Get detections
                result = results[0]
                boxes = result.boxes

                # Create label file path
                label_path = labels_dir / f"{img_path.stem}.txt"

                # Write detections to file
                with open(label_path, 'w') as f:
                    for box in boxes:
                        cls_id = int(box.cls.item())

                        # Apply class mapping if provided
                        if class_mapping and cls_id in class_mapping:
                            cls_id = class_mapping[cls_id]

                        # Get normalized bounding box (YOLO format: x_center, y_center, width, height)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        # Get original image dimensions
                        img_width, img_height = result.orig_shape[1::-1]

                        # Convert to YOLO format (normalized)
                        x_center = (x1 + x2) / (2 * img_width)
                        y_center = (y1 + y2) / (2 * img_height)
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height

                        # Write to file
                        f.write(
                            f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                # Copy image to output directory if specified
                if images_output_dir:
                    import shutil
                    shutil.copy(img_path, images_output_dir / img_path.name)

                labeled_images += 1

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        logger.info(f"Labeled {labeled_images}/{total_images} images")
        return total_images, labeled_images

    def visualize_labels(self,
                         images_dir: Union[str, Path],
                         labels_dir: Union[str, Path],
                         output_dir: Union[str, Path],
                         class_names: Optional[List[str]] = None) -> int:
        """Visualize labeled images with bounding boxes.

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing label files
            output_dir: Directory to save visualizations
            class_names: Optional list of class names

        Returns:
            Number of images visualized
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Use default class names if not provided
        if class_names is None:
            class_names = self.config.get('training.class_names', ['sleeve'])

        # Find all label files
        label_files = list(labels_dir.glob('*.txt'))

        # Define colors for visualization (one per class)
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
        ]

        visualized = 0

        for label_path in tqdm(label_files, desc="Visualizing labels"):
            # Find corresponding image
            img_stem = label_path.stem
            img_path = None

            for ext in ['.jpg', '.png', '.jpeg']:
                potential_path = images_dir / f"{img_stem}{ext}"
                if potential_path.exists():
                    img_path = potential_path
                    break

            if img_path is None:
                logger.warning(f"No image found for label {label_path}")
                continue

            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Failed to read image {img_path}")
                continue

            # Get image dimensions
            height, width = image.shape[:2]

            # Read bounding boxes
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, bbox_width, bbox_height = map(
                                float, parts[1:5])

                            # Convert normalized YOLO format to pixel coordinates
                            x1 = int((x_center - bbox_width/2) * width)
                            y1 = int((y_center - bbox_height/2) * height)
                            x2 = int((x_center + bbox_width/2) * width)
                            y2 = int((y_center + bbox_height/2) * height)

                            # Ensure coordinates are within image boundaries
                            x1 = max(0, min(width-1, x1))
                            y1 = max(0, min(height-1, y1))
                            x2 = max(0, min(width-1, x2))
                            y2 = max(0, min(height-1, y2))

                            # Get color for this class
                            color = colors[class_id % len(colors)]

                            # Get class name
                            class_name = class_names[class_id] if class_id < len(
                                class_names) else f"Class {class_id}"

                            # Draw bounding box
                            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                            # Draw label
                            label_text = f"{class_name}"
                            cv2.putText(image, label_text, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Save visualization
                output_path = output_dir / \
                    f"{img_stem}_labeled{img_path.suffix}"
                cv2.imwrite(str(output_path), image)
                visualized += 1

            except Exception as e:
                logger.error(f"Error visualizing {label_path}: {e}")

        logger.info(
            f"Visualized {visualized}/{len(label_files)} labeled images")
        return visualized

    def semi_supervised_labeling(self,
                                 images_dir: Union[str, Path],
                                 output_dir: Union[str, Path],
                                 confidence_threshold: float = 0.8,
                                 review_threshold: float = 0.5) -> Dict[str, int]:
        """Perform semi-supervised labeling with high confidence predictions and flagging for review.

        Args:
            images_dir: Directory containing images
            output_dir: Directory to save labeled images and annotations
            confidence_threshold: Threshold for automatic acceptance
            review_threshold: Threshold for review flagging

        Returns:
            Dict with counts of auto-labeled, flagged, and total images
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")

        # Setup directories
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)

        labels_dir = output_dir / 'labels'
        labels_dir.mkdir(exist_ok=True, parents=True)

        images_output_dir = output_dir / 'images'
        images_output_dir.mkdir(exist_ok=True, parents=True)

        review_dir = output_dir / 'review'
        review_dir.mkdir(exist_ok=True, parents=True)

        review_labels_dir = review_dir / 'labels'
        review_labels_dir.mkdir(exist_ok=True, parents=True)

        review_images_dir = review_dir / 'images'
        review_images_dir.mkdir(exist_ok=True, parents=True)

        # Find all images
        image_files = list(images_dir.glob('*.jpg')) + \
            list(images_dir.glob('*.png'))
        total_images = len(image_files)

        auto_labeled = 0
        flagged_for_review = 0

        logger.info(
            f"Found {total_images} images for semi-supervised labeling")

        for img_path in tqdm(image_files, desc="Semi-supervised labeling"):
            try:
                # Run inference
                results = self.model(img_path)

                # Skip if no detections
                if len(results) == 0 or len(results[0].boxes) == 0:
                    logger.debug(f"No detections in {img_path}")
                    continue

                # Get detections
                result = results[0]
                boxes = result.boxes

                # Determine if any detection is below high confidence but above review threshold
                confidences = boxes.conf.tolist()
                needs_review = any(review_threshold <= conf <
                                   confidence_threshold for conf in confidences)
                has_high_confidence = any(
                    conf >= confidence_threshold for conf in confidences)

                # If any detection needs review, flag the entire image
                if needs_review:
                    # Copy image
                    import shutil
                    shutil.copy(img_path, review_images_dir / img_path.name)

                    # Save preliminary labels for review
                    label_path = review_labels_dir / f"{img_path.stem}.txt"
                    self._write_detections_to_file(result, boxes, label_path)

                    flagged_for_review += 1

                # If high confidence and not already flagged for review, auto-label
                elif has_high_confidence:
                    # Copy image
                    import shutil
                    shutil.copy(img_path, images_output_dir / img_path.name)

                    # Save labels
                    label_path = labels_dir / f"{img_path.stem}.txt"

                    # Only write high confidence detections
                    high_conf_boxes = [box for box, conf in zip(
                        boxes, confidences) if conf >= confidence_threshold]
                    self._write_detections_to_file(
                        result, high_conf_boxes, label_path)

                    auto_labeled += 1

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        logger.info(
            f"Auto-labeled: {auto_labeled}, Flagged for review: {flagged_for_review}, Total: {total_images}")

        # Create a README for review folder
        with open(review_dir / "README.md", 'w') as f:
            f.write("# Images Flagged for Review\n\n")
            f.write("These images have detections with confidence between "
                    f"{review_threshold} and {confidence_threshold}.\n")
            f.write(
                "Please review and correct the labels before including them in the dataset.\n\n")
            f.write("## Reviewing Process\n")
            f.write("1. Check if the bounding boxes are accurate\n")
            f.write("2. Adjust bounding boxes if needed\n")
            f.write("3. Remove false positives\n")
            f.write("4. Add missing detections\n")
            f.write("5. Move the corrected labels to the main labels directory\n")

        return {
            "auto_labeled": auto_labeled,
            "flagged_for_review": flagged_for_review,
            "total": total_images
        }

    def _write_detections_to_file(self, result, boxes, label_path: Path) -> None:
        """Write detection results to a YOLO format label file.

        Args:
            result: Model result object
            boxes: Detection boxes
            label_path: Path to output label file
        """
        with open(label_path, 'w') as f:
            for box in boxes:
                cls_id = int(box.cls.item())

                # Get normalized bounding box (YOLO format: x_center, y_center, width, height)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Get original image dimensions
                img_width, img_height = result.orig_shape[1::-1]

                # Convert to YOLO format (normalized)
                x_center = (x1 + x2) / (2 * img_width)
                y_center = (y1 + y2) / (2 * img_height)
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                # Write to file
                f.write(
                    f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
