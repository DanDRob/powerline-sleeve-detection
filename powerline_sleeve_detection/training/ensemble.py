import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

from powerline_sleeve_detection.system.config import Config

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """Ensemble detector for combining predictions from multiple models."""

    def __init__(self, config: Config):
        """Initialize the ensemble detector with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.models = []
        self.model_weights = []
        self.ensemble_method = config.get(
            'ensemble.method', 'weighted_boxes_fusion')
        self.iou_threshold = config.get('ensemble.iou_threshold', 0.5)
        self.score_threshold = config.get('ensemble.score_threshold', 0.1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def add_model(self, model_path: str, weight: float = 1.0) -> None:
        """Add a model to the ensemble.

        Args:
            model_path: Path to model file
            weight: Weight for this model's predictions
        """
        try:
            model = YOLO(model_path)
            self.models.append(model)
            self.model_weights.append(weight)
            logger.info(
                f"Added model {model_path} to ensemble with weight {weight}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise

    def load_models_from_config(self) -> None:
        """Load models specified in the configuration."""
        ensemble_models = self.config.get('ensemble.models', [])

        if not ensemble_models:
            logger.warning("No ensemble models specified in config")
            return

        for model_config in ensemble_models:
            model_path = model_config.get('path')
            weight = model_config.get('weight', 1.0)

            if model_path:
                self.add_model(model_path, weight)

        logger.info(f"Loaded {len(self.models)} models for ensemble")

    def predict(self,
                image_path: Union[str, Path],
                return_visualization: bool = False) -> Dict[str, Any]:
        """Run ensemble prediction on an image.

        Args:
            image_path: Path to input image
            return_visualization: Whether to return visualization image

        Returns:
            Dictionary with prediction results
        """
        if not self.models:
            raise ValueError("No models loaded in ensemble")

        # Get predictions from each model
        all_predictions = []

        for i, model in enumerate(self.models):
            try:
                results = model(image_path, conf=self.score_threshold)

                if results and len(results) > 0:
                    result = results[0]
                    all_predictions.append({
                        'boxes': result.boxes,
                        'weight': self.model_weights[i]
                    })

            except Exception as e:
                logger.error(f"Error during prediction with model {i}: {e}")

        if not all_predictions:
            logger.warning(f"No predictions from any model for {image_path}")
            return {'boxes': None, 'labels': [], 'scores': [], 'image': None}

        # Combine predictions
        if self.ensemble_method == 'weighted_boxes_fusion':
            combined_boxes, combined_scores, combined_labels = self._weighted_boxes_fusion(
                all_predictions)
        elif self.ensemble_method == 'non_maximum_suppression':
            combined_boxes, combined_scores, combined_labels = self._non_maximum_suppression(
                all_predictions)
        else:
            logger.warning(
                f"Unknown ensemble method: {self.ensemble_method}, using weighted boxes fusion")
            combined_boxes, combined_scores, combined_labels = self._weighted_boxes_fusion(
                all_predictions)

        # Prepare result
        result = {
            'boxes': combined_boxes,
            'scores': combined_scores,
            'labels': combined_labels
        }

        # Add visualization if requested
        if return_visualization and combined_boxes is not None:
            try:
                image = cv2.imread(str(image_path))
                visualization = self._draw_predictions(
                    image, combined_boxes, combined_scores, combined_labels)
                result['image'] = visualization
            except Exception as e:
                logger.error(f"Error creating visualization: {e}")

        return result

    def _weighted_boxes_fusion(self, predictions: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weighted Boxes Fusion algorithm for ensemble predictions.

        Args:
            predictions: List of prediction dictionaries from different models

        Returns:
            Tuple of (boxes, scores, labels)
        """
        # Extract all detections
        all_boxes = []
        all_scores = []
        all_labels = []
        weights = []

        for pred in predictions:
            boxes = pred['boxes']
            if boxes is None or len(boxes) == 0:
                continue

            # Convert boxes to numpy arrays
            try:
                # Get boxes in xyxy format
                xyxy = boxes.xyxy.cpu().numpy()

                # Get scores
                scores = boxes.conf.cpu().numpy()

                # Get class labels
                labels = boxes.cls.cpu().numpy().astype(int)

                all_boxes.append(xyxy)
                all_scores.append(scores)
                all_labels.append(labels)
                weights.append(pred['weight'])
            except Exception as e:
                logger.error(f"Error processing predictions: {e}")

        if not all_boxes:
            return None, np.array([]), np.array([])

        # Combine all detections into single arrays
        boxes_list = []
        scores_list = []
        labels_list = []
        model_indices = []

        for model_idx, (boxes, scores, labels) in enumerate(zip(all_boxes, all_scores, all_labels)):
            for box, score, label in zip(boxes, scores, labels):
                boxes_list.append(box)
                # Apply model weight
                scores_list.append(score * weights[model_idx])
                labels_list.append(label)
                model_indices.append(model_idx)

        if not boxes_list:
            return None, np.array([]), np.array([])

        # Convert to numpy arrays
        boxes_array = np.array(boxes_list)
        scores_array = np.array(scores_list)
        labels_array = np.array(labels_list)

        # Group by label
        unique_labels = np.unique(labels_array)
        final_boxes = []
        final_scores = []
        final_labels = []

        for label in unique_labels:
            # Get detections for this class
            mask = labels_array == label
            class_boxes = boxes_array[mask]
            class_scores = scores_array[mask]

            # Apply non-maximum suppression
            keep_indices = self._nms(
                class_boxes, class_scores, self.iou_threshold)

            for idx in keep_indices:
                final_boxes.append(class_boxes[idx])
                final_scores.append(class_scores[idx])
                final_labels.append(label)

        # Convert to numpy arrays
        if final_boxes:
            return np.array(final_boxes), np.array(final_scores), np.array(final_labels)
        else:
            return None, np.array([]), np.array([])

    def _non_maximum_suppression(self, predictions: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Non-Maximum Suppression for combining ensemble predictions.

        Args:
            predictions: List of prediction dictionaries from different models

        Returns:
            Tuple of (boxes, scores, labels)
        """
        # Similar to weighted boxes fusion but without weighting
        # Extract all detections
        all_boxes = []
        all_scores = []
        all_labels = []

        for pred in predictions:
            boxes = pred['boxes']
            if boxes is None or len(boxes) == 0:
                continue

            # Convert boxes to numpy arrays
            try:
                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                labels = boxes.cls.cpu().numpy().astype(int)

                all_boxes.append(xyxy)
                all_scores.append(scores)
                all_labels.append(labels)
            except Exception as e:
                logger.error(f"Error processing predictions: {e}")

        if not all_boxes:
            return None, np.array([]), np.array([])

        # Combine all detections
        boxes_list = []
        scores_list = []
        labels_list = []

        for boxes, scores, labels in zip(all_boxes, all_scores, all_labels):
            for box, score, label in zip(boxes, scores, labels):
                boxes_list.append(box)
                scores_list.append(score)
                labels_list.append(label)

        if not boxes_list:
            return None, np.array([]), np.array([])

        # Convert to numpy arrays
        boxes_array = np.array(boxes_list)
        scores_array = np.array(scores_list)
        labels_array = np.array(labels_list)

        # Group by label
        unique_labels = np.unique(labels_array)
        final_boxes = []
        final_scores = []
        final_labels = []

        for label in unique_labels:
            # Get detections for this class
            mask = labels_array == label
            class_boxes = boxes_array[mask]
            class_scores = scores_array[mask]

            # Sort by score
            score_indices = np.argsort(class_scores)[::-1]
            class_boxes = class_boxes[score_indices]
            class_scores = class_scores[score_indices]

            # Apply NMS
            keep_indices = self._nms(
                class_boxes, class_scores, self.iou_threshold)

            for idx in keep_indices:
                final_boxes.append(class_boxes[idx])
                final_scores.append(class_scores[idx])
                final_labels.append(label)

        # Convert to numpy arrays
        if final_boxes:
            return np.array(final_boxes), np.array(final_scores), np.array(final_labels)
        else:
            return None, np.array([]), np.array([])

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression algorithm.

        Args:
            boxes: Bounding boxes in [x1, y1, x2, y2] format
            scores: Confidence scores
            iou_threshold: IoU threshold for suppression

        Returns:
            List of indices to keep
        """
        # If no boxes, return empty list
        if len(boxes) == 0:
            return []

        # Sort by score
        idxs = np.argsort(scores)[::-1]

        # Initialize list of picked indices
        picked = []

        while len(idxs) > 0:
            # Pick the index with highest score
            current = idxs[0]
            picked.append(current)

            # If only one box left, break
            if len(idxs) == 1:
                break

            # Get remaining indices
            idxs = idxs[1:]

            # Get coordinates of current and remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[idxs]

            # Calculate IoU between current box and remaining boxes
            xx1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            yy1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            xx2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            yy2 = np.minimum(current_box[3], remaining_boxes[:, 3])

            # Calculate width and height of overlap
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            # Calculate overlap area
            overlap = w * h

            # Calculate area of boxes
            area_current = (
                current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (
                remaining_boxes[:, 3] - remaining_boxes[:, 1])

            # Calculate IoU
            iou = overlap / (area_current + area_remaining - overlap)

            # Keep boxes with IoU below threshold
            idxs = idxs[iou < iou_threshold]

        return picked

    def _draw_predictions(self,
                          image: np.ndarray,
                          boxes: np.ndarray,
                          scores: np.ndarray,
                          labels: np.ndarray) -> np.ndarray:
        """Draw predictions on an image.

        Args:
            image: Input image
            boxes: Bounding boxes in [x1, y1, x2, y2] format
            scores: Confidence scores
            labels: Class labels

        Returns:
            Image with drawn predictions
        """
        # Colors for different classes
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
        ]

        # Get class names
        class_names = self.config.get('training.class_names', ['sleeve'])

        # Draw each prediction
        result = image.copy()

        for box, score, label in zip(boxes, scores, labels):
            # Get integer coordinates
            x1, y1, x2, y2 = box.astype(int)

            # Get color for this class
            color = colors[int(label) % len(colors)]

            # Get class name
            class_name = class_names[int(label)] if int(
                label) < len(class_names) else f"Class {int(label)}"

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Draw label and score
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(result, label_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result

    def evaluate_ensemble(self,
                          test_dir: Union[str, Path],
                          ground_truth_dir: Union[str, Path]) -> Dict[str, float]:
        """Evaluate ensemble performance on a test dataset.

        Args:
            test_dir: Directory with test images
            ground_truth_dir: Directory with ground truth labels (YOLO format)

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.models:
            raise ValueError("No models loaded in ensemble")

        test_dir = Path(test_dir)
        ground_truth_dir = Path(ground_truth_dir)

        # Find all images
        image_files = list(test_dir.glob('*.jpg')) + \
            list(test_dir.glob('*.png'))

        if not image_files:
            logger.warning(f"No images found in {test_dir}")
            return {}

        # Initialize metrics
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        total_iou = 0
        total_detections = 0

        logger.info(f"Evaluating ensemble on {len(image_files)} images")

        for img_path in tqdm(image_files, desc="Evaluating ensemble"):
            # Get ground truth label file
            gt_file = ground_truth_dir / f"{img_path.stem}.txt"

            if not gt_file.exists():
                logger.warning(
                    f"No ground truth label found for {img_path.name}")
                continue

            # Run ensemble prediction
            prediction = self.predict(img_path)
            pred_boxes = prediction.get('boxes')
            pred_scores = prediction.get('scores')
            pred_labels = prediction.get('labels')

            # Load ground truth
            gt_boxes, gt_labels = self._load_yolo_labels(gt_file, img_path)

            # If no predictions or no ground truth, continue
            if pred_boxes is None or len(pred_boxes) == 0:
                # All ground truth objects are false negatives
                total_false_negatives += len(
                    gt_boxes) if gt_boxes is not None else 0
                continue

            if gt_boxes is None or len(gt_boxes) == 0:
                # All predictions are false positives
                total_false_positives += len(pred_boxes)
                continue

            # Match predictions to ground truth
            true_positives, false_positives, false_negatives, matches = self._match_detections(
                pred_boxes, pred_labels, gt_boxes, gt_labels
            )

            # Calculate IoU for true positives
            iou_sum = 0
            for pred_idx, gt_idx in matches:
                iou = self._calculate_iou(
                    pred_boxes[pred_idx], gt_boxes[gt_idx])
                iou_sum += iou

            # Update metrics
            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives
            total_iou += iou_sum
            total_detections += len(matches)

        # Calculate final metrics
        precision = total_true_positives / (total_true_positives + total_false_positives) if (
            total_true_positives + total_false_positives) > 0 else 0
        recall = total_true_positives / (total_true_positives + total_false_negatives) if (
            total_true_positives + total_false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = total_iou / total_detections if total_detections > 0 else 0

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mean_iou': mean_iou,
            'true_positives': total_true_positives,
            'false_positives': total_false_positives,
            'false_negatives': total_false_negatives
        }

        logger.info(f"Ensemble evaluation results: {metrics}")
        return metrics

    def _load_yolo_labels(self,
                          label_file: Path,
                          image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load YOLO format labels and convert to [x1, y1, x2, y2] format.

        Args:
            label_file: Path to label file
            image_path: Path to corresponding image

        Returns:
            Tuple of (boxes, labels)
        """
        try:
            # Get image dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image {image_path}")
                return None, None

            img_height, img_width = image.shape[:2]

            boxes = []
            labels = []

            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO format: class_id, x_center, y_center, width, height (normalized)
                        x_center, y_center, width, height = map(
                            float, parts[1:5])

                        # Convert to pixel coordinates
                        x_center *= img_width
                        y_center *= img_height
                        width *= img_width
                        height *= img_height

                        # Convert to [x1, y1, x2, y2] format
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2

                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)

            if boxes:
                return np.array(boxes), np.array(labels)
            else:
                return None, None

        except Exception as e:
            logger.error(f"Error loading labels from {label_file}: {e}")
            return None, None

    def _match_detections(self,
                          pred_boxes: np.ndarray,
                          pred_labels: np.ndarray,
                          gt_boxes: np.ndarray,
                          gt_labels: np.ndarray) -> Tuple[int, int, int, List]:
        """Match predicted detections to ground truth.

        Args:
            pred_boxes: Predicted bounding boxes
            pred_labels: Predicted class labels
            gt_boxes: Ground truth bounding boxes
            gt_labels: Ground truth class labels

        Returns:
            Tuple of (true_positives, false_positives, false_negatives, matches)
        """
        # Initialize counters
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Initialize array to track matched ground truth boxes
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        # List to store matched pairs (pred_idx, gt_idx)
        matches = []

        # Sort predictions by confidence (if available)
        if hasattr(pred_boxes, 'scores'):
            pred_indices = np.argsort(pred_boxes.scores)[::-1]
        else:
            pred_indices = np.arange(len(pred_boxes))

        # Match each prediction to ground truth
        for pred_idx in pred_indices:
            pred_box = pred_boxes[pred_idx]
            pred_label = pred_labels[pred_idx]

            # Find best matching ground truth box
            best_iou = self.iou_threshold
            best_gt_idx = -1

            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                # Skip already matched ground truth boxes
                if gt_matched[gt_idx]:
                    continue

                # Skip if class labels don't match
                if pred_label != gt_label:
                    continue

                # Calculate IoU
                iou = self._calculate_iou(pred_box, gt_box)

                # Update best match if IoU is higher
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if a match was found
            if best_gt_idx >= 0:
                # Mark as matched and count as true positive
                gt_matched[best_gt_idx] = True
                true_positives += 1
                matches.append((pred_idx, best_gt_idx))
            else:
                # No match found, count as false positive
                false_positives += 1

        # Count unmatched ground truth boxes as false negatives
        false_negatives = np.sum(~gt_matched)

        return true_positives, false_positives, false_negatives, matches

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two boxes.

        Args:
            box1: First box in [x1, y1, x2, y2] format
            box2: Second box in [x1, y1, x2, y2] format

        Returns:
            IoU value
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate intersection area
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0

        return iou
