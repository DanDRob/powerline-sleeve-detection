#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ..system.config import Config
from ..system.logging import get_logger, setup_logging
from .ensemble import EnsembleDetector
from .monitor import TrainingMonitor

logger = get_logger("evaluate_model")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate powerline sleeve detection models")

    # Main command options
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["single", "ensemble", "compare"],
                        help="Evaluation mode")

    # Model options
    parser.add_argument("--model-path", type=str,
                        help="Path to model for single evaluation")
    parser.add_argument("--model-paths", type=str, nargs="+",
                        help="Multiple model paths for comparison")
    parser.add_argument("--model-names", type=str, nargs="+",
                        help="Names for models being compared")

    # Data options
    parser.add_argument("--test-dir", type=str,
                        help="Directory containing test images")
    parser.add_argument("--ground-truth-dir", type=str,
                        help="Directory containing ground truth labels")
    parser.add_argument("--dataset-yaml", type=str,
                        help="YAML file for dataset (for YOLOv8 .val())")

    # Output options
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--save-visualizations", action="store_true", default=True,
                        help="Save detection visualizations")

    # Evaluation parameters
    parser.add_argument("--confidence-threshold", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for mAP calculation")

    return parser.parse_args()


def evaluate_single_model(config: Config, args) -> Dict[str, Any]:
    """Evaluate a single model on a test dataset."""
    logger.info("Evaluating single model")

    model_path = args.model_path
    if not model_path:
        model_path = config.get('training.custom_model_path')
        if not model_path:
            model_path = config.get('detection.model_paths.custom_model')
            if not model_path:
                raise ValueError("Model path is required for evaluation")

    # Determine evaluation method
    if args.dataset_yaml:
        # Use YOLOv8's built-in validation
        return evaluate_with_yolo_val(config, args, model_path)
    elif args.test_dir and args.ground_truth_dir:
        # Use custom evaluation
        return evaluate_on_test_dir(config, args, model_path)
    else:
        raise ValueError(
            "Either dataset_yaml or test_dir + ground_truth_dir must be provided")


def evaluate_with_yolo_val(config: Config, args, model_path: str) -> Dict[str, Any]:
    """Evaluate a model using YOLOv8's .val() method."""
    logger.info(
        f"Evaluating model {model_path} on dataset {args.dataset_yaml}")

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Load the model
        model = YOLO(model_path)

        # Run validation
        results = model.val(
            data=args.dataset_yaml,
            conf=args.confidence_threshold,
            iou=args.iou_threshold,
            save_json=True,
            save_hybrid=True,
            project=str(output_dir),
            name="yolo_validation"
        )

        # Extract metrics
        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "speed": results.speed.get('inference', 0.0)
        }

        # Format and save results
        result_file = output_dir / "results.json"
        with open(result_file, 'w') as f:
            import json
            json.dump({
                "model": model_path,
                "metrics": metrics,
                "config": {
                    "confidence_threshold": args.confidence_threshold,
                    "iou_threshold": args.iou_threshold,
                    "dataset": args.dataset_yaml
                }
            }, f, indent=2)

        logger.info(f"Evaluation results: mAP50={metrics['mAP50']:.4f}, "
                    f"mAP50-95={metrics['mAP50-95']:.4f}")
        logger.info(f"Results saved to {result_file}")

        return {
            "model": model_path,
            "metrics": metrics,
            "output_dir": str(output_dir),
            "results_file": str(result_file)
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def evaluate_on_test_dir(config: Config, args, model_path: str) -> Dict[str, Any]:
    """Evaluate a model on a directory of test images with ground truth labels."""
    logger.info(
        f"Evaluating model {model_path} on test images in {args.test_dir}")

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    viz_dir = output_dir / "visualizations"
    if args.save_visualizations:
        viz_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Load model
        model = YOLO(model_path)

        # Get all test images
        test_dir = Path(args.test_dir)
        gt_dir = Path(args.ground_truth_dir)

        image_files = list(test_dir.glob("**/*.jpg")) + \
            list(test_dir.glob("**/*.png"))
        logger.info(f"Found {len(image_files)} test images")

        # Set up metrics collection
        all_metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precisions": [],
            "recalls": [],
            "f1_scores": [],
            "confidences": [],
            "ious": []
        }

        # Process each image
        for img_file in tqdm(image_files, desc="Evaluating images"):
            # Find corresponding ground truth label
            label_file = gt_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                logger.warning(f"No ground truth label found for {img_file}")
                continue

            # Get predictions
            results = model(img_file, conf=args.confidence_threshold)

            # Load ground truth boxes
            gt_boxes, gt_classes = load_yolo_labels(label_file, img_file)

            # Get predicted boxes
            pred_boxes = []
            pred_classes = []
            confidences = []

            for r in results:
                boxes = r.boxes.xywh.cpu().numpy()  # x, y, width, height format
                classes = r.boxes.cls.cpu().numpy()  # class ids
                confs = r.boxes.conf.cpu().numpy()  # confidences

                for box, cls, conf in zip(boxes, classes, confs):
                    pred_boxes.append(box)
                    pred_classes.append(cls)
                    confidences.append(conf)

            # Calculate metrics
            tp, fp, fn, matches = match_predictions_to_ground_truth(
                np.array(pred_boxes) if pred_boxes else np.empty((0, 4)),
                np.array(pred_classes) if pred_classes else np.empty(0),
                np.array(gt_boxes) if gt_boxes else np.empty((0, 4)),
                np.array(gt_classes) if gt_classes else np.empty(0),
                iou_threshold=args.iou_threshold
            )

            # Update metrics
            all_metrics["true_positives"] += tp
            all_metrics["false_positives"] += fp
            all_metrics["false_negatives"] += fn

            # Calculate image-level metrics
            if tp + fp > 0:
                precision = tp / (tp + fp)
                all_metrics["precisions"].append(precision)
            if tp + fn > 0:
                recall = tp / (tp + fn)
                all_metrics["recalls"].append(recall)
            if tp + fp + fn > 0:
                f1 = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0
                all_metrics["f1_scores"].append(f1)

            # Save visualization if enabled
            if args.save_visualizations:
                save_detection_visualization(
                    img_file, pred_boxes, gt_boxes, matches,
                    pred_classes, gt_classes, confidences,
                    viz_dir / img_file.name
                )

        # Calculate overall metrics
        total_tp = all_metrics["true_positives"]
        total_fp = all_metrics["false_positives"]
        total_fn = all_metrics["false_negatives"]

        overall_precision = total_tp / \
            (total_tp + total_fp) if total_tp + total_fp > 0 else 0
        overall_recall = total_tp / \
            (total_tp + total_fn) if total_tp + total_fn > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / \
            (overall_precision + overall_recall) if overall_precision + \
            overall_recall > 0 else 0

        avg_precision = np.mean(
            all_metrics["precisions"]) if all_metrics["precisions"] else 0
        avg_recall = np.mean(
            all_metrics["recalls"]) if all_metrics["recalls"] else 0
        avg_f1 = np.mean(all_metrics["f1_scores"]
                         ) if all_metrics["f1_scores"] else 0

        metrics = {
            "precision": float(overall_precision),
            "recall": float(overall_recall),
            "f1_score": float(overall_f1),
            "avg_precision": float(avg_precision),
            "avg_recall": float(avg_recall),
            "avg_f1": float(avg_f1),
            "true_positives": int(total_tp),
            "false_positives": int(total_fp),
            "false_negatives": int(total_fn)
        }

        # Format and save results
        result_file = output_dir / "results.json"
        with open(result_file, 'w') as f:
            import json
            json.dump({
                "model": model_path,
                "metrics": metrics,
                "config": {
                    "confidence_threshold": args.confidence_threshold,
                    "iou_threshold": args.iou_threshold,
                    "test_dir": args.test_dir,
                    "ground_truth_dir": args.ground_truth_dir
                }
            }, f, indent=2)

        # Create summary plots
        create_metric_plots(metrics, output_dir)

        logger.info(f"Evaluation results: Precision={metrics['precision']:.4f}, "
                    f"Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
        logger.info(f"Results saved to {result_file}")

        return {
            "model": model_path,
            "metrics": metrics,
            "output_dir": str(output_dir),
            "results_file": str(result_file)
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def evaluate_ensemble(config: Config, args) -> Dict[str, Any]:
    """Evaluate an ensemble of models on a test dataset."""
    logger.info("Evaluating ensemble model")

    if not args.test_dir or not args.ground_truth_dir:
        raise ValueError(
            "Test directory and ground truth directory are required for ensemble evaluation")

    # Initialize ensemble detector
    ensemble_detector = EnsembleDetector(config)

    # If model paths specified, use them; otherwise use from config
    if args.model_paths:
        # Clear existing models
        ensemble_detector.models = []
        ensemble_detector.weights = []

        # Add models with equal weights if not specified
        weights = args.weights if hasattr(args, 'weights') and args.weights else [
            1.0] * len(args.model_paths)

        for i, model_path in enumerate(args.model_paths):
            ensemble_detector.add_model(model_path, weights[i])
    else:
        # Load models from config
        ensemble_detector.load_models_from_config()

    if not ensemble_detector.models:
        raise ValueError(
            "No models added to ensemble. Configure models in config.yaml or pass --model-paths")

    logger.info(
        f"Ensemble evaluation with {len(ensemble_detector.models)} models")

    # Run ensemble evaluation
    results = ensemble_detector.evaluate_ensemble(
        test_dir=args.test_dir,
        ground_truth_dir=args.ground_truth_dir
    )

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save results
    result_file = output_dir / "ensemble_results.json"
    with open(result_file, 'w') as f:
        import json
        json.dump({
            "models": [str(m) for m in ensemble_detector.models],
            "weights": [float(w) for w in ensemble_detector.weights],
            "metrics": results,
            "config": {
                "ensemble_method": config.get('ensemble.method', 'weighted_boxes_fusion'),
                "iou_threshold": config.get('ensemble.iou_threshold', 0.5),
                "score_threshold": config.get('ensemble.score_threshold', 0.1)
            }
        }, f, indent=2)

    # Create summary plots
    create_metric_plots(results, output_dir, prefix="ensemble_")

    logger.info(f"Ensemble evaluation results: Precision={results['precision']:.4f}, "
                f"Recall={results['recall']:.4f}, F1={results['f1_score']:.4f}")
    logger.info(f"Results saved to {result_file}")

    return {
        "models": [str(m) for m in ensemble_detector.models],
        "metrics": results,
        "output_dir": str(output_dir),
        "results_file": str(result_file)
    }


def compare_models(config: Config, args) -> Dict[str, Any]:
    """Compare multiple models on the same test dataset."""
    logger.info("Comparing multiple models")

    if not args.model_paths:
        raise ValueError("Model paths are required for comparison")

    if args.model_names and len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model names must match number of models")

    model_names = args.model_names if args.model_names else [
        f"Model {i+1}" for i in range(len(args.model_paths))]

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Evaluate each model
    all_results = []

    for i, (model_path, model_name) in enumerate(zip(args.model_paths, model_names)):
        logger.info(f"Evaluating {model_name}: {model_path}")

        # Create subdirectory for this model
        model_dir = output_dir / f"model_{i+1}"
        model_dir.mkdir(exist_ok=True, parents=True)

        # Modify args for this model
        model_args = argparse.Namespace(**vars(args))
        model_args.model_path = model_path
        model_args.output_dir = str(model_dir)

        # Evaluate
        if args.dataset_yaml:
            result = evaluate_with_yolo_val(config, model_args, model_path)
        else:
            result = evaluate_on_test_dir(config, model_args, model_path)

        # Store results with model name
        result["name"] = model_name
        all_results.append(result)

    # Compare results and create comparison plots
    comparison_results = create_comparison_report(all_results, output_dir)

    logger.info(f"Model comparison complete. Results saved to {output_dir}")

    return {
        "models": args.model_paths,
        "names": model_names,
        "comparison": comparison_results,
        "output_dir": str(output_dir)
    }


def load_yolo_labels(label_file: Path, image_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load YOLO format labels and convert to absolute coordinates."""
    import cv2

    # Get image dimensions
    img = cv2.imread(str(image_file))
    if img is None:
        raise ValueError(f"Could not read image: {image_file}")

    img_height, img_width = img.shape[:2]

    boxes = []
    classes = []

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                boxes.append([x_center, y_center, width, height])
                classes.append(class_id)

    return np.array(boxes) if boxes else np.empty((0, 4)), np.array(classes) if classes else np.empty(0)


def match_predictions_to_ground_truth(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[int, int, int, List]:
    """Match predictions to ground truth boxes and calculate metrics."""
    # If no predictions or no ground truth, return early
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 0, 0, 0, []

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), []

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, []

    # Calculate IoU for every pred-gt pair
    matches = []
    matched_gt_indices = set()

    # For each prediction, find the best matching ground truth
    for pred_idx, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
        best_iou = -1
        best_gt_idx = -1

        for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
            # Skip if class doesn't match and we care about class
            if pred_class != gt_class:
                continue

            # Skip if this ground truth is already matched
            if gt_idx in matched_gt_indices:
                continue

            # Calculate IoU
            iou = calculate_iou(pred_box, gt_box)

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx

        # If we found a match
        if best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx, best_iou))
            matched_gt_indices.add(best_gt_idx)

    # Count metrics
    true_positives = len(matches)
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - true_positives

    return true_positives, false_positives, false_negatives, matches


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU between two boxes in [x_center, y_center, width, height] format."""
    # Convert to [x1, y1, x2, y2] format
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    # Calculate intersection area
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection_area = (x2 - x1) * (y2 - y1)

    # Calculate union area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou


def save_detection_visualization(
    image_file: Path,
    pred_boxes: List,
    gt_boxes: List,
    matches: List,
    pred_classes: List,
    gt_classes: List,
    confidences: List,
    output_path: Path
) -> None:
    """Save visualization of predictions and ground truth boxes."""
    import cv2

    # Read image
    img = cv2.imread(str(image_file))
    if img is None:
        logger.warning(f"Could not read image: {image_file}")
        return

    # Convert to RGB (for visualization)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a copy for drawing
    viz_img = img.copy()

    # Draw ground truth boxes (green)
    for i, (box, cls) in enumerate(zip(gt_boxes, gt_classes)):
        x_center, y_center, width, height = map(int, box)
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(viz_img, f"GT: {int(cls)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Initialize matched prediction indices
    matched_pred_indices = {match[0] for match in matches}

    # Draw predicted boxes (blue for matched, red for unmatched)
    for i, (box, cls, conf) in enumerate(zip(pred_boxes, pred_classes, confidences)):
        x_center, y_center, width, height = map(int, box)
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        color = (0, 0, 255) if i not in matched_pred_indices else (255, 0, 0)
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(viz_img, f"Pred: {int(cls)} ({conf:.2f})", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Convert back to BGR for saving
    viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)

    # Save image
    cv2.imwrite(str(output_path), viz_img)


def create_metric_plots(metrics: Dict[str, float], output_dir: Path, prefix: str = "") -> Dict[str, str]:
    """Create plots for performance metrics."""
    output_files = {}

    # Precision, recall, F1 bar chart
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ["precision", "recall", "f1_score"]
    values = [metrics[m] for m in metrics_to_plot]

    plt.bar(metrics_to_plot, values, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.ylim(0, 1.0)
    plt.title("Detection Performance Metrics")
    plt.ylabel("Score")

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')

    output_file = output_dir / f"{prefix}performance_metrics.png"
    plt.savefig(output_file)
    plt.close()

    output_files["performance"] = str(output_file)

    # TP, FP, FN donut chart
    if all(key in metrics for key in ["true_positives", "false_positives", "false_negatives"]):
        plt.figure(figsize=(8, 8))

        labels = ["True Positives", "False Positives", "False Negatives"]
        values = [metrics["true_positives"],
                  metrics["false_positives"], metrics["false_negatives"]]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']

        plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, wedgeprops=dict(width=0.4))
        plt.axis('equal')
        plt.title("Detection Results Breakdown")

        output_file = output_dir / f"{prefix}detection_breakdown.png"
        plt.savefig(output_file)
        plt.close()

        output_files["breakdown"] = str(output_file)

    return output_files


def create_comparison_report(results: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    """Create comparison report and visualizations for multiple models."""
    # Extract metrics for comparison
    models = []
    metrics = ["precision", "recall", "f1_score"]

    if "mAP50" in results[0]["metrics"]:
        metrics.extend(["mAP50", "mAP50-95"])

    metrics_data = {metric: [] for metric in metrics}

    for result in results:
        models.append(result["name"])
        for metric in metrics:
            value = result["metrics"].get(metric, 0.0)
            metrics_data[metric].append(value)

    # Create DataFrame for comparison
    df = pd.DataFrame({
        "Model": models,
        **{metric: metrics_data[metric] for metric in metrics}
    })

    # Save CSV report
    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)

    # Create comparison plot
    plt.figure(figsize=(12, 8))

    x = np.arange(len(models))
    width = 0.2
    offsets = np.linspace(-width, width, len(metrics))

    for i, metric in enumerate(metrics):
        plt.bar(x + offsets[i], metrics_data[metric],
                width, label=metric.capitalize())

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / "model_comparison.png"
    plt.savefig(plot_path)
    plt.close()

    # Create radar plot for multi-dimensional comparison
    plt.figure(figsize=(10, 10))

    # Number of metrics (dimensions)
    N = len(metrics)

    # Create angle for each metric
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Initialize the plot
    ax = plt.subplot(111, polar=True)

    # Draw one axis per metric and add labels
    plt.xticks(angles[:-1], [m.capitalize()
               for m in metrics], color='grey', size=10)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"],
               color="grey", size=8)
    plt.ylim(0, 1)

    # Plot each model
    for i, model in enumerate(models):
        values = [metrics_data[metric][i] for metric in metrics]
        values += values[:1]  # Close the loop

        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    radar_path = output_dir / "radar_comparison.png"
    plt.savefig(radar_path)
    plt.close()

    return {
        "models": models,
        "metrics": metrics,
        "data": df.to_dict(orient="records"),
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "radar_path": str(radar_path)
    }


def main():
    """Main entry point for the evaluation script."""
    args = parse_args()

    # Set up logging
    setup_logging(level=logging.INFO)

    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = Config(config_dict)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Run the selected mode
    try:
        if args.mode == "single":
            evaluate_single_model(config, args)
        elif args.mode == "ensemble":
            evaluate_ensemble(config, args)
        elif args.mode == "compare":
            compare_models(config, args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)

        logger.info(
            f"{args.mode.capitalize()} evaluation completed successfully")

    except Exception as e:
        logger.error(f"Error in {args.mode} evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
