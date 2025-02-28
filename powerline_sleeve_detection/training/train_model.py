#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import yaml
import torch
from tqdm import tqdm

from ..system.config import Config
from ..system.logging import get_logger, setup_logging
from .trainer import SleeveModelTrainer
from .data_manager import DataManager
from .augmentation import AugmentationPipeline
from .hyperparameter_tuning import HyperparameterTuner
from .auto_labeling import AutoLabeler
from .ensemble import EnsembleDetector
from .monitor import TrainingMonitor

logger = get_logger("train_model")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train powerline sleeve detection models with YOLOv8")

    # Main command options
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["prepare", "augment", "train", "tune",
                                 "auto-label", "create-ensemble", "full-workflow"],
                        help="Operation mode")

    # Dataset preparation options
    parser.add_argument("--data", type=str, help="Path to raw data directory")
    parser.add_argument("--splits", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help="Train/val/test splits (must sum to 1.0)")
    parser.add_argument("--single-class", action="store_true", default=True,
                        help="Convert all classes to single 'sleeve' class")
    parser.add_argument("--class-names", type=str, nargs="+",
                        help="Class names (if not single-class)")

    # Augmentation options
    parser.add_argument("--severity", type=str, default="medium",
                        choices=["light", "medium", "heavy"],
                        help="Augmentation severity")
    parser.add_argument("--multiplier", type=int, default=3,
                        help="Augmentation multiplier")

    # Training options
    parser.add_argument("--dataset-yaml", type=str,
                        help="Path to dataset.yaml file")
    parser.add_argument("--base-model", type=str, default="yolov8s.pt",
                        help="Base YOLOv8 model to fine-tune")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--image-size", type=int, help="Input image size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    # Hyperparameter tuning options
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of hyperparameter tuning trials")
    parser.add_argument("--timeout", type=int,
                        help="Timeout for hyperparameter tuning in seconds")

    # Auto-labeling options
    parser.add_argument("--model-path", type=str,
                        help="Path to trained model for auto-labeling")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for auto-labeling")

    # Ensemble options
    parser.add_argument("--model-paths", type=str, nargs="+",
                        help="Paths to models for ensemble")
    parser.add_argument("--weights", type=float, nargs="+",
                        help="Weights for ensemble models")

    return parser.parse_args()


def prepare_dataset(config: Config, args) -> str:
    """Prepare dataset for training."""
    logger.info("Preparing dataset for training")
    data_manager = DataManager(config)

    if not args.data:
        raise ValueError("Data path is required for dataset preparation")

    dataset_yaml = data_manager.prepare_dataset(
        raw_data_path=args.data,
        splits=tuple(args.splits),
        single_class=args.single_class,
        class_names=args.class_names
    )

    logger.info(f"Dataset prepared successfully. YAML config: {dataset_yaml}")
    return dataset_yaml


def augment_dataset(config: Config, args) -> Dict[str, Any]:
    """Augment dataset with synthetic variations."""
    logger.info(f"Augmenting dataset with severity: {args.severity}")

    dataset_path = args.data
    if not dataset_path and config.get('training.dataset_dir'):
        dataset_path = config.get('training.dataset_dir')

    if not dataset_path:
        raise ValueError("Dataset path is required for augmentation")

    augmentation = AugmentationPipeline(severity=args.severity)
    original, augmented = augmentation.augment_dataset(
        dataset_path=dataset_path,
        multiplier=args.multiplier
    )

    logger.info(f"Dataset augmentation complete. Original images: {original}, "
                f"Augmented images: {augmented}")
    return {"original": original, "augmented": augmented}


def train_model(config: Config, args) -> Dict[str, Any]:
    """Train a YOLOv8 model for sleeve detection."""
    logger.info("Training sleeve detection model")

    # Initialize trainer
    trainer = SleeveModelTrainer(config)

    # Set dataset yaml path
    dataset_yaml = args.dataset_yaml
    if not dataset_yaml:
        dataset_yaml = config.get('training.dataset_yaml')
        if not dataset_yaml:
            raise ValueError("Dataset YAML path is required for training")

    # Set base model
    base_model = args.base_model or config.get(
        'training.base_model', 'yolov8s.pt')

    # Initialize model
    trainer.initialize_model(base_model)

    # Train model
    results = trainer.train(
        dataset_yaml=dataset_yaml,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate
    )

    logger.info(f"Training complete. Best model: {results['best_model_path']}")

    # Load and visualize training metrics
    output_dir = args.output_dir or config.get(
        'training.output_dir', 'runs/train')
    monitor = TrainingMonitor(output_dir)

    try:
        results_df = monitor.load_results(
            f"{Path(results['best_model_path']).parent}/results.csv")
        plot_paths = monitor.plot_metrics(results_df)
        logger.info(
            f"Training plots saved to: {', '.join(plot_paths.values())}")
    except Exception as e:
        logger.warning(f"Could not create training visualizations: {e}")

    return results


def tune_hyperparameters(config: Config, args) -> Dict[str, Any]:
    """Run hyperparameter tuning for model optimization."""
    logger.info("Starting hyperparameter tuning")

    # Initialize tuner
    tuner = HyperparameterTuner(config)

    # Set dataset yaml path
    dataset_yaml = args.dataset_yaml
    if not dataset_yaml:
        dataset_yaml = config.get('training.dataset_yaml')
        if not dataset_yaml:
            raise ValueError("Dataset YAML path is required for tuning")

    # Set it in config
    config.set('training.dataset_yaml', dataset_yaml)

    # Set base model
    base_model = args.base_model or config.get(
        'training.base_model', 'yolov8s.pt')
    config.set('training.base_model', base_model)

    # Run tuning study
    results = tuner.run_study(
        n_trials=args.trials,
        timeout=args.timeout,
        study_name="sleeve_detection_tuning"
    )

    logger.info(
        f"Hyperparameter tuning complete. Best score: {results['best_score']}")

    # Apply best parameters and train final model
    tuner.apply_best_params()

    # Log best parameters
    for param, value in results['best_params'].items():
        logger.info(f"Best {param}: {value}")

    return results


def auto_label_images(config: Config, args) -> Dict[str, Any]:
    """Automatically label images using a trained model."""
    logger.info("Starting auto-labeling")

    model_path = args.model_path
    if not model_path:
        model_path = config.get('training.custom_model_path')
        if not model_path:
            model_path = config.get('detection.model_paths.custom_model')
            if not model_path:
                raise ValueError("Model path is required for auto-labeling")

    images_dir = args.data
    if not images_dir:
        raise ValueError("Images directory is required for auto-labeling")

    output_dir = args.output_dir or Path(images_dir).parent / "auto_labeled"

    # Initialize auto-labeler
    auto_labeler = AutoLabeler(config)
    auto_labeler.load_model(model_path)

    # Run auto-labeling
    labeled, total = auto_labeler.label_images(
        images_dir=images_dir,
        output_dir=output_dir,
        confidence_threshold=args.confidence
    )

    logger.info(f"Auto-labeling complete. Labeled {labeled}/{total} images.")

    # Create visualizations
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(exist_ok=True, parents=True)

    try:
        viz_count = auto_labeler.visualize_labels(
            images_dir=images_dir,
            labels_dir=output_dir,
            output_dir=viz_dir,
            class_names=config.get('training.class_names', ['sleeve'])
        )
        logger.info(f"Created {viz_count} visualizations in {viz_dir}")
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")

    return {"labeled": labeled, "total": total, "output_dir": str(output_dir)}


def create_ensemble(config: Config, args) -> Dict[str, Any]:
    """Create an ensemble of models for improved detection."""
    logger.info("Creating model ensemble")

    model_paths = args.model_paths
    if not model_paths:
        raise ValueError("Model paths are required for ensemble creation")

    weights = args.weights
    if weights and len(weights) != len(model_paths):
        raise ValueError("Number of weights must match number of models")

    if not weights:
        weights = [1.0] * len(model_paths)

    # Initialize ensemble detector
    ensemble = EnsembleDetector(config)

    # Add models to ensemble
    for i, model_path in enumerate(model_paths):
        ensemble.add_model(model_path, weights[i])

    # Save ensemble configuration to config
    ensemble_config = {
        "enabled": True,
        "method": config.get('ensemble.method', 'weighted_boxes_fusion'),
        "iou_threshold": config.get('ensemble.iou_threshold', 0.5),
        "score_threshold": config.get('ensemble.score_threshold', 0.1),
        "models": []
    }

    for i, model_path in enumerate(model_paths):
        ensemble_config["models"].append({
            "path": model_path,
            "weight": weights[i]
        })

    config.set('ensemble', ensemble_config)
    config.save()

    logger.info(f"Ensemble created with {len(model_paths)} models")

    return {
        "num_models": len(model_paths),
        "models": model_paths,
        "weights": weights
    }


def run_full_workflow(config: Config, args) -> Dict[str, Any]:
    """Run the complete training workflow from data preparation to ensemble creation."""
    logger.info("Starting full training workflow")

    results = {}

    # 1. Prepare dataset
    try:
        logger.info("Step 1: Preparing dataset")
        dataset_yaml = prepare_dataset(config, args)
        args.dataset_yaml = dataset_yaml
        results["dataset_yaml"] = dataset_yaml
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return {"success": False, "error": str(e), "stage": "dataset_preparation"}

    # 2. Augment dataset
    try:
        logger.info("Step 2: Augmenting dataset")
        aug_results = augment_dataset(config, args)
        results["augmentation"] = aug_results
    except Exception as e:
        logger.error(f"Dataset augmentation failed: {e}")
        return {"success": False, "error": str(e), "stage": "augmentation", **results}

    # 3. Tune hyperparameters
    try:
        logger.info("Step 3: Tuning hyperparameters")
        tune_results = tune_hyperparameters(config, args)
        results["hyperparameter_tuning"] = tune_results
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        return {"success": False, "error": str(e), "stage": "hyperparameter_tuning", **results}

    # 4. Train model with best hyperparameters
    try:
        logger.info("Step 4: Training model with best hyperparameters")
        train_results = train_model(config, args)
        results["training"] = train_results
        args.model_path = train_results["best_model_path"]
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return {"success": False, "error": str(e), "stage": "training", **results}

    # 5. Create ensemble (optional)
    if args.model_paths:
        try:
            logger.info("Step 5: Creating model ensemble")
            if not args.model_paths:
                # If model paths not explicitly provided, use the best model from training
                # plus any default models specified in config
                args.model_paths = [args.model_path]
                default_model = config.get('detection.model_paths.yolov8s')
                if default_model and os.path.exists(default_model):
                    args.model_paths.append(default_model)

            ensemble_results = create_ensemble(config, args)
            results["ensemble"] = ensemble_results
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            return {"success": False, "error": str(e), "stage": "ensemble_creation", **results}

    logger.info("Full training workflow completed successfully")
    return {"success": True, **results}


def main():
    """Main entry point for the training script."""
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

    # Enable training mode in config
    config.set('training.enabled', True)
    config.save()

    # Run the selected mode
    try:
        if args.mode == "prepare":
            prepare_dataset(config, args)
        elif args.mode == "augment":
            augment_dataset(config, args)
        elif args.mode == "train":
            train_model(config, args)
        elif args.mode == "tune":
            tune_hyperparameters(config, args)
        elif args.mode == "auto-label":
            auto_label_images(config, args)
        elif args.mode == "create-ensemble":
            create_ensemble(config, args)
        elif args.mode == "full-workflow":
            run_full_workflow(config, args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)

        logger.info(
            f"{args.mode.capitalize()} operation completed successfully")

    except Exception as e:
        logger.error(f"Error in {args.mode} operation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
