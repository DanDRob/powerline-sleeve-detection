import os
import argparse
import asyncio
import json
import sys
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .system.config import Config
from .system.logging import setup_logging
from .processing.batch_processor import BatchProcessor
from .training.trainer import SleeveModelTrainer
from .training.data_manager import DataManager
from .training.augmentation import AugmentationPipeline
from .training.auto_labeling import AutoLabeler
from .training.hyperparameter_tuning import HyperparameterTuner
from .training.monitor import TrainingMonitor
from .training.ensemble import EnsembleDetector
from .detection.ensemble_integration import EnsembleIntegration
from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv()


def load_config(config_path: str) -> Config:
    """Load configuration from file."""
    try:
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

        config = Config.from_yaml(config_path)
        validation_errors = config.validate()

        if validation_errors:
            print("Config validation errors:")
            for error in validation_errors:
                print(f"- {error}")
            sys.exit(1)

        return config

    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def setup_environment(config: Config):
    """Set up the environment for processing."""
    # Create output directories
    config.create_output_dirs()

    # Configure logging
    log_dir = os.path.join(config.system.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "powerline_detector.log")

    log_level = "DEBUG" if config.system.debug else "INFO"
    setup_logging(log_level, log_file)

    # Set API key from environment variable if not in config
    if not config.api.key and "GOOGLE_API_KEY" in os.environ:
        config.api.key = os.environ["GOOGLE_API_KEY"]


async def process_single_route(config: Config, route_id: str, start_location: str, end_location: str):
    """Process a single route."""
    try:
        # Create batch processor
        processor = BatchProcessor(config)

        # Explicitly initialize detector and check for ensemble
        if config.get('ensemble.enabled', False):
            logging.info("Ensemble detection enabled for this route")

        # Process route
        result = await processor.process_route(
            route_id=route_id,
            start_location=start_location,
            end_location=end_location
        )

        logging.info(f"Route {route_id} processed: {len(result.get('processed_images', []))} images, "
                     f"{result.get('metadata', {}).get('total_detections', 0)} detections")

        return result
    except Exception as e:
        logging.error(f"Error processing route {route_id}: {e}")
        return {"success": False, "error": str(e)}


async def process_route_batch(config: Config, csv_file: str, parallel: bool = True,
                              max_concurrent: int = 2, subset: int = None):
    """Process a batch of routes from a CSV file."""
    try:
        # Create batch processor with ensemble if enabled
        processor = BatchProcessor(config)

        if config.get('ensemble.enabled', False):
            logging.info("Ensemble detection enabled for batch processing")

        # Process batch
        result = await processor.process_batch_from_csv(
            csv_file=csv_file,
            parallel=parallel,
            max_concurrent=max_concurrent,
            subset=subset
        )

        logging.info(f"Batch processed: {result.get('metadata', {}).get('total_routes', 0)} routes, "
                     f"{result.get('metadata', {}).get('total_images', 0)} images, "
                     f"{result.get('metadata', {}).get('total_detections', 0)} detections")

        return result
    except Exception as e:
        logging.error(f"Error processing batch: {e}")
        return {"success": False, "error": str(e)}


def prepare_dataset(config: Config, raw_data_path: str, splits: List[float] = None,
                    single_class: bool = True, class_names: List[str] = None):
    """Prepare a dataset for training from raw data."""
    print("Preparing dataset for training...")

    # Create data manager
    data_manager = DataManager(config)

    # Parse splits if provided
    dataset_splits = splits or [0.7, 0.15, 0.15]
    if len(dataset_splits) != 3:
        print("Error: Splits must contain exactly 3 values (train, val, test)")
        return

    if sum(dataset_splits) != 1.0:
        print(
            f"Warning: Splits do not sum to 1.0 (sum: {sum(dataset_splits)})")

    # Prepare dataset
    try:
        dataset_yaml = data_manager.prepare_dataset(
            raw_data_path=raw_data_path,
            splits=tuple(dataset_splits),
            single_class=single_class,
            class_names=class_names.split(',') if class_names else None
        )

        print(f"Dataset prepared successfully!")
        print(f"Dataset configuration saved to: {dataset_yaml}")

    except Exception as e:
        print(f"Error preparing dataset: {e}")


def create_empty_dataset(config: Config, class_names: str = None):
    """Create an empty dataset structure for manual data collection."""
    print("Creating empty dataset structure...")

    # Create data manager
    data_manager = DataManager(config)

    # Create empty dataset
    try:
        dataset_dir = data_manager.create_empty_dataset(
            class_names=class_names.split(',') if class_names else None
        )

        print(f"Empty dataset created at: {dataset_dir}")
        print("Please place images in the 'raw/images' directory and labels in the 'raw/labels' directory.")

    except Exception as e:
        print(f"Error creating empty dataset: {e}")


def augment_dataset(config: Config, dataset_path: str, output_path: str = None,
                    multiplier: int = 3, severity: str = 'medium'):
    """Augment a dataset with synthetic variations."""
    print(f"Augmenting dataset with severity {severity}...")

    # Create augmentation pipeline
    augmenter = AugmentationPipeline(severity=severity)

    # Augment dataset
    try:
        # For train split
        train_path = os.path.join(dataset_path, 'train')
        if os.path.exists(train_path):
            print(f"Augmenting training split...")
            train_output = os.path.join(
                output_path, 'train') if output_path else None
            orig_count, aug_count = augmenter.augment_dataset(
                dataset_path=train_path,
                output_path=train_output,
                multiplier=multiplier
            )
            print(
                f"Training split: Added {aug_count} augmented images to {orig_count} original images")

        print(f"Dataset augmentation completed successfully!")

    except Exception as e:
        print(f"Error augmenting dataset: {e}")


def train_model(config: Config, base_model: str, dataset_yaml: str, epochs: int = None,
                batch_size: int = None, image_size: int = None, learning_rate: float = None):
    """Train a powerline sleeve detection model."""
    print(f"Starting model training...")

    # Create trainer
    trainer = SleeveModelTrainer(config)

    # Train model
    try:
        # Initialize model
        print(f"Initializing model from {base_model}")
        trainer.initialize_model(base_model)

        # Train model
        print(f"Training model with dataset: {dataset_yaml}")
        results = trainer.train(
            dataset_yaml=dataset_yaml,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            learning_rate=learning_rate
        )

        # Print results
        best_model_path = results.get('best_model_path')
        print(f"Training completed successfully!")
        print(f"Best model saved to: {best_model_path}")

        # Validate model
        print("Validating model...")
        val_results = trainer.validate()
        print(
            f"Validation results: mAP@0.5 = {val_results.box.map50:.4f}, mAP@0.5:0.95 = {val_results.box.map:.4f}")

    except Exception as e:
        print(f"Error training model: {e}")


def tune_hyperparameters(config: Config, dataset_yaml: str, base_model: str, n_trials: int = 20,
                         timeout: int = None):
    """Perform hyperparameter tuning for model training."""
    print(f"Starting hyperparameter tuning...")

    # Update config with dataset path
    config.set('training.dataset_yaml', dataset_yaml)
    config.set('training.base_model', base_model)

    # Create hyperparameter tuner
    tuner = HyperparameterTuner(config)

    # Run hyperparameter tuning
    try:
        print(f"Running {n_trials} hyperparameter tuning trials...")
        results = tuner.run_study(
            n_trials=n_trials,
            timeout=timeout,
            study_name="sleeve_detection_tuning"
        )

        # Print results
        print(f"Hyperparameter tuning completed successfully!")
        print(f"Best hyperparameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        print(f"Best score (mAP): {results['best_score']:.4f}")

        # Apply best parameters
        print("Applying best parameters to config...")
        tuner.apply_best_params()

    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")


def auto_label_images(config: Config, model_path: str, images_dir: str, output_dir: str,
                      confidence: float = 0.5, class_mapping: str = None):
    """Automatically label images using a trained model."""
    print(f"Starting auto-labeling with model: {model_path}")

    # Create auto labeler
    auto_labeler = AutoLabeler(config)

    # Parse class mapping if provided
    mapping = None
    if class_mapping:
        try:
            mapping = {}
            for pair in class_mapping.split(','):
                src, dst = map(int, pair.split(':'))
                mapping[src] = dst
            print(f"Using class mapping: {mapping}")
        except Exception as e:
            print(f"Error parsing class mapping: {e}")
            return

    # Auto-label images
    try:
        # Load model
        auto_labeler.load_model(model_path)

        # Label images
        total, labeled = auto_labeler.label_images(
            images_dir=images_dir,
            output_dir=output_dir,
            confidence_threshold=confidence,
            class_mapping=mapping
        )

        print(f"Auto-labeling completed: {labeled}/{total} images labeled")
        print(f"Labels saved to: {os.path.join(output_dir, 'labels')}")

        # Visualize labels
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        images_output_dir = os.path.join(output_dir, 'images')
        labels_output_dir = os.path.join(output_dir, 'labels')

        if os.path.exists(images_output_dir) and os.path.exists(labels_output_dir):
            print("Generating visualizations of labeled images...")
            visualized = auto_labeler.visualize_labels(
                images_dir=images_output_dir,
                labels_dir=labels_output_dir,
                output_dir=vis_dir
            )
            print(f"Generated {visualized} visualizations in {vis_dir}")

    except Exception as e:
        print(f"Error during auto-labeling: {e}")


def semi_supervised_labeling(config: Config, model_path: str, images_dir: str, output_dir: str,
                             auto_threshold: float = 0.8, review_threshold: float = 0.5):
    """Perform semi-supervised labeling with human review for uncertain predictions."""
    print(f"Starting semi-supervised labeling with model: {model_path}")

    # Create auto labeler
    auto_labeler = AutoLabeler(config)

    # Perform semi-supervised labeling
    try:
        # Load model
        auto_labeler.load_model(model_path)

        # Run semi-supervised labeling
        results = auto_labeler.semi_supervised_labeling(
            images_dir=images_dir,
            output_dir=output_dir,
            confidence_threshold=auto_threshold,
            review_threshold=review_threshold
        )

        print(f"Semi-supervised labeling completed:")
        print(
            f"- Auto-labeled: {results['auto_labeled']}/{results['total']} images")
        print(
            f"- Flagged for review: {results['flagged_for_review']}/{results['total']} images")
        print(f"Images and labels saved to: {output_dir}")
        print(
            f"Images requiring review saved to: {os.path.join(output_dir, 'review')}")

    except Exception as e:
        print(f"Error during semi-supervised labeling: {e}")


def create_ensemble(config: Config, model_paths: str, model_weights: str = None,
                    output_config: str = None):
    """Create an ensemble of models for improved detection performance."""
    print("Creating model ensemble...")

    # Parse model paths and weights
    paths = model_paths.split(',')
    weights = [1.0] * len(paths)

    if model_weights:
        try:
            weights = list(map(float, model_weights.split(',')))
            if len(weights) != len(paths):
                print(
                    f"Error: Number of weights ({len(weights)}) must match number of models ({len(paths)})")
                return
        except Exception as e:
            print(f"Error parsing model weights: {e}")
            return

    # Create ensemble configuration
    ensemble_config = []
    for path, weight in zip(paths, weights):
        ensemble_config.append({
            'path': path,
            'weight': weight
        })

    # Update config
    config.set('ensemble.models', ensemble_config)
    config.set('ensemble.method', 'weighted_boxes_fusion')
    config.set('ensemble.iou_threshold', 0.5)
    config.set('ensemble.score_threshold', 0.1)

    # Save to dedicated config file if requested
    if output_config:
        try:
            with open(output_config, 'w') as f:
                yaml_data = {
                    'ensemble': {
                        'models': ensemble_config,
                        'method': 'weighted_boxes_fusion',
                        'iou_threshold': 0.5,
                        'score_threshold': 0.1
                    }
                }
                # Use config's yaml dumper
                yaml_str = config.to_yaml_str(yaml_data)
                f.write(yaml_str)
            print(f"Ensemble configuration saved to: {output_config}")
        except Exception as e:
            print(f"Error saving ensemble configuration: {e}")

    # Test ensemble
    print("Testing ensemble detector...")
    try:
        ensemble = EnsembleDetector(config)
        ensemble.load_models_from_config()
        print(
            f"Successfully loaded {len(ensemble.models)} models into ensemble")

    except Exception as e:
        print(f"Error creating ensemble: {e}")


def evaluate_ensemble(config: Config, test_dir: str, ground_truth_dir: str):
    """Evaluate ensemble performance on a test dataset."""
    print("Evaluating ensemble performance...")

    # Create ensemble detector
    ensemble = EnsembleDetector(config)

    try:
        # Load models
        ensemble.load_models_from_config()
        print(f"Loaded {len(ensemble.models)} models into ensemble")

        # Evaluate ensemble
        metrics = ensemble.evaluate_ensemble(
            test_dir=test_dir,
            ground_truth_dir=ground_truth_dir
        )

        # Print metrics
        print("\nEnsemble Evaluation Results:")
        print(f"- Precision: {metrics['precision']:.4f}")
        print(f"- Recall: {metrics['recall']:.4f}")
        print(f"- F1 Score: {metrics['f1_score']:.4f}")
        print(f"- Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"- True Positives: {metrics['true_positives']}")
        print(f"- False Positives: {metrics['false_positives']}")
        print(f"- False Negatives: {metrics['false_negatives']}")

    except Exception as e:
        print(f"Error evaluating ensemble: {e}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Powerline Sleeve Detection System")

    # Config options
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")

    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single route processing
    route_parser = subparsers.add_parser(
        "route", help="Process a single route")
    route_parser.add_argument("--id", type=str, required=True,
                              help="Route identifier")
    route_parser.add_argument("--start", type=str, required=True,
                              help="Start location (address or lat,lng)")
    route_parser.add_argument("--end", type=str, required=True,
                              help="End location (address or lat,lng)")

    # Batch processing
    batch_parser = subparsers.add_parser(
        "batch", help="Process multiple routes from CSV")
    batch_parser.add_argument("--csv", type=str, required=True,
                              help="CSV file with route definitions")
    batch_parser.add_argument("--parallel", action="store_true",
                              help="Process routes in parallel")
    batch_parser.add_argument("--max-concurrent", type=int, default=2,
                              help="Maximum number of concurrent routes when using parallel processing")
    batch_parser.add_argument("--subset", type=int,
                              help="Create and process a validation subset with N routes")

    # Dataset preparation
    dataset_parser = subparsers.add_parser(
        "prepare-dataset", help="Prepare a dataset for training")
    dataset_parser.add_argument("--data", type=str, required=True,
                                help="Path to raw data directory")
    dataset_parser.add_argument("--splits", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                                help="Train/val/test splits (must sum to 1.0)")
    dataset_parser.add_argument("--single-class", action="store_true", default=True,
                                help="Convert all classes to a single 'sleeve' class")
    dataset_parser.add_argument("--class-names", type=str,
                                help="Comma-separated list of class names (if not single-class)")

    # Create empty dataset
    empty_dataset_parser = subparsers.add_parser(
        "create-empty-dataset", help="Create an empty dataset structure for manual data collection")
    empty_dataset_parser.add_argument("--class-names", type=str,
                                      help="Comma-separated list of class names (default: 'sleeve')")

    # Dataset augmentation
    augment_parser = subparsers.add_parser(
        "augment-dataset", help="Augment a dataset with synthetic variations")
    augment_parser.add_argument("--dataset", type=str, required=True,
                                help="Path to dataset directory")
    augment_parser.add_argument("--output", type=str,
                                help="Path to output directory (if different from input)")
    augment_parser.add_argument("--multiplier", type=int, default=3,
                                help="Number of augmented versions per original image")
    augment_parser.add_argument("--severity", type=str, default="medium",
                                choices=["light", "medium", "heavy"],
                                help="Augmentation severity")

    # Model training
    train_parser = subparsers.add_parser(
        "train", help="Train a powerline sleeve detection model")
    train_parser.add_argument("--base-model", type=str, required=True,
                              help="Base model to fine-tune (e.g., 'yolov8n.pt')")
    train_parser.add_argument("--dataset", type=str, required=True,
                              help="Path to dataset.yaml file")
    train_parser.add_argument("--epochs", type=int,
                              help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int,
                              help="Batch size for training")
    train_parser.add_argument("--image-size", type=int,
                              help="Image size for training")
    train_parser.add_argument("--learning-rate", type=float,
                              help="Learning rate for training")

    # Hyperparameter tuning
    tune_parser = subparsers.add_parser(
        "tune", help="Perform hyperparameter tuning for model training")
    tune_parser.add_argument("--dataset", type=str, required=True,
                             help="Path to dataset.yaml file")
    tune_parser.add_argument("--base-model", type=str, required=True,
                             help="Base model to fine-tune (e.g., 'yolov8n.pt')")
    tune_parser.add_argument("--trials", type=int, default=20,
                             help="Number of trials to run")
    tune_parser.add_argument("--timeout", type=int,
                             help="Timeout in seconds (None for no timeout)")

    # Auto-labeling
    autolabel_parser = subparsers.add_parser(
        "autolabel", help="Automatically label images using a trained model")
    autolabel_parser.add_argument("--model", type=str, required=True,
                                  help="Path to trained model file")
    autolabel_parser.add_argument("--images", type=str, required=True,
                                  help="Path to directory with images to label")
    autolabel_parser.add_argument("--output", type=str, required=True,
                                  help="Path to output directory for labeled data")
    autolabel_parser.add_argument("--confidence", type=float, default=0.5,
                                  help="Confidence threshold for detections")
    autolabel_parser.add_argument("--class-mapping", type=str,
                                  help="Mapping of model classes to output classes (format: '0:1,2:3')")

    # Semi-supervised labeling
    semilabel_parser = subparsers.add_parser(
        "semilabel", help="Perform semi-supervised labeling with human review")
    semilabel_parser.add_argument("--model", type=str, required=True,
                                  help="Path to trained model file")
    semilabel_parser.add_argument("--images", type=str, required=True,
                                  help="Path to directory with images to label")
    semilabel_parser.add_argument("--output", type=str, required=True,
                                  help="Path to output directory for labeled data")
    semilabel_parser.add_argument("--auto-threshold", type=float, default=0.8,
                                  help="Confidence threshold for automatic labeling")
    semilabel_parser.add_argument("--review-threshold", type=float, default=0.5,
                                  help="Confidence threshold for flagging for review")

    # Ensemble creation
    ensemble_parser = subparsers.add_parser(
        "create-ensemble", help="Create an ensemble of models for improved detection")
    ensemble_parser.add_argument("--models", type=str, required=True,
                                 help="Comma-separated list of model paths")
    ensemble_parser.add_argument("--weights", type=str,
                                 help="Comma-separated list of model weights")
    ensemble_parser.add_argument("--output-config", type=str,
                                 help="Path to save ensemble configuration")

    # Ensemble evaluation
    eval_ensemble_parser = subparsers.add_parser(
        "evaluate-ensemble", help="Evaluate ensemble performance on a test dataset")
    eval_ensemble_parser.add_argument("--test-dir", type=str, required=True,
                                      help="Path to test images directory")
    eval_ensemble_parser.add_argument("--ground-truth-dir", type=str, required=True,
                                      help="Path to ground truth labels directory")

    # Parse arguments
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override debug setting if specified
    if args.debug:
        config.system.debug = True

    # Set up environment
    setup_environment(config)

    # Process command
    if args.command == "route":
        asyncio.run(process_single_route(
            config, args.id, args.start, args.end))
    elif args.command == "batch":
        asyncio.run(process_route_batch(config, args.csv,
                    args.parallel, args.max_concurrent, args.subset))
    elif args.command == "prepare-dataset":
        prepare_dataset(config, args.data, args.splits,
                        args.single_class, args.class_names)
    elif args.command == "create-empty-dataset":
        create_empty_dataset(config, args.class_names)
    elif args.command == "augment-dataset":
        augment_dataset(config, args.dataset, args.output,
                        args.multiplier, args.severity)
    elif args.command == "train":
        train_model(config, args.base_model, args.dataset, args.epochs,
                    args.batch_size, args.image_size, args.learning_rate)
    elif args.command == "tune":
        tune_hyperparameters(config, args.dataset,
                             args.base_model, args.trials, args.timeout)
    elif args.command == "autolabel":
        auto_label_images(config, args.model, args.images, args.output,
                          args.confidence, args.class_mapping)
    elif args.command == "semilabel":
        semi_supervised_labeling(config, args.model, args.images, args.output,
                                 args.auto_threshold, args.review_threshold)
    elif args.command == "create-ensemble":
        create_ensemble(config, args.models, args.weights, args.output_config)
    elif args.command == "evaluate-ensemble":
        evaluate_ensemble(config, args.test_dir, args.ground_truth_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
