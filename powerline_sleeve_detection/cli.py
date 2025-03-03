import os
import argparse
import asyncio
import json
import sys
import logging
import traceback
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
from .training.two_stage_detector import TwoStageDetector
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


def initialize_config(config_path="config.yaml"):
    """Initialize the configuration system."""
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            import yaml
            config_dict = yaml.safe_load(f)

        config = Config.from_dict(config_dict)
        return config
    except Exception as e:
        logging.error(f"Failed to initialize configuration: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Powerline Sleeve Detection Tool")

    parser.add_argument('--config', default='config.yaml',
                        help='Path to config file')

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # Process command
    process_parser = subparsers.add_parser(
        'process', help='Process routes and detect sleeves')
    process_parser.add_argument(
        '--csv', help='CSV file with routes to process')
    process_parser.add_argument(
        '--route-id', help='Specific route ID to process')
    process_parser.add_argument(
        '--start', help='Start location (if processing single route)')
    process_parser.add_argument(
        '--end', help='End location (if processing single route)')
    process_parser.add_argument('--parallel', action='store_true',
                                help='Process routes in parallel')
    process_parser.add_argument('--max-concurrent', type=int, default=2,
                                help='Maximum number of concurrent tasks when running in parallel')
    process_parser.add_argument('--subset', type=int,
                                help='Process only a subset of routes')
    process_parser.add_argument('--visualize', action='store_true',
                                help='Generate visualizations during processing')
    process_parser.add_argument('--export-format', choices=['kml', 'geojson', 'csv', 'all'],
                                default='all', help='Export format for results')
    process_parser.add_argument('--no-cleanup', action='store_true',
                                help='Keep temporary files after processing')

    # Plan command
    plan_parser = subparsers.add_parser(
        'plan', help='Plan routes for data acquisition')
    plan_parser.add_argument('--coverage', choices=['high', 'medium', 'low'],
                             default='medium', help='Desired coverage level')
    plan_parser.add_argument('--region', help='Region to plan routes for')
    plan_parser.add_argument(
        '--output', help='Output file for the planned routes')
    plan_parser.add_argument('--visualize', action='store_true',
                             help='Visualize the planned routes')

    # Train command
    train_parser = subparsers.add_parser(
        'train', help='Train and evaluate sleeve detection models')
    train_parser.add_argument('--mode', choices=['prepare', 'augment', 'train', 'evaluate', 'auto-label',
                                                 'tune', 'monitor', 'create-ensemble'],
                              help='Training mode')
    train_parser.add_argument('--data', help='Path to training data')
    train_parser.add_argument(
        '--model', help='Base model to use for training or evaluation')
    train_parser.add_argument('--dataset', help='Path to dataset YAML file')
    train_parser.add_argument('--epochs', type=int,
                              help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int,
                              help='Training batch size')
    train_parser.add_argument('--img-size', type=int,
                              help='Training image size')
    train_parser.add_argument(
        '--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument(
        '--weight-decay', type=float, help='Weight decay')
    train_parser.add_argument(
        '--device', help='Training device (cpu, cuda, 0, 1, etc.)')
    train_parser.add_argument('--workers', type=int,
                              default=8, help='Number of worker threads')
    train_parser.add_argument(
        '--cache', action='store_true', help='Cache images for faster training')
    train_parser.add_argument(
        '--single-class', action='store_true', help='Treat all objects as a single class')
    train_parser.add_argument(
        '--optimizer', choices=['SGD', 'Adam', 'AdamW'], help='Optimizer to use')
    train_parser.add_argument('--augment', choices=['light', 'medium', 'heavy'],
                              help='Augmentation severity')
    train_parser.add_argument('--confidence', type=float, default=0.25,
                              help='Confidence threshold for detections')
    train_parser.add_argument('--iou-threshold', type=float, default=0.45,
                              help='IoU threshold for NMS')
    train_parser.add_argument('--save-period', type=int, default=10,
                              help='Save checkpoint every x epochs')
    train_parser.add_argument('--output', help='Output directory for results')
    train_parser.add_argument(
        '--resume', action='store_true', help='Resume training from checkpoint')

    # Two-stage detection command (new)
    two_stage_parser = subparsers.add_parser(
        'two-stage', help='Two-stage detection for powerlines and sleeves')
    two_stage_parser.add_argument('--mode', choices=['setup', 'label-powerlines', 'augment-powerlines',
                                                     'train-powerlines', 'detect-powerlines',
                                                     'extract-regions', 'label-sleeves', 'augment-sleeves',
                                                     'train-sleeves', 'run-pipeline'],
                                  required=True, help='Operation mode')
    two_stage_parser.add_argument(
        '--images', help='Directory containing images to process')
    two_stage_parser.add_argument('--output', help='Output directory')
    two_stage_parser.add_argument('--base-dir', default='data/two_stage',
                                  help='Base directory for two-stage detection data')
    two_stage_parser.add_argument(
        '--powerline-model', help='Path to trained powerline model')
    two_stage_parser.add_argument(
        '--sleeve-model', help='Path to trained sleeve model')
    two_stage_parser.add_argument('--augmentations', type=int, default=5,
                                  help='Number of augmented images to generate per original')
    two_stage_parser.add_argument('--severity', choices=['light', 'medium', 'heavy'],
                                  default='medium', help='Augmentation severity')
    two_stage_parser.add_argument('--epochs', type=int, default=100,
                                  help='Number of training epochs')
    two_stage_parser.add_argument('--batch-size', type=int, default=16,
                                  help='Training batch size')
    two_stage_parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'],
                                  default='m', help='Model size (YOLOv5 size)')
    two_stage_parser.add_argument('--conf-threshold', type=float, default=0.25,
                                  help='Confidence threshold for detections')
    two_stage_parser.add_argument('--padding', type=float, default=0.1,
                                  help='Padding around powerline regions when extracting')
    two_stage_parser.add_argument('--hyperparameter-tuning', action='store_true',
                                  help='Perform hyperparameter tuning to find the optimal model')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        config = load_config(args.config)

        # Set up logging and environment
        log_level = "DEBUG" if hasattr(
            config.system, 'debug') and config.system.debug else "INFO"
        log_dir = os.path.join(config.system.output_dir, "logs") if hasattr(
            config.system, 'output_dir') else "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "powerline_detector.log")

        setup_logging(log_level, log_file)
        setup_environment(config)

        if args.command == 'process':
            process_command(args, config)
        elif args.command == 'plan':
            plan_command(args, config)
        elif args.command == 'train':
            train_command(args, config)
        elif args.command == 'two-stage':
            two_stage_command(args, config)
        else:
            print(f"Unknown command: {args.command}")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


def process_command(args, config):
    """Handle processing of routes and detection of sleeves."""
    logger = logging.getLogger("process")

    try:
        if args.route_id:
            # First try to find route in sample_routes.csv if it exists and no start/end provided
            if not args.start or not args.end:
                csv_path = "sample_routes.csv"
                if os.path.exists(csv_path):
                    try:
                        # Read the file directly line by line for maximum simplicity
                        with open(csv_path, 'r') as f:
                            print(
                                f"Looking for route_id: '{args.route_id}' in {csv_path}")
                            lines = f.readlines()
                            header = lines[0].strip().split(',')

                            # Find column indices
                            id_col = header.index('route_id')
                            start_col = header.index('start_location')
                            end_col = header.index('end_location')

                            for line in lines[1:]:  # Skip header
                                if not line.strip():  # Skip empty lines
                                    continue

                                # Simple CSV parsing
                                parts = []
                                in_quotes = False
                                current = ""
                                for char in line:
                                    if char == '"':
                                        in_quotes = not in_quotes
                                    elif char == ',' and not in_quotes:
                                        parts.append(current)
                                        current = ""
                                    else:
                                        current += char
                                parts.append(current)  # Add the last part

                                route_id = parts[id_col].strip()
                                print(f"Checking route: '{route_id}'")

                                if route_id == args.route_id:
                                    # Found the route
                                    start_location = parts[start_col].strip(
                                        '"')
                                    end_location = parts[end_col].strip('"')

                                    print(f"Found route {args.route_id}:")
                                    print(f"  Start: {start_location}")
                                    print(f"  End: {end_location}")
                                    logger.info(f"Found route {args.route_id}")
                                    logger.info(f"  Start: {start_location}")
                                    logger.info(f"  End: {end_location}")

                                    # Process the route
                                    result = asyncio.run(process_single_route(
                                        config=config,
                                        route_id=args.route_id,
                                        start_location=start_location,
                                        end_location=end_location
                                    ))
                                    logger.info(
                                        f"Route processing completed with {result.get('metadata', {}).get('total_detections', 0)} detections")
                                    return

                            print(
                                f"Route {args.route_id} not found in {csv_path}")
                    except Exception as e:
                        print(f"Error processing route from CSV: {str(e)}")
                        logger.error(f"Error processing route from CSV: {e}")
                        traceback.print_exc()

            # If we get here, we didn't find the route or there was an error
            if not args.start or not args.end:
                print(
                    "Error: When not using CSV routes, both --start and --end are required")
                return

            # Process with provided parameters
            result = asyncio.run(process_single_route(
                config=config,
                route_id=args.route_id,
                start_location=args.start,
                end_location=args.end
            ))
            logger.info(
                f"Route processing completed with {result.get('metadata', {}).get('total_detections', 0)} detections")

        elif args.csv:
            # Process batch of routes from CSV
            asyncio.run(process_route_batch(
                config=config,
                csv_file=args.csv,
                parallel=args.parallel,
                max_concurrent=args.max_concurrent,
                subset=args.subset
            ))
            logger.info("Batch processing completed")
        else:
            logger.error("Either --csv or --route-id must be specified")
            print("Error: Either --csv or --route-id must be specified")

    except Exception as e:
        logger.error(f"Error in process command: {e}")
        traceback.print_exc()


def plan_command(args, config):
    """Handle planning of routes for data acquisition."""
    logger = logging.getLogger("plan")

    try:
        logger.info("Route planning not yet implemented")
        print("Route planning functionality is not yet implemented")

    except Exception as e:
        logger.error(f"Error in plan command: {e}")
        traceback.print_exc()


def train_command(args, config):
    """Handle training and evaluation of sleeve detection models."""
    logger = logging.getLogger("train")

    try:
        if not args.mode:
            logger.error("--mode argument is required for train command")
            return

        if args.mode == 'prepare':
            # Prepare dataset
            if not args.data:
                logger.error("--data argument is required for prepare mode")
                return

            splits = [0.7, 0.15, 0.15]  # Default splits
            prepare_dataset(
                config=config,
                raw_data_path=args.data,
                splits=splits,
                single_class=args.single_class
            )

        elif args.mode == 'augment':
            # Augment dataset
            if not args.data or not args.output:
                logger.error(
                    "--data and --output arguments are required for augment mode")
                return

            severity = args.augment or 'medium'
            augment_dataset(
                config=config,
                dataset_path=args.data,
                output_path=args.output,
                multiplier=3,  # Default multiplier
                severity=severity
            )

        elif args.mode == 'train':
            # Train model
            if not args.model or not args.dataset:
                logger.error(
                    "--model and --dataset arguments are required for train mode")
                return

            train_model(
                config=config,
                base_model=args.model,
                dataset_yaml=args.dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=args.img_size,
                learning_rate=args.learning_rate
            )

        elif args.mode == 'auto-label':
            # Auto-label images
            if not args.model or not args.data or not args.output:
                logger.error(
                    "--model, --data, and --output arguments are required for auto-label mode")
                return

            auto_label_images(
                config=config,
                model_path=args.model,
                images_dir=args.data,
                output_dir=args.output,
                confidence=args.confidence
            )

        elif args.mode == 'tune':
            # Tune hyperparameters
            if not args.dataset or not args.model:
                logger.error(
                    "--dataset and --model arguments are required for tune mode")
                return

            tune_hyperparameters(
                config=config,
                dataset_yaml=args.dataset,
                base_model=args.model,
                n_trials=20  # Default number of trials
            )

        elif args.mode == 'create-ensemble':
            # Create ensemble
            if not args.model:
                logger.error(
                    "--model argument is required for create-ensemble mode")
                return

            create_ensemble(
                config=config,
                model_paths=args.model,
                output_config=args.output
            )

        else:
            logger.error(f"Unknown training mode: {args.mode}")

    except Exception as e:
        logger.error(f"Error in train command: {e}")
        traceback.print_exc()


def two_stage_command(args, config):
    """Handle two-stage detection command."""
    logger = logging.getLogger("two-stage")

    try:
        # Initialize the two-stage detector
        detector = TwoStageDetector(args.config)

        if args.mode == 'setup':
            # Create directory structure for two-stage detection
            dirs = detector.prepare_directory_structure(args.base_dir)
            logger.info(f"Created directory structure in {args.base_dir}")
            for category, category_dirs in dirs.items():
                if category != 'base':
                    logger.info(f"{category.capitalize()} directories:")
                    for name, path in category_dirs.items():
                        logger.info(f"  - {name}: {path}")

        elif args.mode == 'label-powerlines':
            if not args.images or not args.output:
                logger.error("--images and --output arguments are required")
                return

            # Setup environment for labeling powerlines
            detector.setup_labeling_environment(
                images_dir=args.images,
                output_dir=args.output,
                target="powerline"
            )

        elif args.mode == 'augment-powerlines':
            if not args.images or not args.output:
                logger.error("--images and --output arguments are required")
                return

            # Augment labeled powerline dataset
            original_count, augmented_count = detector.augment_dataset(
                dataset_dir=args.images,
                output_dir=args.output,
                num_augmentations=args.augmentations,
                severity=args.severity
            )

            logger.info(f"Augmentation complete:")
            logger.info(f"  - Original images: {original_count}")
            logger.info(f"  - Augmented images: {augmented_count}")
            logger.info(
                f"  - Total images: {original_count + augmented_count}")

        elif args.mode == 'train-powerlines':
            if not args.images or not args.output:
                logger.error("--images and --output arguments are required")
                return

            if args.hyperparameter_tuning:
                logger.info(
                    "Starting hyperparameter tuning for powerline detection model")
            else:
                logger.info(
                    "Starting training for powerline detection model with fixed hyperparameters")

            # Train powerline detection model
            model_path = detector.train_model(
                dataset_dir=args.images,
                output_dir=args.output,
                target="powerline",
                epochs=args.epochs,
                batch_size=args.batch_size,
                model_size=args.model_size,
                hyperparameter_tuning=args.hyperparameter_tuning
            )

            if model_path:
                logger.info(
                    f"Powerline model trained successfully: {model_path}")
            else:
                logger.error("Powerline model training failed")

        elif args.mode == 'detect-powerlines':
            if not args.images or not args.output or not args.powerline_model:
                logger.error(
                    "--images, --output, and --powerline-model arguments are required")
                return

            # Set the powerline model path
            detector.powerline_model_path = args.powerline_model

            # Detect powerlines in images
            detection_dir = detector.detect_powerlines(
                images_dir=args.images,
                output_dir=args.output,
                conf_threshold=args.conf_threshold
            )

            if detection_dir:
                logger.info(f"Powerline detection complete: {detection_dir}")
            else:
                logger.error("Powerline detection failed")

        elif args.mode == 'extract-regions':
            if not args.images or not args.output:
                logger.error("--images and --output arguments are required")
                return

            # Extract powerline regions for sleeve labeling
            regions_dir = detector.extract_powerline_regions(
                detection_dir=args.images,
                output_dir=args.output,
                padding=args.padding
            )

            if regions_dir:
                logger.info(f"Powerline regions extracted: {regions_dir}")
            else:
                logger.error("Failed to extract powerline regions")

        elif args.mode == 'label-sleeves':
            if not args.images or not args.output:
                logger.error("--images and --output arguments are required")
                return

            # Setup environment for labeling sleeves
            detector.setup_labeling_environment(
                images_dir=args.images,
                output_dir=args.output,
                target="sleeve"
            )

        elif args.mode == 'augment-sleeves':
            if not args.images or not args.output:
                logger.error("--images and --output arguments are required")
                return

            # Augment labeled sleeve dataset
            original_count, augmented_count = detector.augment_dataset(
                dataset_dir=args.images,
                output_dir=args.output,
                num_augmentations=args.augmentations,
                severity=args.severity
            )

            logger.info(f"Augmentation complete:")
            logger.info(f"  - Original images: {original_count}")
            logger.info(f"  - Augmented images: {augmented_count}")
            logger.info(
                f"  - Total images: {original_count + augmented_count}")

        elif args.mode == 'train-sleeves':
            if not args.images or not args.output:
                logger.error("--images and --output arguments are required")
                return

            # Train sleeve detection model
            model_path = detector.train_model(
                dataset_dir=args.images,
                output_dir=args.output,
                target="sleeve",
                epochs=args.epochs,
                batch_size=args.batch_size,
                model_size=args.model_size
            )

            if model_path:
                logger.info(f"Sleeve model trained successfully: {model_path}")
            else:
                logger.error("Sleeve model training failed")

        elif args.mode == 'run-pipeline':
            if not args.images or not args.output or not args.powerline_model:
                logger.error(
                    "--images, --output, and --powerline-model arguments are required")
                return

            # Run complete detection pipeline
            results = detector.run_complete_pipeline(
                raw_images_dir=args.images,
                output_base_dir=args.output,
                powerline_model_path=args.powerline_model,
                sleeve_model_path=args.sleeve_model
            )

            if results:
                logger.info(f"Detection pipeline completed successfully")
                for key, value in results.items():
                    logger.info(f"  - {key}: {value}")
            else:
                logger.error("Detection pipeline failed")

        else:
            logger.error(f"Unknown mode: {args.mode}")

    except Exception as e:
        logger.error(f"Error in two-stage detection: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
