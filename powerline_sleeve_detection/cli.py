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
    """Main entry point for the command line interface."""
    parser = argparse.ArgumentParser(
        description="Powerline sleeve detection utility"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    subparsers.required = True

    # Process command
    process_parser = subparsers.add_parser(
        "process", help="Process routes and detect sleeves"
    )
    process_parser.add_argument(
        "--route-id", help="Process a specific route ID", required=False
    )
    process_parser.add_argument(
        "--video-list",
        help="Process videos from a list in a CSV file",
        required=False,
    )
    process_parser.add_argument(
        "--model", help="Path to a custom YOLO model", required=False
    )
    process_parser.add_argument(
        "--output", help="Output directory", required=False
    )
    process_parser.add_argument(
        "--batch", help="Process multiple routes from a CSV file", required=False
    )
    process_parser.add_argument(
        "--parallel",
        help="Process batch in parallel",
        action="store_true",
        required=False
    )
    process_parser.add_argument(
        "--max-tasks",
        help="Maximum number of concurrent tasks for parallel processing",
        type=int,
        default=4,
        required=False
    )
    process_parser.add_argument(
        "--validation",
        help="Percentage of routes to use for validation (0-100)",
        type=float,
        required=False
    )

    # Plan command
    plan_parser = subparsers.add_parser(
        "plan", help="Plan routes for data acquisition"
    )
    plan_parser.add_argument(
        "--route-id", help="Plan flights for a specific route ID", required=False
    )
    plan_parser.add_argument(
        "--output", help="Output directory", required=False
    )
    plan_parser.add_argument(
        "--batch", help="Plan multiple routes from a CSV file", required=False
    )

    # Train command - new addition
    train_parser = subparsers.add_parser(
        "train", help="Train and evaluate sleeve detection models"
    )
    train_parser.add_argument(
        "--mode",
        choices=["prepare", "augment", "train", "tune", "auto-label",
                 "create-ensemble", "full-workflow", "evaluate"],
        help="Training or evaluation mode",
        required=True
    )
    train_parser.add_argument(
        "--config",
        help="Path to configuration file",
        default="config.yaml"
    )
    train_parser.add_argument(
        "--data",
        help="Path to raw data directory"
    )
    train_parser.add_argument(
        "--dataset-yaml",
        help="Path to dataset YAML file"
    )
    train_parser.add_argument(
        "--base-model",
        help="Base YOLOv8 model to fine-tune"
    )
    train_parser.add_argument(
        "--model-path",
        help="Path to model for evaluation or auto-labeling"
    )
    train_parser.add_argument(
        "--model-paths",
        nargs="+",
        help="Paths to models for ensemble or comparison"
    )
    train_parser.add_argument(
        "--output-dir",
        help="Output directory for training results"
    )
    train_parser.add_argument(
        "--eval-mode",
        choices=["single", "ensemble", "compare"],
        help="Evaluation mode when --mode=evaluate"
    )
    train_parser.add_argument(
        "--additional-args",
        help="Additional arguments in format 'key1=value1 key2=value2'"
    )

    # Initialize the config
    initialize_config()

    # Parse the arguments
    args = parser.parse_args()

    if args.command == "process":
        process_command(args)
    elif args.command == "plan":
        plan_command(args)
    elif args.command == "train":
        train_command(args)
    else:
        parser.print_help()


def train_command(args):
    """Handle the train command."""
    try:
        # Setup logging
        from powerline_sleeve_detection.system.logging import setup_logging
        setup_logging()
        logger = logging.getLogger("train-cli")

        # Process additional arguments if provided
        additional_params = []
        if args.additional_args:
            # Convert string like "key1=value1 key2=value2" to command line arguments
            for item in args.additional_args.split():
                if "=" in item:
                    key, value = item.split("=", 1)
                    additional_params.extend([f"--{key}", value])
                else:
                    additional_params.append(f"--{item}")

        # Build the command to pass to train.py
        cmd = [sys.executable, "-m", "powerline_sleeve_detection.training.train"]

        # Add config argument
        if args.config:
            cmd.extend(["--config", args.config])

        # If evaluation mode
        if args.mode == "evaluate":
            cmd.append("evaluate")

            # Add evaluation mode
            if args.eval_mode:
                cmd.extend(["--mode", args.eval_mode])

            # Add model path for single evaluation
            if args.model_path:
                cmd.extend(["--model-path", args.model_path])

            # Add model paths for ensemble or comparison
            if args.model_paths:
                model_paths_flat = []
                for path in args.model_paths:
                    model_paths_flat.extend(["--model-paths", path])
                cmd.extend(model_paths_flat)

            # Add output directory
            if args.output_dir:
                cmd.extend(["--output-dir", args.output_dir])

            # Add dataset yaml
            if args.dataset_yaml:
                cmd.extend(["--dataset-yaml", args.dataset_yaml])

        # If training mode
        else:
            cmd.append("train")

            # Add training mode
            cmd.extend(["--mode", args.mode])

            # Add data directory
            if args.data:
                cmd.extend(["--data", args.data])

            # Add dataset yaml
            if args.dataset_yaml:
                cmd.extend(["--dataset-yaml", args.dataset_yaml])

            # Add base model
            if args.base_model:
                cmd.extend(["--base-model", args.base_model])

            # Add model path for auto-labeling
            if args.model_path:
                cmd.extend(["--model-path", args.model_path])

            # Add model paths for ensemble
            if args.model_paths:
                model_paths_flat = []
                for path in args.model_paths:
                    model_paths_flat.extend(["--model-paths", path])
                cmd.extend(model_paths_flat)

            # Add output directory
            if args.output_dir:
                cmd.extend(["--output-dir", args.output_dir])

        # Add any additional parameters
        if additional_params:
            cmd.extend(additional_params)

        # Log the command
        logger.info(f"Executing: {' '.join(cmd)}")

        # Execute the command
        import subprocess
        process = subprocess.run(cmd, check=True)

        # Check if the command was successful
        if process.returncode == 0:
            logger.info(f"Successfully completed {args.mode} operation")
        else:
            logger.error(
                f"Command failed with return code {process.returncode}")
            sys.exit(process.returncode)

    except Exception as e:
        logging.error(f"Error in train command: {e}")
        traceback.print_exc()
        sys.exit(1)


def process_command(args):
    """Handle the process command."""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger("process-cli")

        # Load configuration
        config = load_config(args.config if hasattr(args, "config") else "config.yaml")
        
        # Import csv module
        import csv
        import asyncio

        # Initialize batch processor
        processor = BatchProcessor(config)

        # Process single route
        if args.route_id:
            logger.info(f"Processing single route: {args.route_id}")

            # Override model if specified
            if args.model:
                config.set("detector.model_path", args.model)
                logger.info(f"Using custom model: {args.model}")

            # Override output if specified
            if args.output:
                config.set("system.output_dir", args.output)
                logger.info(f"Using custom output directory: {args.output}")
                
            # Find route details from CSV
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_routes.csv")
            start_location = None
            end_location = None
            
            if not os.path.exists(csv_path):
                logger.error(f"Routes file not found: {csv_path}")
                sys.exit(1)
                
            with open(csv_path, 'r') as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    if row['route_id'] == args.route_id:
                        start_location = row['start_location']
                        end_location = row['end_location']
                        break
            
            if not start_location or not end_location:
                logger.error(f"Route ID {args.route_id} not found in routes file")
                sys.exit(1)
                
            logger.info(f"Found route: {start_location} to {end_location}")
            
            # Process the route using asyncio
            result = asyncio.run(processor.process_route(args.route_id, start_location, end_location))

            if result and result.get("success", False):
                logger.info(f"Successfully processed route {args.route_id}")
                logger.info(
                    f"Results saved to: {result.get('output_dir', 'unknown')}")
            else:
                logger.error(f"Failed to process route {args.route_id}")
                if result and "error" in result:
                    logger.error(f"Error: {result['error']}")
                sys.exit(1)

        # Process video list
        elif args.video_list:
            logger.info(f"Processing videos from list: {args.video_list}")

            # Override model if specified
            if args.model:
                config.set("detector.model_path", args.model)
                logger.info(f"Using custom model: {args.model}")

            # Override output if specified
            if args.output:
                config.set("system.output_dir", args.output)
                logger.info(f"Using custom output directory: {args.output}")

            result = asyncio.run(processor.process_videos_from_csv(args.video_list))

            if result and result.get("success", False):
                logger.info("Successfully processed videos from list")
                logger.info(
                    f"Results saved to: {result.get('output_dir', 'unknown')}")
            else:
                logger.error("Failed to process videos from list")
                if result and "error" in result:
                    logger.error(f"Error: {result['error']}")
                sys.exit(1)

        # Process batch from CSV
        elif args.batch:
            logger.info(f"Processing batch from CSV: {args.batch}")

            # Override model if specified
            if args.model:
                config.set("detector.model_path", args.model)
                logger.info(f"Using custom model: {args.model}")

            # Override output if specified
            if args.output:
                config.set("system.output_dir", args.output)
                logger.info(f"Using custom output directory: {args.output}")

            # Process batch with optional validation subset
            validation_subset = args.validation / \
                100.0 if args.validation is not None else None

            result = asyncio.run(processor.process_batch_from_csv(
                args.batch,
                parallel=args.parallel,
                max_concurrent_tasks=args.max_tasks,
                validation_subset=validation_subset
            ))

            if result and result.get("success", False):
                logger.info("Successfully processed batch")
                logger.info(
                    f"Results saved to: {result.get('output_dir', 'unknown')}")
                if result.get("combined_results_file"):
                    logger.info(
                        f"Combined results saved to: {result['combined_results_file']}")
            else:
                logger.error("Failed to process batch")
                if result and "error" in result:
                    logger.error(f"Error: {result['error']}")
                sys.exit(1)

        # No action specified
        else:
            logger.error(
                "No action specified. Please specify --route-id, --video-list, or --batch")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Error in process command: {e}")
        traceback.print_exc()
        sys.exit(1)


def plan_command(args):
    """Handle the plan command."""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger("plan-cli")

        # Load configuration
        config = load_config(args.config if hasattr(args, "config") else "config.yaml")

        # Import route planner module
        from powerline_sleeve_detection.acquisition.route_planner import RoutePlanner
        import csv

        # Initialize route planner
        planner = RoutePlanner(config)

        # Plan single route
        if args.route_id:
            logger.info(f"Planning route: {args.route_id}")

            # Override output if specified
            if args.output:
                config.set("system.output_dir", args.output)
                logger.info(f"Using custom output directory: {args.output}")
            
            # Find route details from CSV
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_routes.csv")
            start_location = None
            end_location = None
            
            if not os.path.exists(csv_path):
                logger.error(f"Routes file not found: {csv_path}")
                sys.exit(1)
                
            with open(csv_path, 'r') as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    if row['route_id'] == args.route_id:
                        start_location = row['start_location']
                        end_location = row['end_location']
                        break
            
            if not start_location or not end_location:
                logger.error(f"Route ID {args.route_id} not found in routes file")
                sys.exit(1)
                
            logger.info(f"Found route: {start_location} to {end_location}")
            result = planner.plan_route(start_location, end_location)

            if result and result.get("success", False):
                logger.info(f"Successfully planned route {args.route_id}")
                logger.info(
                    f"Plan saved to: {result.get('output_file', 'unknown')}")
            else:
                logger.error(f"Failed to plan route {args.route_id}")
                if result and "error" in result:
                    logger.error(f"Error: {result['error']}")
                sys.exit(1)

        # Plan batch from CSV
        elif args.batch:
            logger.info(f"Planning batch from CSV: {args.batch}")

            # Override output if specified
            if args.output:
                config.set("system.output_dir", args.output)
                logger.info(f"Using custom output directory: {args.output}")

            result = planner.plan_routes_from_csv(args.batch)

            if result and result.get("success", False):
                logger.info("Successfully planned routes")
                logger.info(
                    f"Plans saved to: {result.get('output_dir', 'unknown')}")
                if result.get("summary_file"):
                    logger.info(f"Summary saved to: {result['summary_file']}")
            else:
                logger.error("Failed to plan routes")
                if result and "error" in result:
                    logger.error(f"Error: {result['error']}")
                sys.exit(1)

        # No action specified
        else:
            logger.error(
                "No action specified. Please specify --route-id or --batch")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Error in plan command: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
