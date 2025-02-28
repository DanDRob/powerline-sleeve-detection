#!/usr/bin/env python3
from powerline_sleeve_detection.system.config import Config
from powerline_sleeve_detection.system.logging import setup_logging
import os
import sys
import logging
import argparse
from pathlib import Path
import importlib

# Add the root directory to the path
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = script_dir.parent.parent
sys.path.insert(0, str(root_dir))


# Configure logging
setup_logging(level=logging.INFO)
logger = logging.getLogger("train")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Powerline Sleeve Detection Training and Evaluation Tool")

    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--mode", type=str, required=True,
                              choices=["prepare", "augment", "train", "tune",
                                       "auto-label", "create-ensemble", "full-workflow"],
                              help="Training mode")
    train_parser.add_argument(
        "--data", type=str, help="Path to raw data directory")
    train_parser.add_argument("--splits", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                              help="Train/val/test splits (must sum to 1.0)")
    train_parser.add_argument("--single-class", action="store_true", default=True,
                              help="Convert all classes to single 'sleeve' class")
    train_parser.add_argument("--class-names", type=str, nargs="+",
                              help="Class names (if not single-class)")
    train_parser.add_argument("--severity", type=str, default="medium",
                              choices=["light", "medium", "heavy"],
                              help="Augmentation severity")
    train_parser.add_argument("--multiplier", type=int, default=3,
                              help="Augmentation multiplier")
    train_parser.add_argument("--dataset-yaml", type=str,
                              help="Path to dataset.yaml file")
    train_parser.add_argument("--base-model", type=str, default=None,
                              help="Base YOLOv8 model to fine-tune")
    train_parser.add_argument("--epochs", type=int,
                              help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int,
                              help="Training batch size")
    train_parser.add_argument(
        "--image-size", type=int, help="Input image size")
    train_parser.add_argument(
        "--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument(
        "--output-dir", type=str, help="Output directory")
    train_parser.add_argument("--trials", type=int, default=20,
                              help="Number of hyperparameter tuning trials")
    train_parser.add_argument("--timeout", type=int,
                              help="Timeout for hyperparameter tuning in seconds")
    train_parser.add_argument("--model-path", type=str,
                              help="Path to trained model for auto-labeling")
    train_parser.add_argument("--confidence", type=float, default=0.5,
                              help="Confidence threshold for auto-labeling")
    train_parser.add_argument("--model-paths", type=str, nargs="+",
                              help="Paths to models for ensemble")
    train_parser.add_argument("--weights", type=float, nargs="+",
                              help="Weights for ensemble models")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--mode", type=str, required=True,
                             choices=["single", "ensemble", "compare"],
                             help="Evaluation mode")
    eval_parser.add_argument("--model-path", type=str,
                             help="Path to model for single evaluation")
    eval_parser.add_argument("--model-paths", type=str, nargs="+",
                             help="Multiple model paths for comparison")
    eval_parser.add_argument("--model-names", type=str, nargs="+",
                             help="Names for models being compared")
    eval_parser.add_argument("--test-dir", type=str,
                             help="Directory containing test images")
    eval_parser.add_argument("--ground-truth-dir", type=str,
                             help="Directory containing ground truth labels")
    eval_parser.add_argument("--dataset-yaml", type=str,
                             help="YAML file for dataset (for YOLOv8 .val())")
    eval_parser.add_argument("--output-dir", type=str, default="evaluation_results",
                             help="Directory to save evaluation results")
    eval_parser.add_argument("--save-visualizations", action="store_true", default=True,
                             help="Save detection visualizations")
    eval_parser.add_argument("--confidence-threshold", type=float, default=0.25,
                             help="Confidence threshold for detections")
    eval_parser.add_argument("--iou-threshold", type=float, default=0.5,
                             help="IoU threshold for mAP calculation")

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)

        # Load config
        with open(config_path, 'r') as f:
            import yaml
            config_dict = yaml.safe_load(f)

        config = Config(config_dict)
        logger.info(f"Loaded configuration from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def run_train_command(args):
    """Run the training module with the specified arguments."""
    try:
        # Import the train_model module
        from powerline_sleeve_detection.training.train_model import (
            prepare_dataset, augment_dataset, train_model,
            tune_hyperparameters, auto_label_images, create_ensemble,
            run_full_workflow
        )

        # Load config
        config = load_config(args.config)

        # Enable training mode
        config.set('training.enabled', True)
        config.save()

        # Execute the requested mode
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

    except ImportError as e:
        logger.error(f"Failed to import training module: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in {args.mode} operation: {e}")
        sys.exit(1)


def run_evaluate_command(args):
    """Run the evaluation module with the specified arguments."""
    try:
        # Import the evaluate_model module
        from powerline_sleeve_detection.training.evaluate_model import (
            evaluate_single_model, evaluate_ensemble, compare_models
        )

        # Load config
        config = load_config(args.config)

        # Execute the requested mode
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

    except ImportError as e:
        logger.error(f"Failed to import evaluation module: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in {args.mode} evaluation: {e}")
        sys.exit(1)


def main():
    """Main entry point for the script."""
    args = parse_args()

    # Check if a command was provided
    if not args.command:
        logger.error("No command specified. Use 'train' or 'evaluate'.")
        sys.exit(1)

    # Run the appropriate command
    if args.command == "train":
        run_train_command(args)
    elif args.command == "evaluate":
        run_evaluate_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
