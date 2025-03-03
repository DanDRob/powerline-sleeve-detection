#!/usr/bin/env python3
import os
import logging
import argparse
from pathlib import Path
from powerline_sleeve_detection.training.two_stage_detector import TwoStageDetector


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("two_stage_workflow.log")
        ]
    )
    return logging.getLogger("two-stage-example")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-Stage Powerline Sleeve Detection Example"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--raw-images",
        required=True,
        help="Directory containing raw input images"
    )
    parser.add_argument(
        "--base-dir",
        default="data/two_stage",
        help="Base directory for two-stage detection data"
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="Stage to start from (1-10)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing files"
    )
    return parser.parse_args()


def main():
    """Run the complete two-stage detection workflow."""
    args = parse_args()
    logger = setup_logging()

    logger.info("Starting two-stage detection workflow")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Raw images directory: {args.raw_images}")
    logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Starting from stage: {args.stage}")

    # Initialize the detector
    detector = TwoStageDetector(args.config)

    # Setup directory structure
    if args.stage <= 1:
        logger.info("Stage 1: Setting up directory structure")
        dirs = detector.prepare_directory_structure(args.base_dir)
        logger.info(f"Created directory structure in {args.base_dir}")
    else:
        logger.info(
            f"Skipping Stage 1 (setup) - starting from stage {args.stage}")

    # Define paths based on the directory structure
    base_path = Path(args.base_dir)
    powerline_paths = {
        "raw": base_path / "powerline" / "raw",
        "labeled": base_path / "powerline" / "labeled",
        "augmented": base_path / "powerline" / "augmented",
        "models": base_path / "powerline" / "models",
        "results": base_path / "powerline" / "results",
    }

    sleeve_paths = {
        "raw": base_path / "sleeve" / "raw",
        "labeled": base_path / "sleeve" / "labeled",
        "augmented": base_path / "sleeve" / "augmented",
        "models": base_path / "sleeve" / "models",
        "results": base_path / "sleeve" / "results",
    }

    # Create dirs that might not exist
    for path_dict in [powerline_paths, sleeve_paths]:
        for key, path in path_dict.items():
            os.makedirs(path, exist_ok=True)

    # Copy raw images to powerline/raw directory
    if args.stage <= 2:
        logger.info("Stage 2: Setting up environment for powerline labeling")
        detector.setup_labeling_environment(
            images_dir=args.raw_images,
            output_dir=str(powerline_paths["labeled"]),
            target="powerline"
        )

        logger.info("\nMANUAL STEP REQUIRED:")
        logger.info("Please label the powerlines in the images using LabelImg")
        logger.info(f"Once done, proceed to stage 3 for augmentation")
        logger.info(
            "Exiting workflow - continue after manual labeling is complete")
        return

    # Augment powerline dataset
    if args.stage <= 3:
        logger.info("Stage 3: Augmenting powerline dataset")
        if not list(powerline_paths["labeled"].glob("*.jpg")) and not list(powerline_paths["labeled"].glob("*.png")):
            logger.error(
                "No labeled powerline images found. Please complete the labeling step first.")
            return

        detector.augment_dataset(
            dataset_dir=str(powerline_paths["labeled"]),
            output_dir=str(powerline_paths["augmented"]),
            num_augmentations=5,
            severity="medium"
        )

    # Train powerline model
    if args.stage <= 4:
        logger.info("Stage 4: Training powerline detection model")
        detector.train_model(
            dataset_dir=str(powerline_paths["augmented"]),
            output_dir=str(powerline_paths["models"]),
            target="powerline",
            epochs=100,
            batch_size=16,
            model_size="m"
        )

    # Find the best powerline model
    powerline_model_dir = powerline_paths["models"] / \
        "powerline_detector_m" / "weights"
    powerline_model_path = powerline_model_dir / "best.pt"

    if not powerline_model_path.exists():
        logger.error(f"Powerline model not found at {powerline_model_path}")
        logger.error("Please train a powerline model before proceeding")
        return

    # Detect powerlines in raw images
    if args.stage <= 5:
        logger.info("Stage 5: Detecting powerlines in raw images")
        detector.powerline_model_path = str(powerline_model_path)
        detector.detect_powerlines(
            images_dir=args.raw_images,
            output_dir=str(powerline_paths["results"]),
            conf_threshold=0.25
        )

    # Extract powerline regions
    if args.stage <= 6:
        logger.info("Stage 6: Extracting powerline regions")
        detector.extract_powerline_regions(
            detection_dir=str(powerline_paths["results"]),
            output_dir=str(sleeve_paths["raw"]),
            padding=0.1
        )

    # Setup environment for sleeve labeling
    if args.stage <= 7:
        logger.info("Stage 7: Setting up environment for sleeve labeling")
        detector.setup_labeling_environment(
            images_dir=str(sleeve_paths["raw"]),
            output_dir=str(sleeve_paths["labeled"]),
            target="sleeve"
        )

        logger.info("\nMANUAL STEP REQUIRED:")
        logger.info(
            "Please label the sleeves in the extracted powerline regions using LabelImg")
        logger.info(f"Once done, proceed to stage 8 for augmentation")
        logger.info(
            "Exiting workflow - continue after manual labeling is complete")
        return

    # Augment sleeve dataset
    if args.stage <= 8:
        logger.info("Stage 8: Augmenting sleeve dataset")
        if not list(sleeve_paths["labeled"].glob("*.jpg")) and not list(sleeve_paths["labeled"].glob("*.png")):
            logger.error(
                "No labeled sleeve images found. Please complete the labeling step first.")
            return

        detector.augment_dataset(
            dataset_dir=str(sleeve_paths["labeled"]),
            output_dir=str(sleeve_paths["augmented"]),
            num_augmentations=5,
            severity="medium"
        )

    # Train sleeve model
    if args.stage <= 9:
        logger.info("Stage 9: Training sleeve detection model")
        detector.train_model(
            dataset_dir=str(sleeve_paths["augmented"]),
            output_dir=str(sleeve_paths["models"]),
            target="sleeve",
            epochs=100,
            batch_size=16,
            model_size="m"
        )

    # Find the best sleeve model
    sleeve_model_dir = sleeve_paths["models"] / "sleeve_detector_m" / "weights"
    sleeve_model_path = sleeve_model_dir / "best.pt"

    if not sleeve_model_path.exists():
        logger.warning(f"Sleeve model not found at {sleeve_model_path}")
        logger.warning("Will only perform powerline detection in the pipeline")
    else:
        detector.sleeve_model_path = str(sleeve_model_path)

    # Run complete pipeline
    if args.stage <= 10:
        logger.info("Stage 10: Running complete detection pipeline")
        results = detector.run_complete_pipeline(
            raw_images_dir=args.raw_images,
            output_base_dir=str(base_path / "results"),
            powerline_model_path=str(powerline_model_path),
            sleeve_model_path=str(
                sleeve_model_path) if sleeve_model_path.exists() else None
        )

        if results:
            logger.info("Detection pipeline completed successfully")
            for key, value in results.items():
                logger.info(f"  - {key}: {value}")
        else:
            logger.error("Detection pipeline failed")

    logger.info("Two-stage detection workflow completed")


if __name__ == "__main__":
    main()
