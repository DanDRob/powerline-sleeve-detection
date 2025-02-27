import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import yaml
import torch
from ultralytics import YOLO

from powerline_sleeve_detection.utils.config import Config

logger = logging.getLogger(__name__)


class SleeveModelTrainer:
    """Trainer for powerline sleeve detection models."""

    def __init__(self, config: Config):
        """Initialize the trainer with configuration."""
        self.config = config
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def initialize_model(self, pretrained_model: str) -> None:
        """Initialize the YOLO model from a pretrained checkpoint.

        Args:
            pretrained_model: Path to pretrained model or model type (e.g., 'yolov8n.pt')
        """
        try:
            self.model = YOLO(pretrained_model)
            logger.info(f"Loaded model from {pretrained_model}")
        except Exception as e:
            logger.error(f"Failed to load model from {pretrained_model}: {e}")
            raise

    def train(self,
              dataset_yaml: str,
              epochs: int = None,
              batch_size: int = None,
              image_size: int = None,
              patience: int = None,
              learning_rate: float = None) -> Dict[str, Any]:
        """Train the model on a custom dataset.

        Args:
            dataset_yaml: Path to dataset.yaml file
            epochs: Number of training epochs (override config)
            batch_size: Batch size for training (override config)
            image_size: Image size for training (override config)
            patience: Early stopping patience (override config)
            learning_rate: Learning rate (override config)

        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            raise ValueError(
                "Model not initialized. Call initialize_model first.")

        # Get training params from config, or use overrides if provided
        epochs = epochs or self.config.get('training.epochs', 100)
        batch_size = batch_size or self.config.get('training.batch_size', 16)
        image_size = image_size or self.config.get('training.image_size', 640)
        patience = patience or self.config.get('training.patience', 20)
        learning_rate = learning_rate or self.config.get(
            'training.learning_rate', 0.01)

        try:
            # Train the model
            logger.info(
                f"Starting training for {epochs} epochs with batch size {batch_size}")
            results = self.model.train(
                data=dataset_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                patience=patience,
                lr0=learning_rate,
                device=self.device,
                project=self.config.get('training.output_dir', 'runs/train'),
                name=self.config.get(
                    'training.experiment_name', 'sleeve_detection')
            )

            # Get best model path
            best_model_path = results.best
            logger.info(
                f"Training completed. Best model saved at: {best_model_path}")

            # Save path to best model in config
            self.config.set('model.custom_model_path', str(best_model_path))
            self.config.save()

            return {
                'metrics': results.metrics,
                'best_model_path': best_model_path
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def export_model(self, format: str = 'torchscript') -> str:
        """Export the trained model to different formats.

        Args:
            format: Format to export ('torchscript', 'onnx', etc.)

        Returns:
            Path to exported model
        """
        if self.model is None:
            raise ValueError(
                "Model not initialized. Call initialize_model first.")

        try:
            logger.info(f"Exporting model to {format} format")
            export_path = self.model.export(format=format)
            logger.info(f"Model exported to: {export_path}")
            return str(export_path)
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise

    def validate(self, dataset_yaml: str = None) -> Dict[str, Any]:
        """Validate the model on validation dataset.

        Args:
            dataset_yaml: Path to dataset.yaml file (if None, uses the same as training)

        Returns:
            Dictionary with validation metrics
        """
        if self.model is None:
            raise ValueError(
                "Model not initialized. Call initialize_model first.")

        try:
            dataset_yaml = dataset_yaml or self.config.get(
                'training.dataset_yaml')
            logger.info(f"Validating model on {dataset_yaml}")
            results = self.model.val(data=dataset_yaml)
            logger.info(f"Validation results: mAP@0.5 = {results.box.map50:.4f}, "
                        f"mAP@0.5:0.95 = {results.box.map:.4f}")
            return results
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
