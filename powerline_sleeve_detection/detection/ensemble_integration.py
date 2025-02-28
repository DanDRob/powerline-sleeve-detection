# Add future import for postponed evaluation of annotations
from __future__ import annotations
import os
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import logging
from PIL import Image
import torch
from pathlib import Path

from ..system.config import Config
from ..system.logging import get_logger
from .model_manager import ModelManager
from ..training.ensemble import EnsembleDetector


class EnsembleIntegration:
    """
    Integrates the EnsembleDetector from training module with the existing detection pipeline.
    """

    def __init__(self, config: Config, model_manager: Optional[ModelManager] = None):
        """
        Initialize the ensemble integration.

        Args:
            config: Application configuration
            model_manager: Optional model manager to use
        """
        self.config = config
        self.logger = get_logger("ensemble_integration")
        self.model_manager = model_manager
        self.ensemble_detector = None

        # Load ensemble detector if enabled
        if config.get('ensemble.enabled', False):
            self._init_ensemble_detector()

    def _init_ensemble_detector(self) -> None:
        """Initialize the ensemble detector with models from config."""
        try:
            self.ensemble_detector = EnsembleDetector(self.config)
            self.ensemble_detector.load_models_from_config()
            self.logger.info("Ensemble detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ensemble detector: {e}")
            self.ensemble_detector = None

    def detect_with_ensemble(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run detection using the ensemble detector.

        Args:
            image: PIL Image to detect on

        Returns:
            Dictionary with detection results
        """
        if self.ensemble_detector is None:
            self.logger.error("Ensemble detector not initialized")
            return {"success": False, "error": "Ensemble detector not initialized"}

        # Save image to temporary file for ensemble detector
        try:
            temp_dir = Path(self.config.get('system.temp_dir', './temp'))
            temp_dir.mkdir(exist_ok=True, parents=True)

            temp_path = temp_dir / "temp_ensemble_input.jpg"
            image.save(temp_path)

            # Run prediction
            result = self.ensemble_detector.predict(temp_path)

            # Convert to the format expected by SleeveDetector
            processed_result = self._convert_ensemble_results(
                result, image.size)

            # Add metadata
            return {
                "detections": processed_result,
                "model_name": "ensemble",
                "model_type": "ensemble",
                "success": True,
                "processing_time": 0,  # Could add timing if needed
                "image_size": image.size
            }

        except Exception as e:
            self.logger.error(f"Error in ensemble detection: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # Clean up temporary file
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()

    def _convert_ensemble_results(self,
                                  ensemble_result: Dict[str, Any],
                                  image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Convert ensemble results to the format expected by the SleeveDetector.

        Args:
            ensemble_result: Result from ensemble detector
            image_size: Original image size (width, height)

        Returns:
            List of detection dictionaries
        """
        detections = []
        width, height = image_size

        # Check if we have valid boxes
        if ensemble_result['boxes'] is None:
            return detections

        # Process each detection
        for i, (box, score, label) in enumerate(zip(
                ensemble_result['boxes'],
                ensemble_result['scores'],
                ensemble_result['labels'])):

            x1, y1, x2, y2 = map(float, box)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score),
                "class_id": int(label),
                "normalized_bbox": [x1/width, y1/height, x2/width, y2/height]
            })

        return detections

    def integrate_with_detector(self, detector: 'SleeveDetector') -> None:
        """
        Integrate ensemble detection with an existing detector instance.

        Args:
            detector: SleeveDetector instance to integrate with
        """
        # No need to import SleeveDetector - use duck typing instead

        if not self.ensemble_detector:
            self.logger.warning(
                "No ensemble detector available for integration")
            return

        # Add a reference to the ensemble integration in the detector
        detector.ensemble_integration = self

        # Patch the detector's detect_ensemble method
        original_ensemble_method = detector.detect_ensemble

        def enhanced_detect_ensemble(image):
            # Check if we should use our advanced ensemble instead
            if self.config.get('ensemble.use_advanced_ensemble', True):
                self.logger.debug("Using advanced ensemble detector")
                return self.detect_with_ensemble(image)
            else:
                # Fall back to original ensemble method
                self.logger.debug("Using original ensemble detection")
                return original_ensemble_method(image)

        # Replace the method
        detector.detect_ensemble = enhanced_detect_ensemble

        self.logger.info(
            "Successfully integrated ensemble detector with SleeveDetector")
