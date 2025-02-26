import os
import torch
import time
from typing import Dict, List, Any, Optional, Union
import shutil
import requests
from tqdm import tqdm
import logging
import gc

from ..system.config import Config
from ..system.logging import get_logger


class ModelManager:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("model_manager")
        self.models = {}  # Will store model metadata, not actual models
        self.loaded_models = {}  # Will store actually loaded model objects
        self.device = self._get_device()
        self.model_last_used = {}  # Track when each model was last used

    def _get_device(self) -> torch.device:
        """Determine the device to use for inference"""
        if self.config.detection.device == "auto":
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.detection.device)

        self.logger.info(f"Using device: {device}")
        return device

    def _download_model(self, model_name: str, model_url: str, model_path: str) -> bool:
        """
        Download a model file from URL if it doesn't exist.

        Args:
            model_name: Name of the model
            model_url: URL to download from
            model_path: Path to save the model

        Returns:
            Success status
        """
        if os.path.exists(model_path):
            self.logger.info(
                f"Model {model_name} already exists at {model_path}")
            return True

        self.logger.info(f"Downloading model {model_name} from {model_url}")

        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Make a streaming request
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))

                # Create a progress bar
                progress_bar = tqdm(
                    total=total_size, unit="B", unit_scale=True)

                # Download in chunks
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

                progress_bar.close()

            self.logger.info(
                f"Successfully downloaded model {model_name} to {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {e}")
            # Clean up partial download
            if os.path.exists(model_path):
                os.remove(model_path)
            return False

    def _actually_load_yolov8_model(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """
        Actually load a YOLOv8 model into memory.

        Args:
            model_name: Name of the model
            model_path: Path to the model file

        Returns:
            Dictionary with model information
        """
        try:
            # Import here to avoid loading ultralytics at module level
            from ultralytics import YOLO

            self.logger.info(
                f"Loading YOLOv8 model {model_name} from {model_path}")
            model = YOLO(model_path)

            # Move model to the appropriate device if not already there
            model.to(self.device)

            self.logger.info(f"Successfully loaded YOLOv8 model {model_name}")

            return {
                "model": model,
                "type": "yolov8",
                "name": model_name,
                "path": model_path,
                "loaded_at": time.time()
            }

        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 model {model_name}: {e}")
            return {"success": False, "error": str(e)}

    def _actually_load_yolov5_model(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """
        Actually load a YOLOv5 model into memory.

        Args:
            model_name: Name of the model
            model_path: Path to the model file

        Returns:
            Dictionary with model information
        """
        try:
            self.logger.info(
                f"Loading YOLOv5 model {model_name} from {model_path}")

            # Check if model exists
            if not os.path.exists(model_path):
                self.logger.warning(
                    f"Model {model_name} not found at {model_path}, will use pre-trained YOLOv5s")
                model = torch.hub.load(
                    'ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
            else:
                model = torch.hub.load(
                    'ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)

            # Move model to the appropriate device
            model.to(self.device)

            self.logger.info(f"Successfully loaded YOLOv5 model {model_name}")

            return {
                "model": model,
                "type": "yolov5",
                "name": model_name,
                "path": model_path,
                "loaded_at": time.time()
            }

        except Exception as e:
            self.logger.error(f"Failed to load YOLOv5 model {model_name}: {e}")
            return {"success": False, "error": str(e)}

    def _actually_load_efficientdet_model(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """
        Actually load an EfficientDet model into memory.

        Args:
            model_name: Name of the model
            model_path: Path to the model file

        Returns:
            Dictionary with model information
        """
        try:
            # Import here to avoid loading effdet at module level
            import effdet

            self.logger.info(
                f"Loading EfficientDet model {model_name} from {model_path}")

            # Load model
            model = effdet.create_model_from_config(
                config_name='tf_efficientdet_d0',
                pretrained=False,
                checkpoint_path=model_path
            )

            # Move model to the appropriate device
            model.to(self.device)
            model.eval()

            self.logger.info(
                f"Successfully loaded EfficientDet model {model_name}")

            return {
                "model": model,
                "type": "efficientdet",
                "name": model_name,
                "path": model_path,
                "loaded_at": time.time()
            }

        except Exception as e:
            self.logger.error(
                f"Failed to load EfficientDet model {model_name}: {e}")
            return {"success": False, "error": str(e)}

    def register_models(self) -> Dict[str, Any]:
        """
        Register all models specified in the configuration, but don't load them yet.

        Returns:
            Dictionary with registered models information
        """
        registered_models = []
        failed_models = []

        # Register models based on their paths
        for model_name, model_path in self.config.detection.model_paths.items():
            # Determine model type from name or path
            model_type = "yolov8"  # default
            if "yolov5" in model_name.lower() or "yolov5" in model_path.lower():
                model_type = "yolov5"
            elif "efficientdet" in model_name.lower() or "efficientdet" in model_path.lower():
                model_type = "efficientdet"

            # Check if model file exists
            if not os.path.exists(model_path) and model_type != "yolov5":
                # YOLOv5 can use pretrained models, others need file to exist
                self.logger.warning(f"Model file not found: {model_path}")
                failed_models.append({
                    "name": model_name,
                    "path": model_path,
                    "error": "Model file not found"
                })
                continue

            # Register model metadata
            self.models[model_name] = {
                "type": model_type,
                "path": model_path,
                "name": model_name,
                "registered_at": time.time(),
                "loaded": False
            }

            registered_models.append({
                "name": model_name,
                "type": model_type,
                "path": model_path
            })

            self.logger.info(f"Registered model: {model_name} ({model_type})")

        # Check if we have at least one model registered
        if not self.models:
            self.logger.warning(
                "No models were registered. Will register YOLOv5s as a fallback.")

            # Register YOLOv5s as fallback
            fallback_name = "yolov5s_fallback"
            self.models[fallback_name] = {
                "type": "yolov5",
                "path": "",  # Empty path means use pretrained
                "name": fallback_name,
                "registered_at": time.time(),
                "loaded": False
            }

            registered_models.append({
                "name": fallback_name,
                "type": "yolov5",
                "path": "pretrained"
            })

        self.logger.info(
            f"Registered {len(registered_models)} models. {len(failed_models)} models failed registration.")

        return {
            "success": len(registered_models) > 0,
            "registered_models": registered_models,
            "failed_models": failed_models,
            "total_models": len(registered_models) + len(failed_models)
        }

    def load_model(self, model_name: str) -> Dict[str, Any]:
        """
        Load a specific model by name.

        Args:
            model_name: Name of the model to load

        Returns:
            Dictionary with model information
        """
        # First, check if the model is already loaded
        if model_name in self.loaded_models:
            self.logger.debug(
                f"Model {model_name} already loaded, updating last used time")
            self.model_last_used[model_name] = time.time()
            return self.loaded_models[model_name]

        # Check if the model is registered
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not registered")
            return {"success": False, "error": f"Model {model_name} not registered"}

        # Get model metadata
        model_info = self.models[model_name]
        model_type = model_info["type"]
        model_path = model_info["path"]

        # Before loading a new model, check memory usage and unload least recently used models if needed
        self._manage_loaded_models()

        # Load the model based on its type
        if model_type == "yolov8":
            result = self._actually_load_yolov8_model(model_name, model_path)
        elif model_type == "yolov5":
            result = self._actually_load_yolov5_model(model_name, model_path)
        elif model_type == "efficientdet":
            result = self._actually_load_efficientdet_model(
                model_name, model_path)
        else:
            self.logger.error(f"Unsupported model type: {model_type}")
            return {"success": False, "error": f"Unsupported model type: {model_type}"}

        # Check if loading was successful
        if "model" in result:
            self.loaded_models[model_name] = result
            self.model_last_used[model_name] = time.time()

            # Update model metadata
            self.models[model_name]["loaded"] = True
            self.models[model_name]["last_loaded"] = time.time()

            return result
        else:
            return {"success": False, "error": result.get("error", "Unknown error")}

    def _manage_loaded_models(self, max_models: int = 2) -> None:
        """
        Manage loaded models to control memory usage.
        Unloads least recently used models if we have too many loaded.

        Args:
            max_models: Maximum number of models to keep loaded
        """
        if len(self.loaded_models) <= max_models:
            return

        # Sort models by last used time
        sorted_models = sorted(
            self.model_last_used.items(), key=lambda x: x[1])

        # Unload oldest models until we're within limit
        models_to_unload = len(self.loaded_models) - max_models
        for i in range(models_to_unload):
            if i < len(sorted_models):
                model_name = sorted_models[i][0]
                self.unload_model(model_name)

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_name: Name of the model to unload

        Returns:
            Success status
        """
        if model_name not in self.loaded_models:
            self.logger.debug(f"Model {model_name} is not loaded")
            return False

        try:
            # Remove reference to model
            del self.loaded_models[model_name]
            if model_name in self.model_last_used:
                del self.model_last_used[model_name]

            # Update metadata
            if model_name in self.models:
                self.models[model_name]["loaded"] = False

            # Run garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info(f"Unloaded model {model_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error unloading model {model_name}: {e}")
            return False

    def get_model(self, model_name: str) -> Any:
        """
        Get a model by name, loading it if necessary.

        Args:
            model_name: Name of the model to retrieve

        Returns:
            Model object or None if not found
        """
        # If model isn't already loaded, load it
        if model_name not in self.loaded_models:
            result = self.load_model(model_name)
            if "model" not in result:
                self.logger.warning(
                    f"Failed to load model {model_name}: {result.get('error')}")
                return None

        # Update last used time
        self.model_last_used[model_name] = time.time()

        return self.loaded_models[model_name]["model"]

    def get_ensemble_models(self) -> List[Dict[str, Any]]:
        """
        Get models for ensemble detection with their weights.
        Loads models if not already loaded.

        Returns:
            List of model dictionaries with weights
        """
        # Check if we have ensemble weights in the config
        if not self.config.detection.ensemble_weights:
            # Use equal weights for all models
            weights = {name: 1.0 / len(self.models) for name in self.models}
        else:
            weights = self.config.detection.ensemble_weights

        ensemble_models = []
        for name in list(self.models.keys()):
            if name in weights:
                # Load the model if not already loaded
                if name not in self.loaded_models:
                    result = self.load_model(name)
                    if "model" not in result:
                        self.logger.warning(
                            f"Failed to load model {name} for ensemble: {result.get('error')}")
                        continue

                # Use the loaded model
                ensemble_models.append({
                    **self.loaded_models[name],
                    "weight": weights.get(name, 1.0 / len(self.models))
                })

        # Sort by weight
        ensemble_models.sort(key=lambda x: x["weight"], reverse=True)

        return ensemble_models

    def load_models(self) -> Dict[str, Any]:
        """
        For backward compatibility: Register models and load the first one.

        Returns:
            Dictionary with loaded models information
        """
        # First register all models
        result = self.register_models()

        # Then load the first model as a starting point
        if result["success"] and result["registered_models"]:
            first_model = result["registered_models"][0]["name"]
            self.load_model(first_model)

        return result

    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all registered and loaded models.

        Returns:
            Dictionary with model status information
        """
        status = {
            "registered_models": len(self.models),
            "loaded_models": len(self.loaded_models),
            "models": []
        }

        for name, info in self.models.items():
            model_status = {
                "name": name,
                "type": info["type"],
                "registered": True,
                "loaded": name in self.loaded_models,
                "last_used": self.model_last_used.get(name, None)
            }
            status["models"].append(model_status)

        return status
