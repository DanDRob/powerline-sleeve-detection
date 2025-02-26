import os
import sys
import requests
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any
import logging
import yaml
import json
import hashlib
import time

from ..system.config import Config
from ..system.logging import get_logger


class ModelDownloader:
    """
    Utility for downloading detection models.
    Can automatically download missing models when needed.
    """

    # Model repository URLs - these are the official Ultralytics links for YOLOv8 models
    MODEL_URLS = {
        "yolov8n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8s": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "yolov8m": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "yolov8l": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        "yolov8x": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
        # YOLOv5 models
        "yolov5n": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt",
        "yolov5s": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt",
        "yolov5m": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m.pt",
        "yolov5l": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l.pt",
        "yolov5x": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x.pt",
    }

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("model_downloader")
        self.models_dir = os.path.join(self.config.system.cache_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def download_file(self, url: str, destination: str, chunk_size: int = 8192) -> bool:
        """
        Download a file from URL to the specified destination with progress bar.

        Args:
            url: URL to download from
            destination: Local path to save the file
            chunk_size: Size of chunks to download

        Returns:
            Success status
        """
        try:
            self.logger.info(f"Downloading from {url} to {destination}")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(destination), exist_ok=True)

            # Start download
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))

            # Download with progress bar
            with open(destination, 'wb') as f, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            self.logger.info(f"Download completed: {destination}")
            return True

        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            # Remove partial download if it exists
            if os.path.exists(destination):
                os.remove(destination)
            return False

    def verify_file_hash(self, file_path: str, expected_hash: str, hash_type: str = 'md5') -> bool:
        """
        Verify file hash to ensure integrity.

        Args:
            file_path: Path to the file
            expected_hash: Expected hash value
            hash_type: Hash algorithm ('md5', 'sha1', 'sha256')

        Returns:
            True if hash matches, False otherwise
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return False

        try:
            hash_func = getattr(hashlib, hash_type)()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_func.update(chunk)

            file_hash = hash_func.hexdigest()
            if file_hash == expected_hash:
                self.logger.info(
                    f"Hash verification successful for {file_path}")
                return True
            else:
                self.logger.warning(
                    f"Hash mismatch for {file_path}. Expected: {expected_hash}, Got: {file_hash}")
                return False

        except Exception as e:
            self.logger.error(f"Hash verification failed: {e}")
            return False

    def download_models(self, model_urls: Dict[str, str]) -> Dict[str, bool]:
        """
        Download multiple models from URLs.

        Args:
            model_urls: Dictionary mapping model names to URLs

        Returns:
            Dictionary with download status for each model
        """
        results = {}

        for model_name, url in model_urls.items():
            # Determine destination path
            if model_name in self.config.detection.model_paths:
                destination = self.config.detection.model_paths[model_name]
            else:
                # Default path in models directory
                extension = os.path.splitext(url)[1] or ".pt"
                destination = os.path.join(
                    self.models_dir, f"{model_name}{extension}")

            # Skip if file already exists
            if os.path.exists(destination):
                self.logger.info(
                    f"Model {model_name} already exists at {destination}")
                results[model_name] = True
                continue

            # Download model
            self.logger.info(f"Downloading model {model_name} from {url}")
            success = self.download_file(url, destination)
            results[model_name] = success

            # Add to config if successful
            if success and model_name not in self.config.detection.model_paths:
                self.config.detection.model_paths[model_name] = destination

        return results

    def get_pretrained_model_url(self, model_name: str) -> Optional[str]:
        """
        Get URL for pretrained model.

        Args:
            model_name: Name of the model

        Returns:
            URL for the pretrained model or None
        """
        # Pre-defined URLs for popular models
        model_urls = {
            # YOLOv8 models
            "yolov8n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "yolov8s": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
            "yolov8m": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
            "yolov8l": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
            "yolov8x": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",

            # YOLOv5 models
            "yolov5n": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
            "yolov5s": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
            "yolov5m": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
            "yolov5l": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
            "yolov5x": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt",

            # EfficientDet models
            "efficientdet-d0": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d0-d92fd44f.pth",
            "efficientdet-d1": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d1-4c7ebaf2.pth",
            "efficientdet-d2": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d2-cb4ce77d.pth",
        }

        return model_urls.get(model_name.lower())

    def download_pretrained_models(self, model_names: List[str]) -> Dict[str, bool]:
        """
        Download pretrained models by name.

        Args:
            model_names: List of model names to download

        Returns:
            Dictionary with download status for each model
        """
        model_urls = {}
        missing_models = []

        for name in model_names:
            url = self.get_pretrained_model_url(name)
            if url:
                model_urls[name] = url
            else:
                missing_models.append(name)
                self.logger.warning(f"No predefined URL for model: {name}")

        if missing_models:
            self.logger.warning(
                f"No URLs found for models: {', '.join(missing_models)}")

        return self.download_models(model_urls)

    def download_from_config(self) -> Dict[str, bool]:
        """
        Download all models specified in the configuration.

        Returns:
            Dictionary with download status for each model
        """
        model_paths = self.config.detection.model_paths
        models_to_download = []

        for model_name, path in model_paths.items():
            if not os.path.exists(path):
                models_to_download.append(model_name)

        if not models_to_download:
            self.logger.info("All configured models already exist locally")
            return {}

        self.logger.info(
            f"Downloading models: {', '.join(models_to_download)}")
        return self.download_pretrained_models(models_to_download)

    def check_and_download_missing_models(self) -> Dict[str, str]:
        """
        Check for missing models in the configuration and download them.

        Returns:
            Dictionary mapping model names to paths
        """
        model_paths = {}

        # Check each model in the configuration
        for model_name, model_path in self.config.detection.model_paths.items():
            # If the model path doesn't exist, try to download it
            if not os.path.exists(model_path):
                # Extract the base model name (remove extensions and paths)
                base_name = os.path.splitext(os.path.basename(model_path))[0]

                # Check if this is a known model we can download
                if base_name in self.MODEL_URLS:
                    self.logger.info(
                        f"Model {model_name} not found at {model_path}, attempting to download...")
                    downloaded_path = self.download_model(base_name)

                    if downloaded_path:
                        # Update the config with the new path
                        self.config.detection.model_paths[model_name] = downloaded_path
                        model_paths[model_name] = downloaded_path
                    else:
                        self.logger.warning(
                            f"Failed to download model {base_name}")
                else:
                    self.logger.warning(
                        f"Unknown model: {base_name}, cannot download automatically")
            else:
                # Model exists
                model_paths[model_name] = model_path

        # Save the updated config
        if self.config._config_file:
            self.config.save(self.config._config_file)

        return model_paths

    def download_all_available_models(self) -> Dict[str, str]:
        """
        Download all available models from the known URLs.

        Returns:
            Dictionary mapping model names to paths
        """
        model_paths = {}

        for model_name in self.MODEL_URLS:
            path = self.download_model(model_name)
            if path:
                model_paths[model_name] = path

        return model_paths

    def list_available_models(self) -> List[str]:
        """
        List all available models that can be downloaded.

        Returns:
            List of available model names
        """
        return list(self.MODEL_URLS.keys())

    def list_downloaded_models(self) -> Dict[str, str]:
        """
        List all downloaded models.

        Returns:
            Dictionary mapping model names to paths
        """
        downloaded = {}

        for model_name in self.MODEL_URLS:
            path = os.path.join(self.models_dir, f"{model_name}.pt")
            if os.path.exists(path):
                downloaded[model_name] = path

        return downloaded


def main():
    """Command-line interface for the model downloader."""
    import argparse

    parser = argparse.ArgumentParser(description="Download detection models")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--list", action="store_true",
                        help="List available models")
    parser.add_argument("--list-downloaded", action="store_true",
                        help="List downloaded models")
    parser.add_argument("--download", type=str, nargs="+",
                        help="Download specific models")
    parser.add_argument("--download-all", action="store_true",
                        help="Download all available models")
    parser.add_argument("--check-missing", action="store_true",
                        help="Check for missing models in config and download them")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if model exists")

    args = parser.parse_args()

    # Load config
    try:
        if not os.path.exists(args.config):
            print(f"Config file not found: {args.config}")
            print("Creating a default config...")
            config = Config()
        else:
            config = Config.from_yaml(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    downloader = ModelDownloader(config)

    if args.list:
        print("Available models:")
        for model in downloader.list_available_models():
            print(f"  - {model}")

    elif args.list_downloaded:
        print("Downloaded models:")
        for model, path in downloader.list_downloaded_models().items():
            print(f"  - {model}: {path}")

    elif args.download:
        for model in args.download:
            path = downloader.download_model(model, args.force)
            if path:
                print(f"Downloaded {model} to {path}")
            else:
                print(f"Failed to download {model}")

    elif args.download_all:
        print("Downloading all available models...")
        models = downloader.download_all_available_models()
        print(f"Downloaded {len(models)} models")

    elif args.check_missing:
        print("Checking for missing models in config...")
        models = downloader.check_and_download_missing_models()
        if models:
            print(f"Checked {len(models)} models:")
            for model, path in models.items():
                if os.path.exists(path):
                    print(f"  - {model}: {path} (Found)")
                else:
                    print(f"  - {model}: {path} (Missing)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
