import os
import yaml
import json
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
import logging
from dotenv import load_dotenv

# Load .env file at module level
load_dotenv()


@dataclass
class SamplingWeights:
    historical: float = 0.4
    environmental: float = 0.3
    infrastructure: float = 0.3


@dataclass
class APIConfig:
    key: str = os.getenv("GOOGLE_API_KEY")
    min_request_interval: float = 0.1
    max_retries: int = 3
    retry_delay: float = 1.0
    image_width: int = 1024
    image_height: int = 1024


@dataclass
class SamplingConfig:
    base_interval: float = 20.0  # meters
    max_extra_points: int = 5
    importance_threshold: float = 0.6
    weights: SamplingWeights = field(default_factory=SamplingWeights)


@dataclass
class DetectionConfig:
    model_paths: Dict[str, str] = field(default_factory=dict)
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    device: str = "cuda"
    batch_size: int = 8


@dataclass
class CameraConfig:
    altitude: float = 2.5  # meters
    pitch_values: List[int] = field(default_factory=lambda: [0, 30, 60])
    fov_default: List[int] = field(default_factory=lambda: [55])
    fov_special: List[int] = field(default_factory=lambda: [40])
    relative_angles_right: List[int] = field(
        default_factory=lambda: [90, 45, 135])
    relative_angles_left: List[int] = field(
        default_factory=lambda: [270, 310, 230])


@dataclass
class PowerlineConfig:
    offset_distance: float = 10.0  # meters
    side: str = "both"  # "right" or "left" or "both"


@dataclass
class CacheConfig:
    enabled: bool = True
    location: str = ".cache"
    max_size_mb: int = 5000  # 5GB max cache size
    expire_after: int = 604800  # 1 week in seconds


@dataclass
class SystemConfig:
    output_dir: str = "output"
    num_workers: int = 4
    debug: bool = False
    cache_dir: str = ".cache"
    config_check_interval: int = 300  # seconds between checking for config changes


@dataclass
class Config:
    api: APIConfig = field(default_factory=APIConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    powerline: PowerlineConfig = field(default_factory=PowerlineConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Tracking config file and last modification time for hot reloading
    _config_file: str = None
    _last_modified: float = 0
    _logger = None

    def __post_init__(self):
        self._apply_environment_variables()

    def _get_logger(self):
        if self._logger is None:
            self._logger = logging.getLogger("powerline_detector.config")
        return self._logger

    def _apply_environment_variables(self):
        """Apply environment variables to override configuration values."""
        # API configuration
        self.api.key = os.environ.get("GOOGLE_API_KEY", self.api.key)

        if not self.api.key:
            self.api.key = os.environ.get("POWERLINE_API_KEY", "")
        if not self.api.key:
            self.api.key = os.environ.get("GOOGLE_API_KEY", "")

        # System configuration
        self.system.output_dir = os.environ.get(
            "POWERLINE_OUTPUT_DIR", self.system.output_dir)
        self.system.debug = os.environ.get(
            "POWERLINE_DEBUG", "").lower() == "true" or self.system.debug

        # Cache configuration
        self.cache.enabled = os.environ.get(
            "POWERLINE_CACHE_ENABLED", "").lower() != "false" and self.cache.enabled
        self.cache.location = os.environ.get(
            "POWERLINE_CACHE_DIR", self.cache.location)

        # Detection configuration
        self.detection.device = os.environ.get(
            "POWERLINE_DEVICE", self.detection.device)

        # Convert string to float for thresholds if defined in environment
        conf_threshold = os.environ.get("POWERLINE_CONFIDENCE_THRESHOLD")
        if conf_threshold:
            try:
                self.detection.confidence_threshold = float(conf_threshold)
            except ValueError:
                self._get_logger().warning(
                    f"Invalid confidence threshold in environment: {conf_threshold}")

        # Detection model paths from environment variables
        model_paths_str = os.environ.get("POWERLINE_MODEL_PATHS")
        if model_paths_str:
            try:
                # Format: "model1:path1,model2:path2"
                pairs = [pair.split(":")
                         for pair in model_paths_str.split(",")]
                for model, path in pairs:
                    self.detection.model_paths[model.strip()] = path.strip()
            except Exception as e:
                self._get_logger().warning(
                    f"Failed to parse model paths from environment: {e}")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        config = cls.from_dict(config_dict)
        # Store file path and modification time for hot reloading
        config._config_file = yaml_path
        config._last_modified = os.path.getmtime(yaml_path)
        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary."""
        config = cls()

        # API config
        if "api" in config_dict:
            config.api = APIConfig(**config_dict["api"])

        # Sampling config
        if "sampling" in config_dict:
            sampling_dict = config_dict["sampling"].copy()
            if "weights" in sampling_dict:
                sampling_dict["weights"] = SamplingWeights(
                    **sampling_dict["weights"])
            config.sampling = SamplingConfig(**sampling_dict)

        # Detection config
        if "detection" in config_dict:
            config.detection = DetectionConfig(**config_dict["detection"])

        # Camera config
        if "camera" in config_dict:
            config.camera = CameraConfig(**config_dict["camera"])

        # Powerline config
        if "powerline" in config_dict:
            config.powerline = PowerlineConfig(**config_dict["powerline"])

        # System config
        if "system" in config_dict:
            config.system = SystemConfig(**config_dict["system"])

        # Cache config
        if "cache" in config_dict:
            config.cache = CacheConfig(**config_dict["cache"])

        # Apply environment variable overrides
        config._apply_environment_variables()
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "api": asdict(self.api),
            "sampling": {**asdict(self.sampling), "weights": asdict(self.sampling.weights)},
            "detection": asdict(self.detection),
            "camera": asdict(self.camera),
            "powerline": asdict(self.powerline),
            "system": asdict(self.system),
            "cache": asdict(self.cache),
        }

    def save(self, output_path: str) -> None:
        """Save configuration to a file (YAML or JSON)."""
        config_dict = self.to_dict()

        if output_path.endswith(".yaml") or output_path.endswith(".yml"):
            with open(output_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif output_path.endswith(".json"):
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file extension: {output_path}")

        # Update tracking for hot reloading
        if output_path == self._config_file:
            self._last_modified = os.path.getmtime(output_path)

    def validate(self) -> List[str]:
        """Validate the configuration and return a list of validation errors."""
        errors = []

        # API validation
        if not self.api.key:
            errors.append("API key is required")

        # Sampling validation
        if self.sampling.base_interval <= 0:
            errors.append("Sampling base_interval must be positive")
        if not 0 <= self.sampling.importance_threshold <= 1:
            errors.append(
                "Sampling importance_threshold must be between 0 and 1")

        weights_sum = (
            self.sampling.weights.historical +
            self.sampling.weights.environmental +
            self.sampling.weights.infrastructure
        )
        if abs(weights_sum - 1.0) > 0.001:
            errors.append(
                f"Sampling weights must sum to 1.0, got {weights_sum}")

        # Detection validation
        if self.detection.confidence_threshold < 0 or self.detection.confidence_threshold > 1:
            errors.append(
                "Detection confidence_threshold must be between 0 and 1")
        if self.detection.iou_threshold < 0 or self.detection.iou_threshold > 1:
            errors.append("Detection iou_threshold must be between 0 and 1")

        # Powerline validation
        if self.powerline.side not in ["right", "left", "both"]:
            errors.append("Powerline side must be 'right', 'left', or 'both'")

        # Cache validation
        if self.cache.max_size_mb <= 0:
            errors.append("Cache max_size_mb must be positive")

        return errors

    def create_output_dirs(self) -> None:
        """Create output directories specified in the configuration."""
        os.makedirs(self.system.output_dir, exist_ok=True)
        os.makedirs(os.path.join(
            self.system.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.system.output_dir,
                    "detections"), exist_ok=True)
        os.makedirs(os.path.join(
            self.system.output_dir, "reports"), exist_ok=True)
        os.makedirs(self.cache.location, exist_ok=True)

    def check_for_updates(self) -> bool:
        """
        Check if the config file has been modified since last load.
        Returns True if config was reloaded.
        """
        if not self._config_file:
            return False

        try:
            current_mtime = os.path.getmtime(self._config_file)
            if current_mtime > self._last_modified:
                # Config file has changed, reload it
                logger = self._get_logger()
                logger.info(
                    f"Configuration file {self._config_file} has changed, reloading...")

                # Load the new configuration
                with open(self._config_file, "r") as f:
                    config_dict = yaml.safe_load(f)

                # Update this instance
                new_config = self.from_dict(config_dict)

                # Copy all fields from new_config to self
                for key, value in new_config.__dict__.items():
                    if not key.startswith("_"):  # Skip private attributes
                        setattr(self, key, value)

                # Update modification time
                self._last_modified = current_mtime
                logger.info("Configuration reloaded successfully")
                return True
        except Exception as e:
            self._get_logger().error(f"Error checking for config updates: {e}")

        return False

    def get(self, path: str, default: Any = None) -> Any:
        """
        Access nested configuration values using dot notation with a default value.
        
        Args:
            path: Path to configuration value using dot notation (e.g., 'detection.confidence_threshold')
            default: Default value to return if the path doesn't exist
            
        Returns:
            The configuration value at the specified path, or the default if not found
        """
        parts = path.split('.')
        current = self
        
        try:
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        except Exception:
            return default
            
    def set(self, path: str, value: Any) -> None:
        """
        Set a configuration value at the specified path using dot notation.
        Creates the path if it doesn't exist and it's a simple attribute path.
        
        Args:
            path: Path to configuration value using dot notation (e.g., 'detection.confidence_threshold')
            value: Value to set
        """
        parts = path.split('.')
        current = self
        
        # Navigate to the parent object
        for i in range(len(parts) - 1):
            part = parts[i]
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                self._get_logger().warning(
                    f"Cannot set config value at {path}: {part} not found")
                return
        
        # Set the attribute on the parent object
        last_part = parts[-1]
        if hasattr(current, last_part):
            setattr(current, last_part, value)
        else:
            self._get_logger().warning(
                f"Cannot set config value at {path}: {last_part} not found")


# Create default configuration
default_config = Config()
