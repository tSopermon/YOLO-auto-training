"""
Configuration manager for YOLO model training.
Loads and validates configurations from constants.py.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import torch
from dataclasses import dataclass, field
import yaml
import json

from .constants import (
    COMMON_TRAINING,
    DATASET_CONFIG,
    ROBOFLOW_CONFIG,
    YOLO11_CONFIG,
    YOLOV8_CONFIG,
    YOLOV5_CONFIG,
    AUGMENTATION_CONFIG,
    EVAL_CONFIG,
    LOGGING_CONFIG,
    EXPORT_CONFIG,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


@dataclass
class YOLOConfig:
    """Configuration class for YOLO training."""

    # Model configuration
    model_type: str
    weights: str
    pretrained: bool = True

    # Training configuration
    epochs: int = field(default_factory=lambda: COMMON_TRAINING["epochs"])
    batch_size: int = field(default_factory=lambda: COMMON_TRAINING["batch_size"])
    image_size: int = field(default_factory=lambda: COMMON_TRAINING["image_size"])
    num_workers: int = field(default_factory=lambda: COMMON_TRAINING["num_workers"])
    device: str = field(default_factory=lambda: COMMON_TRAINING["device"])
    seed: int = field(default_factory=lambda: COMMON_TRAINING["seed"])
    patience: int = field(default_factory=lambda: COMMON_TRAINING["patience"])
    deterministic: bool = field(
        default_factory=lambda: COMMON_TRAINING["deterministic"]
    )
    single_cls: bool = field(default_factory=lambda: COMMON_TRAINING["single_cls"])
    rect: bool = field(default_factory=lambda: COMMON_TRAINING["rect"])
    cos_lr: bool = field(default_factory=lambda: COMMON_TRAINING["cos_lr"])
    close_mosaic: int = field(default_factory=lambda: COMMON_TRAINING["close_mosaic"])
    resume: bool = field(default_factory=lambda: COMMON_TRAINING["resume"])

    # Model-specific configuration
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Dataset configuration
    dataset_config: Dict[str, Any] = field(default_factory=lambda: DATASET_CONFIG)

    # Augmentation configuration
    augmentation_config: Dict[str, Any] = field(
        default_factory=lambda: AUGMENTATION_CONFIG
    )

    # Evaluation configuration
    eval_config: Dict[str, Any] = field(default_factory=lambda: EVAL_CONFIG)

    # Logging configuration
    logging_config: Dict[str, Any] = field(default_factory=lambda: LOGGING_CONFIG)

    # Export configuration
    export_config: Dict[str, Any] = field(default_factory=lambda: EXPORT_CONFIG)

    def __post_init__(self):
        """Validate and set up configuration after initialization."""
        self._validate_config()
        self._setup_paths()
        self._setup_device()
        # Don't load data.yaml immediately - it will be loaded after dataset preparation
        # self._load_data_yaml()
        self._setup_wandb()

    def _validate_config(self):
        """Validate configuration values."""
        if self.model_type not in ["yolo11", "yolov8", "yolov5"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if not isinstance(self.image_size, (int, list, tuple)):
            raise ValueError(f"Invalid image size: {self.image_size}")

        if self.batch_size < 1:
            raise ValueError(f"Invalid batch size: {self.batch_size}")

        if self.patience < 0:
            raise ValueError(f"Invalid patience value: {self.patience}")

        # Validate export formats
        valid_formats = {"onnx", "torchscript", "openvino", "coreml", "tensorrt"}
        invalid_formats = set(self.export_config["export_formats"]) - valid_formats
        if invalid_formats:
            raise ValueError(f"Invalid export formats: {invalid_formats}")

        # Validate dataset paths exist
        for path_key in ["train_path", "valid_path", "test_path"]:
            path = Path(self.dataset_config[path_key])
            if not path.exists():
                logger.warning(f"Dataset path does not exist: {path}")

    def _setup_paths(self):
        """Set up and validate paths."""
        # Create necessary directories
        for path in [
            Path(self.logging_config["log_dir"]),
            Path(self.export_config["export_dir"]),
            Path(self.dataset_config["train_path"]),
            Path(self.dataset_config["valid_path"]),
            Path(self.dataset_config["test_path"]),
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _setup_device(self):
        """Set up and validate device configuration."""
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        if self.device == "cuda":
            # Log GPU information
            gpu_count = torch.cuda.device_count()
            logger.info(f"Using GPU(s): {gpu_count} device(s) available")
            for i in range(gpu_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(
                    f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
                )

    def _setup_wandb(self):
        """Set up Weights & Biases logging if enabled."""
        if self.logging_config.get("wandb", False):
            try:
                import wandb

                if not wandb.api.api_key:
                    logger.warning(
                        "W&B enabled but API key not found. Disabling W&B logging."
                    )
                    self.logging_config["wandb"] = False
            except ImportError:
                logger.warning(
                    "W&B enabled but package not installed. Disabling W&B logging."
                )
                self.logging_config["wandb"] = False

    def _load_data_yaml(self):
        """Load and validate data.yaml configuration."""
        yaml_path = Path(self.dataset_config["data_yaml_path"])
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {yaml_path}")

        with open(yaml_path) as f:
            self.data_yaml = yaml.safe_load(f)

        required_keys = ["path", "train", "val", "nc", "names"]
        if not all(key in self.data_yaml for key in required_keys):
            raise ValueError(f"data.yaml missing required keys: {required_keys}")

        # Validate class names
        if not isinstance(self.data_yaml["names"], (list, dict)):
            raise ValueError("Class names must be a list or dict")

    def load_data_yaml(self):
        """Load data.yaml configuration after dataset preparation."""
        self._load_data_yaml()

    @classmethod
    def create(cls, model_type: str) -> "YOLOConfig":
        """Factory method to create configuration for specific YOLO version."""
        if model_type == "yolo11":
            model_config = YOLO11_CONFIG
        elif model_type == "yolov8":
            model_config = YOLOV8_CONFIG
        elif model_type == "yolov5":
            model_config = YOLOV5_CONFIG
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return cls(
            model_type=model_type,
            weights=model_config["weights"],
            model_config=model_config,
        )

    def save(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file."""
        if save_path is None:
            save_path = Path(self.logging_config["log_dir"]) / "config.yaml"

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary
        config_dict = {
            "model_type": self.model_type,
            "weights": self.weights,
            "pretrained": self.pretrained,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "device": self.device,
            "seed": self.seed,
            "patience": self.patience,
            "deterministic": self.deterministic,
            "single_cls": self.single_cls,
            "rect": self.rect,
            "cos_lr": self.cos_lr,
            "close_mosaic": self.close_mosaic,
            "resume": self.resume,
            "model_config": self.model_config,
            "dataset_config": self.dataset_config,
            "augmentation_config": self.augmentation_config,
            "eval_config": self.eval_config,
            "logging_config": self.logging_config,
            "export_config": self.export_config,
        }

        # Save as YAML
        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # Also save as JSON for better compatibility
        json_path = save_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=4)

        logger.info(f"Configuration saved to {save_path} and {json_path}")

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> "YOLOConfig":
        """Load configuration from file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Support both YAML and JSON formats
        if config_path.suffix == ".json":
            with open(config_path) as f:
                config_dict = json.load(f)
        else:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration based on model type."""
        return {
            "optimizer": self.model_config["optimizer"],
            "lr": self.model_config["learning_rate"],
            "momentum": self.model_config.get("warmup_momentum", 0.937),
            "weight_decay": self.model_config.get("weight_decay", 0.0005),
        }

    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get learning rate scheduler configuration."""
        return {
            "scheduler": self.model_config["lr_scheduler"],
            "warmup_epochs": self.model_config["warmup_epochs"],
            "warmup_momentum": self.model_config["warmup_momentum"],
            "warmup_bias_lr": self.model_config.get("warmup_bias_lr", 0.1),
        }

    def get_training_args(self) -> Dict[str, Any]:
        """Get all training arguments for YOLO training."""
        return {
            # Basic training
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.image_size,
            "device": self.device,
            "workers": self.num_workers,
            "patience": self.patience,
            "deterministic": self.deterministic,
            "single_cls": self.single_cls,
            "rect": self.rect,
            "cos_lr": self.cos_lr,
            "close_mosaic": self.close_mosaic,
            "resume": self.resume,
            # Model specific
            "weights": self.weights,
            "pretrained": self.pretrained,
            # Data
            "data": str(self.dataset_config["data_yaml_path"]),
            "cache": self.dataset_config.get("cache", False),
            # Augmentation
            **{k: v for k, v in self.augmentation_config.items() if v > 0},
            # Evaluation
            "val": self.eval_config.get("plots", True),
            "save_period": self.logging_config.get("save_period", -1),
        }


def get_config(model_type: str = "yolov8") -> YOLOConfig:
    """
    Get configuration for specified YOLO version.

    Args:
        model_type: One of ["yolo11", "yolov8", "yolov5"]

    Returns:
        YOLOConfig instance
    """
    return YOLOConfig.create(model_type)


def setup_logging(config: YOLOConfig) -> None:
    """Set up logging configuration."""
    log_dir = Path(config.logging_config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create rotating file handler
    from logging.handlers import RotatingFileHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                log_dir / "training.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            ),
            logging.StreamHandler(),
        ],
    )


# Example usage:
if __name__ == "__main__":
    # Create configuration for YOLOv8
    config = get_config("yolov8")

    # Set up logging
    setup_logging(config)

    # Save configuration
    config.save()

    # Log some information
    logger.info(f"Using model: {config.model_type}")
    logger.info(f"Training device: {config.device}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Image size: {config.image_size}")
    logger.info(f"Optimizer config: {config.get_optimizer_config()}")
    logger.info(f"Scheduler config: {config.get_scheduler_config()}")
    logger.info(f"Training args: {config.get_training_args()}")
