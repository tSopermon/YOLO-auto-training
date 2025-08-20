"""
Configuration package for YOLO training.
Provides configuration management and constants.
"""

from .config import YOLOConfig, get_config, setup_logging

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

__all__ = [
    # Configuration classes
    "YOLOConfig",
    "get_config",
    "setup_logging",
    # Configuration constants
    "COMMON_TRAINING",
    "DATASET_CONFIG",
    "ROBOFLOW_CONFIG",
    "YOLO11_CONFIG",
    "YOLOV8_CONFIG",
    "YOLOV5_CONFIG",
    "AUGMENTATION_CONFIG",
    "EVAL_CONFIG",
    "LOGGING_CONFIG",
    "EXPORT_CONFIG",
    "PROJECT_ROOT",
]
