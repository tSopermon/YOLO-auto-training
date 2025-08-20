"""
Constants for YOLO model training configurations.
Based on latest Ultralytics and Roboflow best practices.
"""

from pathlib import Path
from typing import Dict, Any

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
EXPORT_ROOT = PROJECT_ROOT / "exported_models"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
PRETRAINED_WEIGHTS_ROOT = PROJECT_ROOT / "pretrained_weights"

# Dataset configuration
DATASET_CONFIG = {
    "train_path": str(DATASET_ROOT / "train"),
    "valid_path": str(DATASET_ROOT / "valid"),
    "test_path": str(DATASET_ROOT / "test"),
    "data_yaml_path": str(DATASET_ROOT / "data.yaml"),
    "single_class": False,  # Single or multiple class training
    "rect": False,  # Rectangular training (recommended for validation only)
    "cache": False,  # Cache images for faster training
}

# Roboflow configuration
ROBOFLOW_CONFIG = {
    "workspace": "your-workspace",  # Change this
    "project": "your-project",  # Change this
    "version": 1,  # Change this
    "api_key": None,  # Will be loaded from environment variable ROBOFLOW_API_KEY
    "format": "yolov8",  # Export format
    "preprocessing": {
        "auto_orient": True,
        "resize": {"width": 640, "height": 640, "format": "squared"},  # or "fit"
    },
}

# Common training parameters (based on latest Ultralytics defaults)
COMMON_TRAINING = {
    "seed": 42,
    "epochs": 100,
    "patience": 50,  # Early stopping patience
    "batch_size": 8,  # Reduced for 1024x1024 images
    "image_size": 1024,  # Optimized for automotive parts detection
    "num_workers": 4,  # Reduced for smaller dataset
    "device": "cuda",  # or "cpu"
    "pretrained": True,
    "deterministic": True,  # Deterministic mode for reproducibility
    "single_cls": False,  # Single class mode
    "rect": False,  # Rectangular training
    "cos_lr": True,  # Cosine LR scheduler
    "close_mosaic": 10,  # Close mosaic augmentation last N epochs
    "resume": False,  # Resume training from last checkpoint
}

# Model-specific configurations (updated per latest Ultralytics docs)
YOLOV8_CONFIG = {
    "model_type": "yolov8",
    "weights": str(PRETRAINED_WEIGHTS_ROOT / "yolov8n.pt"),  # or s/m/l/x
    "learning_rate": 0.01,
    "optimizer": "auto",  # Ultralytics AutoOptimizer
    "lr_scheduler": "cosine",  # Cosine decay
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "box": 7.5,  # Box loss gain
    "cls": 0.5,  # Cls loss gain
    "dfl": 1.5,  # DFL loss gain
    "pose": 12.0,  # Pose loss gain
    "kobj": 1.0,  # Keypoint obj loss gain
    "label_smoothing": 0.0,
    "nbs": 64,  # Nominal batch size
    "overlap_mask": True,  # Masks should overlap during training
    "mask_ratio": 4,  # Mask downsample ratio
    "dropout": 0.0,  # Use dropout regularization
    "val": True,  # Validate during training
}

YOLOV5_CONFIG = {
    "model_type": "yolov5",
    "weights": str(PRETRAINED_WEIGHTS_ROOT / "yolov5nu.pt"),  # or n/m/l/x
    "learning_rate": 0.01,
    "optimizer": "SGD",
    "lr_scheduler": "cosine",
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 0.05,
    "cls": 0.5,
    "cls_pw": 1.0,
    "obj": 1.0,
    "obj_pw": 1.0,
    "fl_gamma": 0.0,
    "label_smoothing": 0.0,
    "nbs": 64,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
}

# Latest YOLO version configuration
YOLO11_CONFIG = {
    "model_type": "yolo11",
    "weights": str(PRETRAINED_WEIGHTS_ROOT / "yolo11n.pt"),  # or s/m/l/x
    "learning_rate": 0.01,
    "optimizer": "auto",
    "lr_scheduler": "cosine",
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "label_smoothing": 0.0,
    "nbs": 64,
    "val": True,
}

# Augmentation settings (updated per latest recommendations)
AUGMENTATION_CONFIG = {
    # Mosaic and Mixup
    "mosaic": 1.0,  # Mosaic probability
    "mixup": 0.0,  # Mixup probability
    "copy_paste": 0.0,  # Copy-paste probability
    # HSV augmentation
    "hsv_h": 0.015,  # Hue gain
    "hsv_s": 0.7,  # Saturation gain
    "hsv_v": 0.4,  # Value gain
    # Geometric transformations
    "degrees": 0.0,  # Rotation (+/- deg)
    "translate": 0.1,  # Translation (+/- fraction)
    "scale": 0.5,  # Scale (+/- gain)
    "shear": 0.0,  # Shear (+/- deg)
    "perspective": 0.0,  # Perspective (+/- fraction)
    "flipud": 0.0,  # Vertical flip probability
    "fliplr": 0.5,  # Horizontal flip probability
    # Other augmentations
    "blur": 0.0,  # Blur probability
    "blur_sigma": 0.1,  # Blur sigma
    "grayscale": 0.0,  # Grayscale probability
    "auto_augment": False,  # Use AutoAugment
}

# Evaluation settings (updated per latest metrics)
EVAL_CONFIG = {
    "conf_thres": 0.001,  # Confidence threshold
    "iou_thres": 0.6,  # NMS IoU threshold
    "max_det": 300,  # Maximum detections per image
    "half": True,  # Use FP16 half-precision inference
    "metrics": [
        "mAP50",  # mAP at IoU=50%
        "mAP50-95",  # mAP at IoU=50-95%
        "precision",  # Precision
        "recall",  # Recall
        "f1",  # F1 score
        "speed",  # Inference speed
    ],
    "plots": True,  # Generate plots during validation
    "save_json": False,  # Save results to JSON file
    "save_hybrid": False,  # Save hybrid version of labels
    "save_conf": False,  # Save confidences in --save-txt labels
}

# Logging configuration (with TensorBoard and Weights & Biases support)
LOGGING_CONFIG = {
    "log_dir": str(PROJECT_ROOT / "logs"),
    "project_name": "yolo_training",  # Default project name for training results
    "tensorboard": True,  # Use TensorBoard logging
    "wandb": False,  # Use Weights & Biases logging
    "wandb_project": "yolo_training",  # W&B project name
    "log_metrics_interval": 20,  # Log metrics every N iterations
    "save_checkpoint_interval": 10,  # Save checkpoint every N epochs
    "num_checkpoint_keep": 5,  # Number of checkpoints to keep
    "save_period": -1,  # Save checkpoint every N epochs (-1 to disable)
}

# Export configuration (updated with latest formats)
EXPORT_CONFIG = {
    "export_formats": [
        "onnx",  # ONNX format
        "torchscript",  # TorchScript format
        "openvino",  # OpenVINO format
        "coreml",  # CoreML format
        "tensorrt",  # TensorRT format
    ],
    "export_dir": str(EXPORT_ROOT),
    "include_nms": True,  # Include NMS in exported model
    "batch_size": 1,  # Export batch size
    "half": True,  # FP16 quantization
    "int8": False,  # INT8 quantization
    "simplify": True,  # ONNX simplification
    "dynamic": False,  # Dynamic axes
}
