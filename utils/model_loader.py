"""
Model loading utilities for YOLO training.
Handles loading different YOLO versions and managing checkpoints.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
import torch
import torch.nn as nn

from .checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


def load_yolo_model(
    config: "YOLOConfig",
    checkpoint_manager: CheckpointManager,
    resume_path: Optional[str] = None,
) -> nn.Module:
    """
    Load YOLO model for training.

    Args:
        config: YOLO configuration
        checkpoint_manager: Checkpoint manager instance
        resume_path: Path to checkpoint to resume from

    Returns:
        Loaded YOLO model
    """
    model = None

    # Try to resume from checkpoint first
    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        model = _load_from_checkpoint(resume_path, config)
    else:
        # Try to load latest checkpoint
        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            logger.info(f"Loading latest checkpoint: {latest_checkpoint}")
            model = _load_from_checkpoint(latest_checkpoint, config)

    # If no checkpoint found, create new model
    if model is None:
        logger.info("Creating new YOLO model")
        model = _create_new_model(config)

    # Move model to device
    device = torch.device(config.device)
    model = model.to(device)

    # Log model information
    _log_model_info(model, config)

    return model


def _load_from_checkpoint(
    checkpoint_path: Union[str, Path], config: "YOLOConfig"
) -> Optional[nn.Module]:
    """Load model from checkpoint file."""
    try:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract model state
        if "model" in checkpoint:
            model_state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            model_state = checkpoint["state_dict"]
        else:
            logger.warning("Checkpoint format not recognized")
            return None

        # Create model and load state
        model = _create_new_model(config)
        model.load_state_dict(model_state)

        logger.info(f"Successfully loaded model from checkpoint: {checkpoint_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def _create_new_model(config: "YOLOConfig") -> nn.Module:
    """Create new YOLO model based on configuration."""
    model_type = config.model_type

    if model_type == "yolov8":
        return _create_yolov8_model(config)
    elif model_type == "yolov5":
        return _create_yolov5_model(config)
    elif model_type == "yolo11":
        return _create_yolo11_model(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _create_yolov8_model(config: "YOLOConfig") -> nn.Module:
    """Create YOLOv8 model using Ultralytics."""
    try:
        from ultralytics import YOLO

        # Check if weights file exists locally
        weights_path = Path(config.weights)
        if not weights_path.exists():
            logger.info(f"Weights file not found: {weights_path}")
            logger.info("Automatically downloading weights...")

            # Extract model type and size from weights path
            weights_name = weights_path.name
            if "yolov8" in weights_name:
                model_type = "yolov8"
                size = weights_name.replace("yolov8", "").replace(".pt", "")
            else:
                raise ValueError(f"Unknown weights format: {weights_name}")

            # Download weights automatically
            from .download_pretrained_weights import download_weights

            weights_dir = Path("pretrained_weights")
            weights_dir.mkdir(exist_ok=True)

            if download_weights(model_type, size, weights_dir):
                logger.info(f"✅ Weights downloaded successfully: {weights_path}")
            else:
                raise FileNotFoundError(f"Failed to download weights: {weights_path}")

        # Use local weights file
        model = YOLO(str(weights_path))
        logger.info(f"Created YOLOv8 model using local weights: {weights_path}")
        return model.model  # Return the underlying PyTorch model

    except ImportError:
        logger.error("Ultralytics not installed. Install with: pip install ultralytics")
        raise
    except Exception as e:
        logger.error(f"Failed to create YOLOv8 model: {e}")
        raise


def _create_yolov5_model(config: "YOLOConfig") -> nn.Module:
    """Create YOLOv5 model."""
    try:
        # Try to import from ultralytics first
        from ultralytics import YOLO

        # Check if weights file exists locally
        weights_path = Path(config.weights)
        if not weights_path.exists():
            logger.info(f"Weights file not found: {weights_path}")
            logger.info("Automatically downloading weights...")

            # Extract model type and size from weights path
            weights_name = weights_path.name
            if "yolov5" in weights_name:
                model_type = "yolov5"
                size = weights_name.replace("yolov5", "").replace(".pt", "")
            elif "yolov8" in weights_name:
                model_type = "yolov8"
                size = weights_name.replace("yolov8", "").replace(".pt", "")
            elif "yolo11" in weights_name:
                model_type = "yolo11"
                size = weights_name.replace("yolo11", "").replace(".pt", "")
            else:
                raise ValueError(f"Unknown weights format: {weights_name}")

            # Download weights automatically
            from .download_pretrained_weights import download_weights

            weights_dir = Path("pretrained_weights")
            weights_dir.mkdir(exist_ok=True)

            if download_weights(model_type, size, weights_dir):
                logger.info(f"✅ Weights downloaded successfully: {weights_path}")
            else:
                raise FileNotFoundError(f"Failed to download weights: {weights_path}")

        # Use local weights file
        model = YOLO(str(weights_path))
        logger.info(f"Created YOLOv5 model using local weights: {weights_path}")
        return model.model

    except ImportError:
        # Fallback to YOLOv5 repository
        logger.info("Ultralytics not available, trying YOLOv5 repository...")
        return _create_yolov5_from_repo(config)


def _create_yolov5_from_repo(config: "YOLOConfig") -> nn.Module:
    """Create YOLOv5 model from repository."""
    try:
        # This would require the YOLOv5 repository to be cloned
        # For now, we'll raise an error and provide instructions
        raise ImportError(
            "YOLOv5 repository not found. Please clone it first:\n"
            "git clone https://github.com/ultralytics/yolov5\n"
            "cd yolov5 && pip install -r requirements.txt"
        )
    except Exception as e:
        logger.error(f"Failed to create YOLOv5 model from repository: {e}")
        raise


def _create_yolo11_model(config: "YOLOConfig") -> nn.Module:
    """Create YOLO11 model."""
    try:
        # Try to import from ultralytics first
        from ultralytics import YOLO

        # Check if weights file exists locally
        weights_path = Path(config.weights)
        if not weights_path.exists():
            logger.info(f"Weights file not found: {weights_path}")
            logger.info("Automatically downloading weights...")

            # Extract model type and size from weights path
            weights_name = weights_path.name
            if "yolo11" in weights_name:
                model_type = "yolo11"
                size = weights_name.replace("yolo11", "").replace(".pt", "")
            else:
                raise ValueError(f"Unknown weights format: {weights_name}")

            # Download weights automatically
            from .download_pretrained_weights import download_weights

            weights_dir = Path("pretrained_weights")
            weights_dir.mkdir(exist_ok=True)

            if download_weights(model_type, size, weights_dir):
                logger.info(f"✅ Weights downloaded successfully: {weights_path}")
            else:
                raise FileNotFoundError(f"Failed to download weights: {weights_path}")

        # Use local weights file
        model = YOLO(str(weights_path))
        logger.info(f"Created YOLO11 model using local weights: {weights_path}")
        return model.model

    except ImportError:
        # Fallback to YOLO11 repository
        logger.info("Ultralytics not available, trying YOLO11 repository...")
        return _create_yolo11_from_repo(config)


def _create_yolo11_from_repo(config: "YOLOConfig") -> nn.Module:
    """Create YOLO11 model from repository."""
    try:
        # This would require the YOLO11 repository to be cloned
        # For now, we'll raise an error and provide instructions
        raise ImportError(
            "YOLO11 repository not found. Please clone it first:\n"
            "git clone https://github.com/ultralytics/yolo11\n"
            "cd yolo11 && pip install -r requirements.txt"
        )
    except Exception as e:
        logger.error(f"Failed to create YOLO11 model from repository: {e}")
        raise


def _log_model_info(model: nn.Module, config: "YOLOConfig") -> None:
    """Log model information."""
    device = next(model.parameters()).device
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=" * 50)
    logger.info("Model Information")
    logger.info("=" * 50)
    logger.info(f"Model Type: {config.model_type}")
    logger.info(f"Device: {device}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Model Architecture: {type(model).__name__}")

    # Log model structure summary
    if hasattr(model, "named_modules"):
        logger.info("Model Structure:")
        for name, module in list(model.named_modules())[:10]:  # Show first 10 modules
            logger.info(f"  {name}: {type(module).__name__}")
        if len(list(model.named_modules())) > 10:
            logger.info(
                f"  ... and {len(list(model.named_modules())) - 10} more modules"
            )

    logger.info("=" * 50)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_manager: CheckpointManager,
    config: "YOLOConfig",
) -> Path:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training metrics
        checkpoint_manager: Checkpoint manager
        config: Training configuration

    Returns:
        Path to saved checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "model_type": config.model_type,
            "image_size": config.image_size,
            "batch_size": config.batch_size,
        },
    }

    checkpoint_path = checkpoint_manager.save_checkpoint(checkpoint, epoch, metrics)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def load_optimizer_state(
    checkpoint_path: Union[str, Path],
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load optimizer state from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        optimizer: Optimizer to load state into

    Returns:
        Epoch number from checkpoint
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"Loaded optimizer state from checkpoint")

        epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resuming from epoch {epoch}")

        return epoch

    except Exception as e:
        logger.error(f"Failed to load optimizer state: {e}")
        return 0
