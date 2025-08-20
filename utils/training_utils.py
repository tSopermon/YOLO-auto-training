"""
Training utilities for YOLO models.
Handles training loop, validation, and optimization.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from .data_loader import create_dataloader
from .model_loader import save_checkpoint, load_optimizer_state
from .checkpoint_manager import CheckpointManager
from .training_monitor import TrainingMonitor

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    config: "YOLOConfig",
    checkpoint_manager: CheckpointManager,
    monitor: TrainingMonitor,
) -> Dict[str, Any]:
    """
    Train YOLO model.

    Args:
        model: Model to train
        config: Training configuration
        checkpoint_manager: Checkpoint manager
        monitor: Training monitor

    Returns:
        Training results and metrics
    """
    logger.info("Starting model training...")

    # Set up data loaders
    train_loader = create_dataloader(config, split="train", augment=True, shuffle=True)
    val_loader = create_dataloader(config, split="valid", augment=False, shuffle=False)

    # Set up optimizer and scheduler
    optimizer = _create_optimizer(model, config)
    scheduler = _create_scheduler(optimizer, config)

    # Set up loss function
    criterion = _create_loss_function(config)

    # Training state
    device = torch.device(config.device)
    model = model.to(device)
    start_epoch = 0

    # Try to resume from checkpoint
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    if latest_checkpoint:
        checkpoint_data = checkpoint_manager.resume_training(latest_checkpoint)
        if checkpoint_data:
            start_epoch = load_optimizer_state(latest_checkpoint, optimizer)
            logger.info(f"Resuming training from epoch {start_epoch}")

    # Training loop
    best_metric = float("inf")  # Lower is better for loss
    training_history = []

    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()

        # Training phase
        train_metrics = _train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, config
        )

        # Validation phase
        val_metrics = _validate_epoch(
            model, val_loader, criterion, device, epoch, config
        )

        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics.get("val_loss", float("inf")))
            else:
                scheduler.step()

        # Combine metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_metrics["val_loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time() - epoch_start_time,
            **train_metrics,
            **val_metrics,
        }

        # Log metrics
        monitor.log_metrics(epoch_metrics, epoch)

        # Save checkpoint
        is_best = val_metrics.get("val_loss", float("inf")) < best_metric
        if is_best:
            best_metric = val_metrics.get("val_loss", float("inf"))

        save_checkpoint(
            model, optimizer, epoch, epoch_metrics, checkpoint_manager, config
        )

        # Log epoch summary
        logger.info(
            f"Epoch {epoch:3d}/{config.epochs}: "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
            f"Time: {epoch_metrics['epoch_time']:.2f}s"
        )

        training_history.append(epoch_metrics)

        # Early stopping check
        if _should_stop_early(training_history, config.patience):
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Final validation
    final_metrics = _validate_epoch(
        model, val_loader, criterion, device, config.epochs, config
    )

    # Training summary
    training_results = {
        "total_epochs": len(training_history),
        "best_epoch": np.argmin([m["val_loss"] for m in training_history]),
        "best_val_loss": best_metric,
        "final_metrics": final_metrics,
        "training_history": training_history,
        "config": config,
    }

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_metric:.4f}")
    logger.info(f"Total epochs trained: {len(training_history)}")

    return training_results


def _train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: "YOLOConfig",
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = len(train_loader)

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, (images, labels, paths, shapes) in enumerate(pbar):
        # Move data to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if hasattr(config, "max_grad_norm") and config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()

        # Update metrics
        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}",
            }
        )

        # Log batch metrics
        if batch_idx % config.logging_config.get("log_metrics_interval", 20) == 0:
            logger.debug(
                f"Epoch {epoch}, Batch {batch_idx}/{num_batches}: "
                f"Loss: {loss.item():.4f}"
            )

    # Calculate average loss
    avg_loss = total_loss / num_batches

    return {"train_loss": avg_loss, "train_batches": num_batches}


def _validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: "YOLOConfig",
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()

    total_loss = 0.0
    num_batches = len(val_loader)

    # Progress bar
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

    with torch.no_grad():
        for batch_idx, (images, labels, paths, shapes) in enumerate(pbar):
            # Move data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Update metrics
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}",
                }
            )

    # Calculate average loss
    avg_loss = total_loss / num_batches

    return {"val_loss": avg_loss, "val_batches": num_batches}


def _create_optimizer(model: nn.Module, config: "YOLOConfig") -> optim.Optimizer:
    """Create optimizer based on configuration."""
    model_config = config.model_config

    if model_config.get("optimizer", "auto") == "auto":
        # Use Ultralytics AutoOptimizer
        try:
            from ultralytics.utils.torch_utils import get_optimizer

            optimizer = get_optimizer(model, config.model_config["learning_rate"])
            logger.info("Using Ultralytics AutoOptimizer")
        except ImportError:
            # Fallback to Adam
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.model_config["learning_rate"],
                weight_decay=config.model_config.get("weight_decay", 0.0005),
            )
            logger.info("Using Adam optimizer (fallback)")
    else:
        # Use specified optimizer
        optimizer_name = model_config["optimizer"].lower()
        lr = config.model_config["learning_rate"]
        weight_decay = config.model_config.get("weight_decay", 0.0005)

        if optimizer_name == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.937,
                weight_decay=weight_decay,
                nesterov=True,
            )
        elif optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        logger.info(f"Using {optimizer_name} optimizer")

    return optimizer


def _create_scheduler(
    optimizer: optim.Optimizer, config: "YOLOConfig"
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler based on configuration."""
    model_config = config.model_config
    scheduler_name = model_config.get("lr_scheduler", "cosine").lower()

    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=model_config.get("min_lr", 1e-6)
        )
    elif scheduler_name == "reduce_lr_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=model_config.get("min_lr", 1e-6),
        )
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == "none":
        scheduler = None
    else:
        logger.warning(f"Unknown scheduler: {scheduler_name}, using none")
        scheduler = None

    if scheduler:
        logger.info(f"Using {scheduler_name} learning rate scheduler")

    return scheduler


def _create_loss_function(config: "YOLOConfig") -> nn.Module:
    """Create loss function based on configuration."""
    # For YOLO models, we typically use a combination of losses
    # This is a simplified version - in practice, you'd use the model's built-in loss

    # Placeholder loss function
    # In practice, YOLO models have their own loss computation
    class YOLOLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.mse_loss = nn.MSELoss()
            self.bce_loss = nn.BCEWithLogitsLoss()

        def forward(self, outputs, targets):
            # This is a simplified loss - actual YOLO loss is more complex
            # For now, just return a dummy loss
            return torch.tensor(0.0, requires_grad=True, device=outputs.device)

    return YOLOLoss()


def _should_stop_early(training_history: list, patience: int) -> bool:
    """Check if training should stop early."""
    if patience <= 0 or len(training_history) < patience:
        return False

    # Get validation losses from last patience epochs
    recent_losses = [h["val_loss"] for h in training_history[-patience:]]

    # Check if loss is not improving
    min_loss = min(recent_losses)
    if recent_losses[-1] > min_loss:
        return True

    return False


def validate_model(
    model: nn.Module, config: "YOLOConfig", monitor: TrainingMonitor
) -> Dict[str, Any]:
    """
    Validate trained model.

    Args:
        model: Trained model
        config: Model configuration
        monitor: Training monitor

    Returns:
        Validation results
    """
    logger.info("Starting model validation...")

    # Create validation data loader
    val_loader = create_dataloader(config, split="valid", augment=False, shuffle=False)

    # Set up loss function
    criterion = _create_loss_function(config)

    # Validate
    device = torch.device(config.device)
    model = model.to(device)

    val_metrics = _validate_epoch(
        model, val_loader, criterion, device, config.epochs, config
    )

    # Log validation metrics
    monitor.log_metrics(val_metrics, config.epochs)

    logger.info(f"Validation completed. Loss: {val_metrics['val_loss']:.4f}")

    return val_metrics
