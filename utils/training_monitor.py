"""
Training monitor for YOLO models.
Provides comprehensive monitoring, logging, and visualization of training progress.
"""

import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import json
import yaml
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Comprehensive training monitor for YOLO models."""

    def __init__(
        self,
        config: "YOLOConfig",
        log_dir: Union[str, Path],
        enable_tensorboard: bool = True,
        enable_wandb: bool = False,
    ):
        """
        Initialize training monitor.

        Args:
            config: Training configuration
            log_dir: Directory to save logs and visualizations
            enable_tensorboard: Whether to enable TensorBoard logging
            enable_wandb: Whether to enable Weights & Biases logging
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Monitoring state
        self.metrics_history = defaultdict(list)
        self.start_time = time.time()
        self.epoch_times = []

        # Setup logging backends
        self.tensorboard_writer = None
        self.wandb_run = None

        if enable_tensorboard:
            self._setup_tensorboard()

        if enable_wandb:
            self._setup_wandb()

        # Create metrics file
        self.metrics_file = self.log_dir / "training_metrics.jsonl"

        logger.info(f"Training monitor initialized: {self.log_dir}")

    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_dir = self.log_dir / "tensorboard"
            tensorboard_dir.mkdir(exist_ok=True)

            self.tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_dir))
            logger.info(f"TensorBoard logging enabled: {tensorboard_dir}")

        except ImportError:
            logger.warning(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
        except Exception as e:
            logger.warning(f"Failed to setup TensorBoard: {e}")

    def _setup_wandb(self) -> None:
        """Setup Weights & Biases logging."""
        try:
            import wandb

            # Initialize W&B run
            wandb_config = {
                "model_type": self.config.model_type,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "image_size": self.config.image_size,
                "learning_rate": self.config.model_config.get("learning_rate", 0.01),
                "optimizer": self.config.model_config.get("optimizer", "auto"),
                "lr_scheduler": self.config.model_config.get("lr_scheduler", "cosine"),
                "device": self.config.device,
            }

            self.wandb_run = wandb.init(
                project=self.config.logging_config.get(
                    "project_name",
                    self.config.logging_config.get("wandb_project", "yolo_training"),
                ),
                config=wandb_config,
                name=f"{self.config.model_type}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=[self.config.model_type, "object_detection", "yolo"],
            )

            logger.info(f"Weights & Biases logging enabled: {self.wandb_run.name}")

        except ImportError:
            logger.warning(
                "Weights & Biases not available. Install with: pip install wandb"
            )
        except Exception as e:
            logger.warning(f"Failed to setup Weights & Biases: {e}")

    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
    ) -> None:
        """
        Log metrics for a training epoch.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            learning_rate: Current learning rate
        """
        # Record epoch time
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)

        # Combine all metrics
        all_metrics = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "epoch_time": epoch_time,
            "learning_rate": learning_rate,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        # Update metrics history
        for key, value in all_metrics.items():
            if key != "timestamp":
                self.metrics_history[key].append(value)

        # Log to TensorBoard
        if self.tensorboard_writer:
            self._log_to_tensorboard(epoch, all_metrics)

        # Log to Weights & Biases
        if self.wandb_run:
            self._log_to_wandb(epoch, all_metrics)

        # Save to JSONL file
        self._save_metrics_to_file(all_metrics)

        # Log to console
        self._log_to_console(epoch, all_metrics)

    def _log_to_tensorboard(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics to TensorBoard."""
        try:
            # Log scalar metrics
            for key, value in metrics.items():
                if key not in ["epoch", "timestamp"] and isinstance(
                    value, (int, float)
                ):
                    self.tensorboard_writer.add_scalar(key, value, epoch)

            # Log learning rate
            self.tensorboard_writer.add_scalar(
                "learning_rate", metrics["learning_rate"], epoch
            )

            # Log training vs validation loss
            if "train_loss" in metrics and "val_loss" in metrics:
                self.tensorboard_writer.add_scalars(
                    "loss_comparison",
                    {"train": metrics["train_loss"], "validation": metrics["val_loss"]},
                    epoch,
                )

        except Exception as e:
            logger.warning(f"Failed to log to TensorBoard: {e}")

    def _log_to_wandb(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics to Weights & Biases."""
        try:
            # Prepare metrics for W&B
            wandb_metrics = {}
            for key, value in metrics.items():
                if key not in ["epoch", "timestamp"] and isinstance(
                    value, (int, float)
                ):
                    wandb_metrics[key] = value

            # Log to W&B
            self.wandb_run.log(wandb_metrics, step=epoch)

        except Exception as e:
            logger.warning(f"Failed to log to Weights & Biases: {e}")

    def _save_metrics_to_file(self, metrics: Dict[str, float]) -> None:
        """Save metrics to JSONL file."""
        try:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save metrics to file: {e}")

    def _log_to_console(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics to console."""
        train_loss = metrics.get("train_loss", "N/A")
        val_loss = metrics.get("val_loss", "N/A")
        lr = metrics.get("learning_rate", "N/A")

        # Format numeric values, handle string values
        train_loss_str = (
            f"{train_loss:.4f}"
            if isinstance(train_loss, (int, float))
            else str(train_loss)
        )
        val_loss_str = (
            f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
        )
        lr_str = f"{lr:.6f}" if isinstance(lr, (int, float)) else str(lr)

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss_str} | "
            f"Val Loss: {val_loss_str} | "
            f"LR: {lr_str}"
        )

    def log_batch_metrics(
        self,
        epoch: int,
        batch_idx: int,
        num_batches: int,
        batch_metrics: Dict[str, float],
    ) -> None:
        """
        Log metrics for a training batch.

        Args:
            epoch: Current epoch number
            batch_idx: Current batch index
            num_batches: Total number of batches
            batch_metrics: Batch metrics
        """
        # Log to TensorBoard if enabled
        if (
            self.tensorboard_writer
            and batch_idx % self.config.logging_config.get("log_metrics_interval", 10)
            == 0
        ):
            global_step = epoch * num_batches + batch_idx

            for key, value in batch_metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(
                        f"batch_{key}", value, global_step
                    )

    def log_model_info(self, model: torch.nn.Module) -> None:
        """
        Log model information.

        Args:
            model: PyTorch model
        """
        try:
            # Calculate model size
            param_size = 0
            buffer_size = 0

            for param in model.parameters():
                param_size += param.nelement() * param.element_size()

            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            total_size_mb = (param_size + buffer_size) / 1024**2

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            model_info = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": total_params - trainable_params,
                "model_size_mb": total_size_mb,
                "model_size_gb": total_size_mb / 1024,
            }

            # Log to TensorBoard
            if self.tensorboard_writer:
                for key, value in model_info.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(f"model/{key}", value, 0)

            # Log to W&B
            if self.wandb_run:
                self.wandb_run.config.update(model_info)

            # Save to file
            model_info_file = self.log_dir / "model_info.json"
            with open(model_info_file, "w") as f:
                json.dump(model_info, f, indent=4)

            logger.info(f"Model information logged: {model_info}")

        except Exception as e:
            logger.warning(f"Failed to log model info: {e}")

    def log_config(self) -> None:
        """Log training configuration."""
        try:
            config_info = {
                "model_type": self.config.model_type,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "image_size": self.config.image_size,
                "learning_rate": self.config.model_config.get("learning_rate", 0.01),
                "optimizer": self.config.model_config.get("optimizer", "auto"),
                "lr_scheduler": self.config.model_config.get("lr_scheduler", "cosine"),
                "device": self.config.device,
                "patience": self.config.patience,
                "deterministic": self.config.deterministic,
                "single_cls": self.config.single_cls,
                "rect": self.config.rect,
                "cos_lr": self.config.cos_lr,
            }

            # Save to file
            config_file = self.log_dir / "training_config.json"
            with open(config_file, "w") as f:
                json.dump(config_info, f, indent=4)

            # Log to W&B
            if self.wandb_run:
                self.wandb_run.config.update(config_info)

            logger.info("Training configuration logged")

        except Exception as e:
            logger.warning(f"Failed to log config: {e}")

    def create_training_plots(self, save_plots: bool = True) -> None:
        """Create training visualization plots."""
        if not self.metrics_history:
            logger.warning("No metrics available for plotting")
            return

        try:
            # Create plots directory
            plots_dir = self.log_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Plot training vs validation loss
            self._plot_loss_comparison(plots_dir, save_plots)

            # Plot learning rate
            self._plot_learning_rate(plots_dir, save_plots)

            # Plot other metrics
            self._plot_other_metrics(plots_dir, save_plots)

            logger.info(f"Training plots created in {plots_dir}")

        except Exception as e:
            logger.warning(f"Failed to create training plots: {e}")

    def _plot_loss_comparison(self, plots_dir: Path, save_plots: bool) -> None:
        """Plot training vs validation loss."""
        if (
            "train_loss" not in self.metrics_history
            or "val_loss" not in self.metrics_history
        ):
            return

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.metrics_history["train_loss"]) + 1)

        plt.plot(
            epochs,
            self.metrics_history["train_loss"],
            "b-",
            label="Training Loss",
            linewidth=2,
        )
        plt.plot(
            epochs,
            self.metrics_history["val_loss"],
            "r-",
            label="Validation Loss",
            linewidth=2,
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_plots:
            plt.savefig(plots_dir / "loss_comparison.png", dpi=300, bbox_inches="tight")

        plt.show()

    def _plot_learning_rate(self, plots_dir: Path, save_plots: bool) -> None:
        """Plot learning rate over time."""
        if "learning_rate" not in self.metrics_history:
            return

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.metrics_history["learning_rate"]) + 1)

        plt.plot(epochs, self.metrics_history["learning_rate"], "g-", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        if save_plots:
            plt.savefig(plots_dir / "learning_rate.png", dpi=300, bbox_inches="tight")

        plt.show()

    def _plot_other_metrics(self, plots_dir: Path, save_plots: bool) -> None:
        """Plot other training metrics."""
        # Find other numeric metrics
        other_metrics = {}
        for key, values in self.metrics_history.items():
            if key not in [
                "epoch",
                "timestamp",
                "epoch_time",
                "learning_rate",
                "train_loss",
                "val_loss",
            ]:
                if values and all(isinstance(v, (int, float)) for v in values):
                    other_metrics[key] = values

        if not other_metrics:
            return

        # Create subplots for other metrics
        num_metrics = len(other_metrics)
        cols = min(3, num_metrics)
        rows = (num_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, (metric_name, values) in enumerate(other_metrics.items()):
            if i < len(axes):
                epochs = range(1, len(values) + 1)
                axes[i].plot(epochs, values, linewidth=2)
                axes[i].set_xlabel("Epoch")
                axes[i].set_ylabel(metric_name.replace("_", " ").title())
                axes[i].set_title(f'{metric_name.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(num_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_plots:
            plt.savefig(plots_dir / "other_metrics.png", dpi=300, bbox_inches="tight")

        plt.show()

    def generate_training_report(self) -> str:
        """Generate comprehensive training report."""
        if not self.metrics_history:
            return "No training metrics available for report generation."

        # Calculate training statistics
        total_epochs = len(self.metrics_history.get("epoch", []))
        total_time = time.time() - self.start_time

        if "train_loss" in self.metrics_history and "val_loss" in self.metrics_history:
            best_train_loss = min(self.metrics_history["train_loss"])
            best_val_loss = min(self.metrics_history["val_loss"])
            final_train_loss = self.metrics_history["train_loss"][-1]
            final_val_loss = self.metrics_history["val_loss"][-1]
        else:
            best_train_loss = best_val_loss = final_train_loss = final_val_loss = "N/A"

        if "learning_rate" in self.metrics_history:
            initial_lr = self.metrics_history["learning_rate"][0]
            final_lr = self.metrics_history["learning_rate"][-1]
        else:
            initial_lr = final_lr = "N/A"

        report = f"""
YOLO Training Report
====================

Training Summary:
----------------
Model Type: {self.config.model_type}
Total Epochs: {total_epochs}
Total Training Time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)
Average Epoch Time: {total_time/total_epochs:.2f} seconds per epoch

Loss Metrics:
-------------
Best Training Loss: {best_train_loss}
Best Validation Loss: {best_val_loss}
Final Training Loss: {final_train_loss}
Final Validation Loss: {final_val_loss}

Learning Rate:
--------------
Initial Learning Rate: {initial_lr}
Final Learning Rate: {final_lr}

Configuration:
--------------
Batch Size: {self.config.batch_size}
Image Size: {self.config.image_size}
Device: {self.config.device}
Optimizer: {self.config.model_config.get('optimizer', 'auto')}
LR Scheduler: {self.config.model_config.get('lr_scheduler', 'cosine')}
Patience: {self.config.patience}
"""

        return report

    def save_training_summary(self) -> None:
        """Save training summary to files."""
        try:
            # Generate report
            report = self.generate_training_report()

            # Save text report
            report_file = self.log_dir / "training_report.txt"
            with open(report_file, "w") as f:
                f.write(report)

            # Save metrics summary
            summary = {
                "training_summary": {
                    "total_epochs": len(self.metrics_history.get("epoch", [])),
                    "total_time_seconds": time.time() - self.start_time,
                    "best_train_loss": min(
                        self.metrics_history.get("train_loss", [float("inf")])
                    ),
                    "best_val_loss": min(
                        self.metrics_history.get("val_loss", [float("inf")])
                    ),
                    "final_train_loss": (
                        self.metrics_history.get("train_loss", [])[-1]
                        if self.metrics_history.get("train_loss")
                        else None
                    ),
                    "final_val_loss": (
                        self.metrics_history.get("val_loss", [])[-1]
                        if self.metrics_history.get("val_loss")
                        else None
                    ),
                },
                "metrics_history": dict(self.metrics_history),
                "config": {
                    "model_type": self.config.model_type,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "image_size": self.config.image_size,
                },
            }

            summary_file = self.log_dir / "training_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=4, default=str)

            logger.info(f"Training summary saved to {self.log_dir}")

        except Exception as e:
            logger.warning(f"Failed to save training summary: {e}")

    def close(self) -> None:
        """Close monitoring and cleanup resources."""
        try:
            # Close TensorBoard writer
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
                logger.info("TensorBoard writer closed")

            # Close W&B run
            if self.wandb_run:
                self.wandb_run.finish()
                logger.info("Weights & Biases run finished")

            # Create final plots and summary
            self.create_training_plots(save_plots=True)
            self.save_training_summary()

            logger.info("Training monitor closed successfully")

        except Exception as e:
            logger.warning(f"Error during monitor cleanup: {e}")


def create_training_monitor(
    config: "YOLOConfig",
    log_dir: Union[str, Path],
    enable_tensorboard: Optional[bool] = None,
    enable_wandb: Optional[bool] = None,
) -> TrainingMonitor:
    """
    Create training monitor with configuration-based settings.

    Args:
        config: Training configuration
        log_dir: Directory to save logs
        enable_tensorboard: Override config TensorBoard setting
        enable_wandb: Override config W&B setting

    Returns:
        TrainingMonitor instance
    """
    if enable_tensorboard is None:
        enable_tensorboard = config.logging_config.get("tensorboard", True)

    if enable_wandb is None:
        enable_wandb = config.logging_config.get("wandb", False)

    return TrainingMonitor(
        config=config,
        log_dir=log_dir,
        enable_tensorboard=enable_tensorboard,
        enable_wandb=enable_wandb,
    )
