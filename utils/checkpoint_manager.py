"""
Checkpoint manager for YOLO training.
Handles saving, loading, and managing model checkpoints.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import shutil

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints during training."""

    def __init__(self, save_dir: Path, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        self.metadata_file = self.save_dir / "checkpoints.json"
        self.checkpoints = self._load_metadata()
        
        # Ensure metadata file exists
        if not self.metadata_file.exists():
            self._save_metadata()

        logger.info(f"Checkpoint manager initialized: {self.save_dir}")
        logger.info(f"Maximum checkpoints to keep: {self.max_checkpoints}")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")

        return {"checkpoints": [], "best_checkpoint": None}

    def _save_metadata(self):
        """Save checkpoint metadata to file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.checkpoints, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """
        Save model checkpoint.

        Args:
            checkpoint: Checkpoint data to save
            epoch: Current epoch number
            metrics: Training metrics
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint filename
        checkpoint_name = f"checkpoint_epoch_{epoch:03d}.pt"
        checkpoint_path = self.save_dir / checkpoint_name

        # Save checkpoint
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

        # Update metadata
        checkpoint_info = {
            "path": str(checkpoint_path),
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": self._get_timestamp(),
            "is_best": is_best,
        }

        self.checkpoints["checkpoints"].append(checkpoint_info)

        # Update best checkpoint if needed
        if is_best:
            self.checkpoints["best_checkpoint"] = checkpoint_info
            logger.info(f"New best checkpoint: epoch {epoch}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        # Save metadata
        self._save_metadata()

        return checkpoint_path

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the latest checkpoint path."""
        if not self.checkpoints["checkpoints"]:
            return None

        # Sort by epoch and get the latest
        sorted_checkpoints = sorted(
            self.checkpoints["checkpoints"], key=lambda x: x["epoch"], reverse=True
        )

        latest_path = Path(sorted_checkpoints[0]["path"])
        if latest_path.exists():
            return latest_path
        else:
            logger.warning(f"Latest checkpoint not found: {latest_path}")
            return None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get the best checkpoint path based on metrics."""
        if not self.checkpoints["best_checkpoint"]:
            return None

        best_path = Path(self.checkpoints["best_checkpoint"]["path"])
        if best_path.exists():
            return best_path
        else:
            logger.warning(f"Best checkpoint not found: {best_path}")
            return None

    def get_checkpoint_by_epoch(self, epoch: int) -> Optional[Path]:
        """Get checkpoint path for specific epoch."""
        for checkpoint_info in self.checkpoints["checkpoints"]:
            if checkpoint_info["epoch"] == epoch:
                checkpoint_path = Path(checkpoint_info["path"])
                if checkpoint_path.exists():
                    return checkpoint_path
                else:
                    logger.warning(
                        f"Checkpoint for epoch {epoch} not found: {checkpoint_path}"
                    )
                    return None

        logger.warning(f"No checkpoint found for epoch {epoch}")
        return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return self.checkpoints["checkpoints"]

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoints["checkpoints"]) <= self.max_checkpoints:
            return

        # Sort checkpoints by epoch (oldest first)
        sorted_checkpoints = sorted(
            self.checkpoints["checkpoints"], key=lambda x: x["epoch"]
        )

        # Keep the best checkpoint and the most recent ones
        checkpoints_to_keep = []

        # Always keep the best checkpoint
        if self.checkpoints["best_checkpoint"]:
            checkpoints_to_keep.append(self.checkpoints["best_checkpoint"])

        # Add the most recent checkpoints (excluding best checkpoint if already included)
        # Calculate how many recent checkpoints we can add
        remaining_slots = self.max_checkpoints - len(checkpoints_to_keep)
        if remaining_slots > 0:
            recent_checkpoints = sorted_checkpoints[-remaining_slots:]
            for checkpoint in recent_checkpoints:
                if checkpoint not in checkpoints_to_keep:
                    checkpoints_to_keep.append(checkpoint)

        # Remove old checkpoints
        checkpoints_to_remove = [
            c for c in self.checkpoints["checkpoints"] if c not in checkpoints_to_keep
        ]

        for checkpoint in checkpoints_to_remove:
            try:
                checkpoint_path = Path(checkpoint["path"])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint['path']}: {e}")

        # Update metadata
        self.checkpoints["checkpoints"] = checkpoints_to_keep
        logger.info(f"Kept {len(checkpoints_to_keep)} checkpoints")

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime

        return datetime.now().isoformat()

    def resume_training(
        self, checkpoint_path: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (if None, use latest)

        Returns:
            Checkpoint data or None if not found
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None:
            logger.warning("No checkpoint found for resuming training")
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint for resuming: {e}")
            return None

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress from checkpoints."""
        if not self.checkpoints["checkpoints"]:
            return {"status": "No checkpoints found"}

        # Sort checkpoints by epoch
        sorted_checkpoints = sorted(
            self.checkpoints["checkpoints"], key=lambda x: x["epoch"]
        )

        # Extract metrics over time
        epochs = [c["epoch"] for c in sorted_checkpoints]
        metrics_over_time = {}

        # Get all metric keys
        all_metrics = set()
        for checkpoint in sorted_checkpoints:
            all_metrics.update(checkpoint["metrics"].keys())

        # Build metrics over time
        for metric in all_metrics:
            metrics_over_time[metric] = [
                checkpoint["metrics"].get(metric, 0.0)
                for checkpoint in sorted_checkpoints
            ]

        summary = {
            "status": "Training in progress",
            "total_checkpoints": len(sorted_checkpoints),
            "epochs": epochs,
            "metrics_over_time": metrics_over_time,
            "latest_epoch": max(epochs),
            "best_checkpoint": self.checkpoints["best_checkpoint"],
            "checkpoint_dir": str(self.save_dir),
        }

        return summary

    def cleanup_all(self):
        """Remove all checkpoints and metadata."""
        try:
            # Remove all checkpoint files
            for checkpoint_info in self.checkpoints["checkpoints"]:
                checkpoint_path = Path(checkpoint_info["path"])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()

            # Remove metadata file
            if self.metadata_file.exists():
                self.metadata_file.unlink()

            # Reset internal state
            self.checkpoints = {"checkpoints": [], "best_checkpoint": None}

            logger.info("All checkpoints cleaned up")

        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")

    def __str__(self) -> str:
        """String representation of checkpoint manager."""
        return f"CheckpointManager(save_dir={self.save_dir}, checkpoints={len(self.checkpoints['checkpoints'])})"
