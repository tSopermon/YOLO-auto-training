"""
Tests for checkpoint manager functionality.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import torch

from utils.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """Test CheckpointManager class functionality."""

    def test_initialization(self, temp_dir):
        """Test checkpoint manager initialization."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=5)

        assert manager.save_dir == checkpoint_dir
        assert manager.max_checkpoints == 5
        assert checkpoint_dir.exists()

        # Check metadata file creation
        metadata_file = checkpoint_dir / "checkpoints.json"
        assert metadata_file.exists()

    def test_save_checkpoint(self, temp_dir, sample_checkpoint):
        """Test saving a checkpoint."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            checkpoint=sample_checkpoint,
            epoch=5,
            metrics={"train_loss": 0.5, "val_loss": 0.6},
            is_best=False,
        )

        # Check file exists
        assert checkpoint_path.exists()
        assert checkpoint_path.name == "checkpoint_epoch_005.pt"

        # Check metadata
        metadata = manager.checkpoints
        assert len(metadata["checkpoints"]) == 1
        assert metadata["checkpoints"][0]["epoch"] == 5
        assert metadata["checkpoints"][0]["is_best"] is False

    def test_save_best_checkpoint(self, temp_dir, sample_checkpoint):
        """Test saving best checkpoint."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Save best checkpoint
        checkpoint_path = manager.save_checkpoint(
            checkpoint=sample_checkpoint,
            epoch=5,
            metrics={"train_loss": 0.5, "val_loss": 0.6},
            is_best=True,
        )

        # Check metadata
        metadata = manager.checkpoints
        assert metadata["best_checkpoint"] is not None
        assert metadata["best_checkpoint"]["epoch"] == 5
        assert metadata["best_checkpoint"]["is_best"] is True

    def test_get_latest_checkpoint(self, temp_dir, sample_checkpoint):
        """Test getting latest checkpoint."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Save multiple checkpoints
        manager.save_checkpoint(sample_checkpoint, 5, {"loss": 0.5})
        manager.save_checkpoint(sample_checkpoint, 10, {"loss": 0.4})
        manager.save_checkpoint(sample_checkpoint, 15, {"loss": 0.3})

        # Get latest
        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert latest.name == "checkpoint_epoch_015.pt"

    def test_get_best_checkpoint(self, temp_dir, sample_checkpoint):
        """Test getting best checkpoint."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(
            save_dir=checkpoint_dir, max_checkpoints=5
        )  # Increased to avoid cleanup

        # Save checkpoints with different metrics
        manager.save_checkpoint(sample_checkpoint, 5, {"val_loss": 0.6}, is_best=False)
        manager.save_checkpoint(sample_checkpoint, 10, {"val_loss": 0.4}, is_best=True)
        manager.save_checkpoint(sample_checkpoint, 15, {"val_loss": 0.8}, is_best=False)

        # Get best
        best = manager.get_best_checkpoint()
        assert best is not None
        assert best.name == "checkpoint_epoch_010.pt"

    def test_get_checkpoint_by_epoch(self, temp_dir, sample_checkpoint):
        """Test getting checkpoint by specific epoch."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(
            save_dir=checkpoint_dir, max_checkpoints=5
        )  # Increased to avoid cleanup

        # Save checkpoint
        manager.save_checkpoint(sample_checkpoint, 5, {"loss": 0.5})

        # Get by epoch
        checkpoint = manager.get_checkpoint_by_epoch(5)
        assert checkpoint is not None
        assert checkpoint.name == "checkpoint_epoch_005.pt"

        # Get non-existent epoch
        checkpoint = manager.get_checkpoint_by_epoch(10)
        assert checkpoint is None

    def test_list_checkpoints(self, temp_dir, sample_checkpoint):
        """Test listing all checkpoints."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(
            save_dir=checkpoint_dir, max_checkpoints=5
        )  # Increased to avoid cleanup

        # Save multiple checkpoints
        manager.save_checkpoint(sample_checkpoint, 5, {"loss": 0.5})
        manager.save_checkpoint(sample_checkpoint, 10, {"loss": 0.4})

        # List checkpoints
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2
        assert checkpoints[0]["epoch"] == 5
        assert checkpoints[1]["epoch"] == 10

    def test_cleanup_old_checkpoints(self, temp_dir, sample_checkpoint):
        """Test cleanup of old checkpoints."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Save more checkpoints than max allowed
        for epoch in range(1, 7):  # 6 checkpoints
            manager.save_checkpoint(sample_checkpoint, epoch, {"loss": 0.5})

        # Should keep only 3 checkpoints
        assert len(manager.checkpoints["checkpoints"]) == 3

        # Check files
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(checkpoint_files) == 3

    def test_cleanup_preserves_best_checkpoint(self, temp_dir, sample_checkpoint):
        """Test that cleanup preserves the best checkpoint."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Save checkpoints with best in middle
        manager.save_checkpoint(sample_checkpoint, 1, {"val_loss": 0.8})
        manager.save_checkpoint(sample_checkpoint, 2, {"val_loss": 0.4}, is_best=True)
        manager.save_checkpoint(sample_checkpoint, 3, {"val_loss": 0.9})
        manager.save_checkpoint(sample_checkpoint, 4, {"val_loss": 0.7})
        manager.save_checkpoint(sample_checkpoint, 5, {"val_loss": 0.6})

        # Should keep best checkpoint (epoch 2) and 2 most recent
        checkpoints = manager.list_checkpoints()
        epochs = [c["epoch"] for c in checkpoints]

        # Best checkpoint should be preserved
        assert 2 in epochs
        # Most recent should be preserved
        assert 5 in epochs
        assert 4 in epochs

    def test_resume_training(self, temp_dir, sample_checkpoint):
        """Test resuming training from checkpoint."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Save checkpoint
        manager.save_checkpoint(sample_checkpoint, 5, {"loss": 0.5})

        # Resume training
        checkpoint_data = manager.resume_training()
        assert checkpoint_data is not None
        assert checkpoint_data["epoch"] == 5

    def test_resume_training_specific_checkpoint(self, temp_dir, sample_checkpoint):
        """Test resuming from specific checkpoint."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(
            save_dir=checkpoint_dir, max_checkpoints=5
        )  # Increased to avoid cleanup

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(sample_checkpoint, 5, {"loss": 0.5})

        # Resume from specific checkpoint
        checkpoint_data = manager.resume_training(checkpoint_path)
        assert checkpoint_data is not None
        assert checkpoint_data["epoch"] == 5

    def test_resume_training_no_checkpoints(self, temp_dir):
        """Test resuming training when no checkpoints exist."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Try to resume
        checkpoint_data = manager.resume_training()
        assert checkpoint_data is None

    def test_get_training_summary(self, temp_dir, sample_checkpoint):
        """Test getting training summary."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Save multiple checkpoints
        manager.save_checkpoint(
            sample_checkpoint, 5, {"train_loss": 0.5, "val_loss": 0.6}
        )
        manager.save_checkpoint(
            sample_checkpoint, 10, {"train_loss": 0.4, "val_loss": 0.5}
        )
        manager.save_checkpoint(
            sample_checkpoint, 15, {"train_loss": 0.3, "val_loss": 0.4}
        )

        # Get summary
        summary = manager.get_training_summary()

        assert summary["total_checkpoints"] == 3
        assert summary["latest_epoch"] == 15
        assert "metrics_over_time" in summary
        assert "train_loss" in summary["metrics_over_time"]
        assert "val_loss" in summary["metrics_over_time"]

    def test_get_training_summary_no_checkpoints(self, temp_dir):
        """Test getting training summary with no checkpoints."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        summary = manager.get_training_summary()
        assert summary["status"] == "No checkpoints found"

    def test_cleanup_all(self, temp_dir, sample_checkpoint):
        """Test cleaning up all checkpoints."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Save some checkpoints
        manager.save_checkpoint(sample_checkpoint, 5, {"loss": 0.5})
        manager.save_checkpoint(sample_checkpoint, 10, {"loss": 0.4})

        # Cleanup all
        manager.cleanup_all()

        # Checkpoints should be removed
        assert len(manager.checkpoints["checkpoints"]) == 0
        assert manager.checkpoints["best_checkpoint"] is None

        # Files should be removed
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(checkpoint_files) == 0

        # Metadata file should be removed
        metadata_file = checkpoint_dir / "checkpoints.json"
        assert not metadata_file.exists()

    def test_metadata_persistence(self, temp_dir, sample_checkpoint):
        """Test that metadata persists across manager instances."""
        checkpoint_dir = temp_dir / "checkpoints"

        # Create first manager and save checkpoint
        manager1 = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)
        manager1.save_checkpoint(sample_checkpoint, 5, {"loss": 0.5}, is_best=True)

        # Create second manager (should load existing metadata)
        manager2 = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Should have the same checkpoints
        assert len(manager2.checkpoints["checkpoints"]) == 1
        assert manager2.checkpoints["best_checkpoint"] is not None
        assert manager2.checkpoints["best_checkpoint"]["epoch"] == 5

    def test_metadata_corruption_handling(self, temp_dir):
        """Test handling of corrupted metadata file."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create corrupted metadata file
        metadata_file = checkpoint_dir / "checkpoints.json"
        metadata_file.write_text("invalid json content")

        # Manager should handle corruption gracefully
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Should have empty metadata
        assert len(manager.checkpoints["checkpoints"]) == 0
        assert manager.checkpoints["best_checkpoint"] is None

    def test_checkpoint_file_missing(self, temp_dir, sample_checkpoint):
        """Test handling when checkpoint file is missing."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(sample_checkpoint, 5, {"loss": 0.5})

        # Remove checkpoint file manually
        checkpoint_path.unlink()

        # Should handle missing file gracefully
        latest = manager.get_latest_checkpoint()
        assert latest is None

        # Should log warning about missing file
        # (This is tested in the actual implementation)

    def test_string_representation(self, temp_dir):
        """Test string representation of checkpoint manager."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        string_repr = str(manager)
        assert "CheckpointManager" in string_repr
        assert str(checkpoint_dir) in string_repr
        # The manager creates a metadata file during initialization, but checkpoints list is empty
        assert "checkpoints=0" in string_repr


class TestCheckpointManagerIntegration:
    """Test checkpoint manager integration scenarios."""

    def test_full_training_cycle(self, temp_dir, sample_checkpoint):
        """Test complete training cycle with checkpoint management."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=5)

        # Simulate training epochs
        for epoch in range(1, 11):
            metrics = {"train_loss": 1.0 / epoch, "val_loss": 1.0 / epoch + 0.1}

            # Save checkpoint
            is_best = epoch == 5  # Epoch 5 is best
            manager.save_checkpoint(sample_checkpoint, epoch, metrics, is_best=is_best)

        # Verify final state - should keep only 5 checkpoints due to max_checkpoints limit
        assert len(manager.checkpoints["checkpoints"]) == 5
        assert manager.checkpoints["best_checkpoint"]["epoch"] == 5

        # Get training summary
        summary = manager.get_training_summary()
        assert summary["total_checkpoints"] == 5
        # Note: best_epoch and best_val_loss are not in the summary structure
        assert summary["best_checkpoint"]["epoch"] == 5

    def test_checkpoint_rotation(self, temp_dir, sample_checkpoint):
        """Test checkpoint rotation when max limit is reached."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=3)

        # Save more checkpoints than limit
        for epoch in range(1, 8):  # 7 checkpoints
            metrics = {"loss": 1.0 / epoch}
            is_best = epoch == 3  # Epoch 3 is best
            manager.save_checkpoint(sample_checkpoint, epoch, metrics, is_best=is_best)

        # Should keep only 3 checkpoints
        assert len(manager.checkpoints["checkpoints"]) == 3

        # Best checkpoint should be preserved
        assert manager.checkpoints["best_checkpoint"]["epoch"] == 3

        # Most recent checkpoints should be preserved
        epochs = [c["epoch"] for c in manager.checkpoints["checkpoints"]]
        assert 3 in epochs  # Best checkpoint
        assert 7 in epochs  # Most recent
        assert 6 in epochs  # Second most recent
        # Should have exactly 3 checkpoints
        assert len(epochs) == 3

    def test_resume_and_continue_training(self, temp_dir, sample_checkpoint):
        """Test resuming training and continuing."""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(save_dir=checkpoint_dir, max_checkpoints=5)

        # Initial training
        for epoch in range(1, 6):
            metrics = {"loss": 1.0 / epoch}
            manager.save_checkpoint(sample_checkpoint, epoch, metrics)

        # Resume training
        checkpoint_data = manager.resume_training()
        assert checkpoint_data["epoch"] == 5

        # Continue training
        for epoch in range(6, 11):
            metrics = {"loss": 1.0 / epoch}
            manager.save_checkpoint(sample_checkpoint, epoch, metrics)

        # Verify final state - should keep only 5 checkpoints due to max_checkpoints limit
        assert len(manager.checkpoints["checkpoints"]) == 5
        # The last checkpoint should be the most recent one kept
        assert max(c["epoch"] for c in manager.checkpoints["checkpoints"]) == 10
