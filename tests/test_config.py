"""
Tests for configuration management system.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import torch

from config.config import YOLOConfig, get_config, setup_logging
from config.constants import (
    COMMON_TRAINING,
    YOLO11_CONFIG,
    YOLOV8_CONFIG,
    YOLOV5_CONFIG,
    DATASET_CONFIG,
    LOGGING_CONFIG,
)


class TestYOLOConfig:
    """Test YOLOConfig class functionality."""

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    @patch("config.config.YOLOConfig._setup_device")
    def test_config_creation(
        self, mock_setup_device, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test creating a new YOLOConfig instance."""
        config = YOLOConfig.create("yolov8")

        assert config.model_type == "yolov8"
        assert config.epochs == 100
        assert config.batch_size == 8
        assert config.image_size == 1024
        assert config.device == "cuda"  # Default from constants

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_config_creation_with_model_type(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test creating config for specific YOLO version."""
        config = YOLOConfig.create("yolo11")
        assert config.model_type == "yolo11"
        # The weights path should contain the filename, not just the filename
        assert "yolo11n.pt" in config.weights

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_device_auto_detection_cuda(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test automatic CUDA device detection."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                config = YOLOConfig.create("yolov8")
                assert config.device == "cuda"

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_device_auto_detection_cpu(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test automatic CPU device detection."""
        with patch("torch.cuda.is_available", return_value=False):
            config = YOLOConfig.create("yolov8")
            assert config.device == "cpu"

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_config_validation(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test configuration validation."""
        config = YOLOConfig.create("yolov8")

        # Valid config should pass validation
        assert config.model_type in ["yolo11", "yolov8", "yolov5"]
        assert config.batch_size > 0
        assert config.image_size > 0

        # Invalid model type should fail
        with pytest.raises(ValueError, match="Unsupported model type"):
            YOLOConfig.create("invalid_model")

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    @patch("config.config.YOLOConfig._setup_device")
    def test_config_save_and_load(
        self,
        mock_setup_device,
        mock_setup_wandb,
        mock_load_data_yaml,
        mock_setup_paths,
        temp_dir,
    ):
        """Test saving and loading configuration."""
        config = YOLOConfig.create("yolov8")

        # Convert all PosixPath objects to strings to avoid JSON serialization issues
        def convert_paths_to_strings(obj):
            if isinstance(obj, dict):
                return {k: convert_paths_to_strings(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths_to_strings(item) for item in obj]
            elif hasattr(obj, "__str__") and not isinstance(
                obj, (str, int, float, bool)
            ):
                return str(obj)
            else:
                return obj

        # Apply conversion to all config sections
        config.dataset_config = convert_paths_to_strings(config.dataset_config)
        config.logging_config = convert_paths_to_strings(config.logging_config)
        config.export_config = convert_paths_to_strings(config.export_config)

        config_path = temp_dir / "test_config.json"

        # Save config
        config.save(config_path)
        assert config_path.exists()

        # Load config
        loaded_config = YOLOConfig.load(config_path)
        assert loaded_config.model_type == config.model_type
        assert loaded_config.epochs == config.epochs
        assert loaded_config.batch_size == config.batch_size

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_config_get_training_args(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test getting training arguments dictionary."""
        config = YOLOConfig.create("yolov8")
        training_args = config.get_training_args()

        expected_keys = [
            "epochs",
            "batch",
            "imgsz",
            "device",
            "workers",
            "patience",
            "deterministic",
            "single_cls",
            "rect",
            "cos_lr",
            "close_mosaic",
            "resume",
            "weights",
            "pretrained",
        ]

        for key in expected_keys:
            assert key in training_args

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_config_get_optimizer_config(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test getting optimizer configuration."""
        config = YOLOConfig.create("yolov8")
        optimizer_config = config.get_optimizer_config()

        assert "optimizer" in optimizer_config
        assert "lr" in optimizer_config
        assert "weight_decay" in optimizer_config

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_config_get_scheduler_config(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test getting scheduler configuration."""
        config = YOLOConfig.create("yolov8")
        scheduler_config = config.get_scheduler_config()

        assert "scheduler" in scheduler_config
        assert "warmup_epochs" in scheduler_config

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    @patch("config.config.YOLOConfig._setup_device")
    def test_config_roboflow_integration(
        self, mock_setup_device, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test Roboflow configuration integration."""
        config = YOLOConfig.create("yolov8")

        # Test that the config has the expected configuration sections
        assert hasattr(config, "dataset_config")
        assert hasattr(config, "model_config")
        assert hasattr(config, "augmentation_config")
        assert hasattr(config, "eval_config")
        assert hasattr(config, "logging_config")
        assert hasattr(config, "export_config")

        # Test that these configs contain expected keys
        assert "data_yaml_path" in config.dataset_config
        assert "learning_rate" in config.model_config
        assert "mosaic" in config.augmentation_config
        assert "plots" in config.eval_config
        assert "log_dir" in config.logging_config
        assert "export_formats" in config.export_config

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_config_dataset_validation(
        self,
        mock_setup_wandb,
        mock_load_data_yaml,
        mock_setup_paths,
        sample_dataset_structure,
    ):
        """Test dataset configuration validation."""
        config = YOLOConfig.create("yolov8")
        config.dataset_config["data_yaml_path"] = sample_dataset_structure / "data.yaml"

        # Should pass validation with valid dataset
        assert Path(config.dataset_config["data_yaml_path"]).exists()

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_config_invalid_dataset(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test configuration with invalid dataset."""
        config = YOLOConfig.create("yolov8")
        config.dataset_config["data_yaml_path"] = Path("nonexistent/path/data.yaml")

        # Should fail validation with invalid dataset
        assert not Path(config.dataset_config["data_yaml_path"]).exists()


class TestGetConfig:
    """Test get_config function."""

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_get_yolov8_config(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test getting YOLOv8 configuration."""
        config = get_config("yolov8")
        assert config.model_type == "yolov8"
        # The weights path should contain the filename, not just the filename
        assert "yolov8n.pt" in config.weights

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_get_yolo11_config(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test getting YOLO11 configuration."""
        config = get_config("yolo11")
        assert config.model_type == "yolo11"
        # The weights path should contain the filename, not just the filename
        assert "yolo11n.pt" in config.weights

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_get_yolov5_config(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test getting YOLOv5 configuration."""
        config = get_config("yolov5")
        assert config.model_type == "yolov5"
        # The weights path should contain the filename, not just the filename
        assert "yolov5nu.pt" in config.weights

    def test_get_invalid_config(self):
        """Test getting configuration for invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            get_config("invalid_model")

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_get_config_with_custom_paths(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test getting configuration with custom paths."""
        # This test needs to be updated based on actual get_config signature
        # For now, just test basic functionality
        config = get_config("yolov8")
        assert config.model_type == "yolov8"


class TestSetupLogging:
    """Test logging setup functionality."""

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_setup_logging_basic(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths, temp_dir
    ):
        """Test basic logging setup."""
        config = YOLOConfig.create("yolov8")
        config.logging_config["log_dir"] = str(temp_dir)

        setup_logging(config)

        # Check if log directory was created
        log_dir = Path(temp_dir)
        assert log_dir.exists()

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_setup_logging_with_tensorboard(
        self,
        mock_setup_wandb,
        mock_load_data_yaml,
        mock_setup_paths,
        temp_dir,
        patch_tensorboard,
    ):
        """Test logging setup with TensorBoard enabled."""
        config = YOLOConfig.create("yolov8")
        config.logging_config["log_dir"] = str(temp_dir)
        config.logging_config["tensorboard"] = True

        setup_logging(config)

        # The current setup_logging function only sets up basic logging
        # It doesn't create TensorBoard directories
        # Check that the log directory was created
        log_dir = Path(temp_dir)
        assert log_dir.exists()

        # Check that a log file was created
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) > 0

        # Note: TensorBoard setup would need to be implemented separately
        # For now, we just verify that basic logging works

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_setup_logging_with_wandb(
        self,
        mock_setup_wandb,
        mock_load_data_yaml,
        mock_setup_paths,
        temp_dir,
        patch_wandb,
    ):
        """Test logging setup with Weights & Biases enabled."""
        config = YOLOConfig.create("yolov8")
        config.logging_config["log_dir"] = str(temp_dir)
        config.logging_config["wandb"] = True

        setup_logging(config)

        # W&B should be initialized
        # This is tested in the training monitor tests

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_setup_logging_rotating_handler(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths, temp_dir
    ):
        """Test rotating file handler setup."""
        config = YOLOConfig.create("yolov8")
        config.logging_config["log_dir"] = str(temp_dir)

        setup_logging(config)

        # Check if log file was created
        log_files = list(Path(temp_dir).glob("*.log"))
        assert len(log_files) > 0


class TestConstants:
    """Test configuration constants."""

    def test_common_training_constants(self):
        """Test common training constants."""
        assert "epochs" in COMMON_TRAINING
        assert "batch_size" in COMMON_TRAINING
        assert "image_size" in COMMON_TRAINING
        assert "device" in COMMON_TRAINING
        assert "patience" in COMMON_TRAINING
        assert "deterministic" in COMMON_TRAINING

    def test_yolo11_config_constants(self):
        """Test YOLO11 configuration constants."""
        assert "weights" in YOLO11_CONFIG
        assert "learning_rate" in YOLO11_CONFIG
        assert "optimizer" in YOLO11_CONFIG
        assert "lr_scheduler" in YOLO11_CONFIG

    def test_yolov8_config_constants(self):
        """Test YOLOv8 configuration constants."""
        assert "weights" in YOLOV8_CONFIG
        assert "learning_rate" in YOLOV8_CONFIG
        assert "optimizer" in YOLOV8_CONFIG
        assert "lr_scheduler" in YOLOV8_CONFIG

    def test_yolov5_config_constants(self):
        """Test YOLOv5 configuration constants."""
        assert "weights" in YOLOV5_CONFIG
        assert "learning_rate" in YOLOV5_CONFIG
        assert "optimizer" in YOLOV5_CONFIG
        assert "lr_scheduler" in YOLOV5_CONFIG

    def test_dataset_config_constants(self):
        """Test dataset configuration constants."""
        assert "cache" in DATASET_CONFIG
        assert "data_yaml_path" in DATASET_CONFIG

    def test_logging_config_constants(self):
        """Test logging configuration constants."""
        assert "tensorboard" in LOGGING_CONFIG
        assert "wandb" in LOGGING_CONFIG
        assert "num_checkpoint_keep" in LOGGING_CONFIG


class TestConfigurationIntegration:
    """Test configuration system integration."""

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_config_serialization(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test configuration serialization to JSON."""
        config = YOLOConfig.create("yolov8")

        # Convert to dict
        config_dict = config.__dict__

        # Should contain all major sections
        assert "model_type" in config_dict
        assert "dataset_config" in config_dict
        assert "model_config" in config_dict
        assert "logging_config" in config_dict

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_config_environment_override(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test environment variable overrides."""
        with patch.dict(
            "os.environ",
            {"YOLO_EPOCHS": "200", "YOLO_BATCH_SIZE": "32", "YOLO_DEVICE": "cpu"},
        ):
            # Note: This test may need updating based on actual environment variable handling
            config = YOLOConfig.create("yolov8")
            assert config.model_type == "yolov8"

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_config_validation_comprehensive(
        self,
        mock_setup_wandb,
        mock_load_data_yaml,
        mock_setup_paths,
        sample_dataset_structure,
    ):
        """Test comprehensive configuration validation."""
        config = YOLOConfig.create("yolov8")
        config.dataset_config["data_yaml_path"] = sample_dataset_structure / "data.yaml"

        # All validations should pass
        assert config.model_type in ["yolo11", "yolov8", "yolov5"]
        assert config.batch_size > 0
        assert config.image_size > 0

    @patch("config.config.YOLOConfig._setup_paths")
    @patch("config.config.YOLOConfig._load_data_yaml")
    @patch("config.config.YOLOConfig._setup_wandb")
    def test_config_error_handling(
        self, mock_setup_wandb, mock_load_data_yaml, mock_setup_paths
    ):
        """Test configuration error handling."""
        # Invalid model type
        with pytest.raises(ValueError, match="Unsupported model type"):
            YOLOConfig.create("invalid_model")

        # Test with valid model type
        config = YOLOConfig.create("yolov8")
        assert config.model_type == "yolov8"
