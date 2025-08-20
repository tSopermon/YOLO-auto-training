"""
Tests for utility modules.
Tests evaluation, export, training, and training monitor functionality.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Import utilities to test
from utils.evaluation import YOLOEvaluator, evaluate_model, visualize_predictions
from utils.export_utils import YOLOExporter, export_model, export_to_onnx
from utils.training import YOLOTrainer, train_model, validate_model
from utils.training_monitor import TrainingMonitor, create_training_monitor


class MockYOLOConfig:
    """Mock configuration for testing."""

    def __init__(self):
        self.model_type = "yolov8"
        self.epochs = 10
        self.batch_size = 4
        self.image_size = 640
        self.device = "cpu"
        self.patience = 5
        self.deterministic = True
        self.single_cls = False
        self.rect = False
        self.cos_lr = True
        self.close_mosaic = 10
        self.resume = False
        self.num_workers = 2
        self.pretrained = True

        self.model_config = {
            "learning_rate": 0.01,
            "optimizer": "auto",
            "lr_scheduler": "cosine",
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "weight_decay": 0.0005,
        }

        self.dataset_config = {
            "data_yaml_path": Path("tests/fixtures/dataset/data.yaml"),
            "cache": False,
        }

        self.eval_config = {
            "conf_thres": 0.001,
            "iou_thres": 0.6,
            "max_det": 300,
            "half": True,
        }

        self.logging_config = {
            "log_dir": Path("tests/logs"),
            "tensorboard": True,
            "wandb": False,
            "wandb_project": "yolo_training",
            "log_metrics_interval": 20,
            "save_checkpoint_interval": 10,
            "num_checkpoint_keep": 5,
            "save_period": -1,
        }

        self.export_config = {
            "export_formats": ["onnx", "torchscript"],
            "export_dir": Path("tests/exports"),
            "include_nms": True,
            "batch_size": 1,
            "half": True,
            "int8": False,
            "simplify": True,
            "dynamic": False,
        }

        self.augmentation_config = {
            "mosaic": 1.0,
            "mixup": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
        }

        self.data_yaml = {"names": ["class_0", "class_1"], "nc": 2}

    def get_optimizer_config(self):
        return self.model_config

    def get_scheduler_config(self):
        return self.model_config

    def get_training_args(self):
        return {
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.image_size,
            "device": self.device,
        }


class MockModel(nn.Module):
    """Mock YOLO model for testing."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes * 6)  # 6 values per detection

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # Reshape to (batch_size, num_detections, 6)
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_classes, 6)
        return x


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(self, num_batches=2, batch_size=4):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.dataset = Mock()

        # Create a proper method for __len__
        def dataset_len(self):
            return num_batches * batch_size

        self.dataset.__len__ = dataset_len

    def __iter__(self):
        for i in range(self.num_batches):
            # Create mock batch data
            images = torch.randn(self.batch_size, 3, 640, 640)
            labels = torch.zeros(self.batch_size, 10, 5)
            labels[:, 0, :] = torch.tensor([0, 0.5, 0.5, 0.2, 0.3])
            paths = [f"image_{i}_{j}.jpg" for j in range(self.batch_size)]
            shapes = torch.tensor([[640, 640]] * self.batch_size)

            yield images, labels, paths, shapes

    def __len__(self):
        return self.num_batches


class MockCheckpointManager:
    """Mock checkpoint manager for testing."""

    def __init__(self):
        self.checkpoints = []

    def save_checkpoint(self, checkpoint, epoch, metrics, is_best=False):
        checkpoint_info = {
            "path": f"checkpoint_epoch_{epoch:03d}.pt",
            "epoch": epoch,
            "metrics": metrics,
            "is_best": is_best,
        }
        self.checkpoints.append(checkpoint_info)
        return Path(checkpoint_info["path"])

    def get_latest_checkpoint(self):
        if self.checkpoints:
            return Path(self.checkpoints[-1]["path"])
        return None


class TestEvaluation:
    """Test evaluation utilities."""

    def test_yolo_evaluator_initialization(self):
        """Test YOLOEvaluator initialization."""
        config = MockYOLOConfig()
        model = MockModel()
        class_names = ["class_0", "class_1"]

        evaluator = YOLOEvaluator(model, config, class_names, "cpu")

        assert evaluator.model == model
        assert evaluator.config == config
        assert evaluator.class_names == class_names
        assert evaluator.device == "cpu"

    def test_evaluate_dataset(self):
        """Test dataset evaluation."""
        config = MockYOLOConfig()
        model = MockModel()
        class_names = ["class_0", "class_1"]
        dataloader = MockDataLoader()

        evaluator = YOLOEvaluator(model, config, class_names, "cpu")

        with patch.object(evaluator, "_get_predictions") as mock_predictions:
            mock_predictions.return_value = torch.randn(4, 2, 6)

            metrics = evaluator.evaluate_dataset(dataloader)

            assert isinstance(metrics, dict)
            assert "mAP0.5" in metrics
            assert "mAP50-95" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1_score" in metrics

    def test_visualize_predictions(self):
        """Test prediction visualization."""
        image = torch.randn(640, 640, 3).numpy()
        predictions = torch.tensor([[100, 100, 200, 200, 0.9, 0]]).numpy()
        class_names = ["class_0", "class_1"]

        # Test visualization function
        result = visualize_predictions(image, predictions, class_names)

        assert result.shape == image.shape
        assert result.dtype == image.dtype


class TestExport:
    """Test export utilities."""

    def test_yolo_exporter_initialization(self):
        """Test YOLOExporter initialization."""
        config = MockYOLOConfig()
        model = MockModel()

        exporter = YOLOExporter(model, config)

        assert exporter.model == model
        assert exporter.config == config
        assert exporter.export_dir.exists()

    def test_export_to_onnx(self):
        """Test ONNX export."""
        config = MockYOLOConfig()
        model = MockModel()

        exporter = YOLOExporter(model, config)

        with patch("torch.onnx.export") as mock_export:
            mock_export.return_value = None

            onnx_path = exporter._export_to_onnx()

            assert onnx_path is not None
            assert onnx_path.suffix == ".onnx"

    def test_export_to_torchscript(self):
        """Test TorchScript export."""
        config = MockYOLOConfig()
        model = MockModel()

        exporter = YOLOExporter(model, config)

        with patch("torch.jit.script") as mock_script:
            with patch("torch.jit.save") as mock_save:
                with patch("torch.jit.load") as mock_load:
                    with patch("pathlib.Path.exists") as mock_exists:
                        # Mock the file operations
                        mock_script.return_value = Mock()
                        mock_save.return_value = None
                        mock_load.return_value = Mock()
                        mock_exists.return_value = True

                        torchscript_path = exporter._export_to_torchscript()

                        assert torchscript_path is not None
                        assert torchscript_path.suffix == ".pt"

    def test_export_all_formats(self):
        """Test exporting to all formats."""
        config = MockYOLOConfig()
        model = MockModel()

        exporter = YOLOExporter(model, config)

        with patch.object(exporter, "_export_to_onnx") as mock_onnx:
            with patch.object(exporter, "_export_to_torchscript") as mock_torchscript:
                mock_onnx.return_value = Path("test.onnx")
                mock_torchscript.return_value = Path("test.pt")

                exported_models = exporter.export_all_formats()

                assert "onnx" in exported_models
                assert "torchscript" in exported_models
                assert len(exported_models) == 2


class TestTraining:
    """Test training utilities."""

    def test_yolo_trainer_initialization(self):
        """Test YOLOTrainer initialization."""
        config = MockYOLOConfig()
        model = MockModel()
        train_loader = MockDataLoader()
        val_loader = MockDataLoader()
        checkpoint_manager = MockCheckpointManager()

        trainer = YOLOTrainer(
            model, config, train_loader, val_loader, checkpoint_manager, "cpu"
        )

        assert trainer.model == model
        assert trainer.config == config
        assert trainer.train_loader == train_loader
        assert trainer.val_loader == val_loader
        assert trainer.checkpoint_manager == checkpoint_manager

    def test_setup_optimizer(self):
        """Test optimizer setup."""
        config = MockYOLOConfig()
        model = MockModel()
        train_loader = MockDataLoader()
        val_loader = MockDataLoader()
        checkpoint_manager = MockCheckpointManager()

        trainer = YOLOTrainer(
            model, config, train_loader, val_loader, checkpoint_manager, "cpu"
        )

        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)

    def test_setup_scheduler(self):
        """Test scheduler setup."""
        config = MockYOLOConfig()
        model = MockModel()
        train_loader = MockDataLoader()
        val_loader = MockDataLoader()
        checkpoint_manager = MockCheckpointManager()

        trainer = YOLOTrainer(
            model, config, train_loader, val_loader, checkpoint_manager, "cpu"
        )

        # Scheduler might be None depending on config
        if trainer.scheduler is not None:
            # Check if it's a valid PyTorch scheduler
            assert hasattr(trainer.scheduler, "step")
            assert hasattr(trainer.scheduler, "get_last_lr")
            # logger.info(f"Scheduler type: {type(trainer.scheduler).__name__}") # This line was not in the original file, so it's not added.
        else:
            # logger.info("No scheduler configured") # This line was not in the original file, so it's not added.
            pass

    def test_train_epoch(self):
        """Test training for one epoch."""
        config = MockYOLOConfig()
        model = MockModel()
        train_loader = MockDataLoader()
        val_loader = MockDataLoader()
        checkpoint_manager = MockCheckpointManager()

        trainer = YOLOTrainer(
            model, config, train_loader, val_loader, checkpoint_manager, "cpu"
        )

        with patch.object(trainer, "criterion") as mock_criterion:
            mock_criterion.return_value = torch.tensor(0.5, requires_grad=True)

            metrics = trainer._train_epoch(0)

            assert "train_loss" in metrics
            assert "learning_rate" in metrics
            assert isinstance(metrics["train_loss"], float)
            assert isinstance(metrics["learning_rate"], float)

    def test_validate_epoch(self):
        """Test validation for one epoch."""
        config = MockYOLOConfig()
        model = MockModel()
        train_loader = MockDataLoader()
        val_loader = MockDataLoader()
        checkpoint_manager = MockCheckpointManager()

        trainer = YOLOTrainer(
            model, config, train_loader, val_loader, checkpoint_manager, "cpu"
        )

        with patch.object(trainer, "criterion") as mock_criterion:
            mock_criterion.return_value = torch.tensor(0.6)

            metrics = trainer._validate_epoch(0)

            assert "val_loss" in metrics
            assert isinstance(metrics["val_loss"], float)


class TestTrainingMonitor:
    """Test training monitor."""

    def test_training_monitor_initialization(self):
        """Test TrainingMonitor initialization."""
        config = MockYOLOConfig()
        log_dir = Path("tests/logs")

        monitor = TrainingMonitor(config, log_dir)

        assert monitor.config == config
        assert monitor.log_dir == log_dir
        assert log_dir.exists()

    def test_log_epoch_metrics(self):
        """Test logging epoch metrics."""
        config = MockYOLOConfig()
        log_dir = Path("tests/logs")

        monitor = TrainingMonitor(
            config, log_dir, enable_tensorboard=False, enable_wandb=False
        )

        train_metrics = {"train_loss": 0.5}
        val_metrics = {"val_loss": 0.6}
        learning_rate = 0.01

        monitor.log_epoch_metrics(1, train_metrics, val_metrics, learning_rate)

        assert "epoch" in monitor.metrics_history
        assert "train_train_loss" in monitor.metrics_history  # Prefixed with 'train_'
        assert "val_val_loss" in monitor.metrics_history  # Prefixed with 'val_'
        assert "learning_rate" in monitor.metrics_history

    def test_create_training_plots(self):
        """Test creating training plots."""
        config = MockYOLOConfig()
        log_dir = Path("tests/logs")

        monitor = TrainingMonitor(
            config, log_dir, enable_tensorboard=False, enable_wandb=False
        )

        # Add some mock metrics
        monitor.metrics_history["train_loss"] = [0.5, 0.4, 0.3]
        monitor.metrics_history["val_loss"] = [0.6, 0.5, 0.4]
        monitor.metrics_history["learning_rate"] = [0.01, 0.008, 0.006]

        # Test plot creation
        monitor.create_training_plots(save_plots=False)

        # Check if plots directory was created
        plots_dir = log_dir / "plots"
        assert plots_dir.exists()

    def test_generate_training_report(self):
        """Test training report generation."""
        config = MockYOLOConfig()
        log_dir = Path("tests/logs")

        monitor = TrainingMonitor(
            config, log_dir, enable_tensorboard=False, enable_wandb=False
        )

        # Add some mock metrics
        monitor.metrics_history["epoch"] = [1, 2, 3]
        monitor.metrics_history["train_loss"] = [0.5, 0.4, 0.3]
        monitor.metrics_history["val_loss"] = [0.6, 0.5, 0.4]
        monitor.metrics_history["learning_rate"] = [0.01, 0.008, 0.006]

        report = monitor.generate_training_report()

        assert isinstance(report, str)
        assert "YOLO Training Report" in report
        assert "Model Type: yolov8" in report
        assert "Total Epochs: 3" in report

    def test_create_training_monitor(self):
        """Test create_training_monitor factory function."""
        config = MockYOLOConfig()
        log_dir = Path("tests/logs")

        monitor = create_training_monitor(config, log_dir)

        assert isinstance(monitor, TrainingMonitor)
        assert monitor.config == config
        assert monitor.log_dir == log_dir


class TestIntegration:
    """Test integration between utility modules."""

    def test_full_training_workflow(self):
        """Test complete training workflow integration."""
        config = MockYOLOConfig()
        model = MockModel()
        train_loader = MockDataLoader()
        val_loader = MockDataLoader()
        checkpoint_manager = MockCheckpointManager()

        # Test trainer creation
        trainer = YOLOTrainer(
            model, config, train_loader, val_loader, checkpoint_manager, "cpu"
        )

        # Test monitor creation
        monitor = TrainingMonitor(config, config.logging_config["log_dir"])

        # Test evaluation
        evaluator = YOLOEvaluator(model, config, ["class_0", "class_1"], "cpu")

        # Test export
        exporter = YOLOExporter(model, config)

        assert trainer is not None
        assert monitor is not None
        assert evaluator is not None
        assert exporter is not None

    def test_utility_functions(self):
        """Test standalone utility functions."""
        config = MockYOLOConfig()
        model = MockModel()
        train_loader = MockDataLoader()
        val_loader = MockDataLoader()
        checkpoint_manager = MockCheckpointManager()

        # Test train_model function
        with patch.object(YOLOTrainer, "train") as mock_train:
            mock_train.return_value = {"final_epoch": 5, "best_metric": 0.3}

            results = train_model(
                model, config, train_loader, val_loader, checkpoint_manager, "cpu"
            )

            assert "final_epoch" in results
            assert "best_metric" in results

        # Test validate_model function
        with patch.object(YOLOEvaluator, "evaluate_dataset") as mock_eval:
            mock_eval.return_value = {"mAP0.5": 0.8, "precision": 0.85}

            metrics = validate_model(
                model, config, val_loader, ["class_0", "class_1"], "cpu"
            )

            assert "mAP0.5" in metrics
            assert "precision" in metrics


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up test files after each test."""
    yield

    # Clean up test directories
    test_dirs = ["tests/logs", "tests/exports", "tests/fixtures"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)

    # Clean up test files
    test_files = ["test.onnx", "test.pt"]
    for test_file in test_files:
        if Path(test_file).exists():
            Path(test_file).unlink()
