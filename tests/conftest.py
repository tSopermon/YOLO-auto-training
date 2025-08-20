"""
Pytest configuration and common fixtures for YOLO training tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
import cv2
from unittest.mock import Mock, patch
import uuid

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import YOLOConfig
from config.constants import COMMON_TRAINING, YOLOV8_CONFIG


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=YOLOConfig)
    config.model_type = "yolov8"
    config.epochs = 10
    config.batch_size = 4
    config.image_size = 640
    config.device = "cpu"
    config.patience = 5
    config.deterministic = False
    config.single_cls = False
    config.rect = False
    config.cos_lr = True
    config.close_mosaic = 5
    config.resume = False
    config.weights = "yolov8n.pt"
    config.pretrained = True
    config.num_workers = 2

    # Dataset config
    config.dataset_config = {
        "data_yaml_path": Path("tests/fixtures/dataset/data.yaml"),
        "cache": False,
    }

    # Model config
    config.model_config = {
        "learning_rate": 0.01,
        "optimizer": "auto",
        "lr_scheduler": "cosine",
        "weight_decay": 0.0005,
        "min_lr": 1e-6,
    }

    # Logging config
    config.logging_config = {
        "log_dir": "tests/fixtures/logs",
        "tensorboard": False,
        "wandb": False,
        "num_checkpoint_keep": 3,
        "save_period": -1,
        "log_metrics_interval": 10,
    }

    # Augmentation config
    config.augmentation_config = {
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "blur": 0.01,
        "grayscale": 0.0,
        "auto_augment": False,
    }

    # Eval config
    config.eval_config = {
        "plots": True,
        "save_json": False,
        "save_hybrid": False,
        "conf": 0.001,
        "iou_thres": 0.6,
        "max_det": 300,
        "half": True,
    }

    return config


@pytest.fixture
def sample_dataset_structure(temp_dir):
    """Create a sample dataset structure for testing."""
    # Use unique identifier to avoid conflicts
    unique_id = str(uuid.uuid4())[:8]
    dataset_dir = temp_dir / f"dataset_{unique_id}"
    dataset_dir.mkdir()

    # Create splits
    for split in ["train", "valid", "test"]:
        (dataset_dir / split / "labels").mkdir(parents=True)

    # Create sample images
    for split in ["train", "valid", "test"]:
        for i in range(5):  # 5 images per split to ensure batch_size=4 works
            img_path = dataset_dir / split / f"image_{i}.jpg"
            # Create a real image file that OpenCV can read
            img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img_array)

    # Create sample labels
    for split in ["train", "valid", "test"]:
        for i in range(5):  # 5 labels per split to ensure batch_size=4 works
            label_path = dataset_dir / split / "labels" / f"image_{i}.txt"
            # Create sample YOLO format labels
            label_content = f"0 0.5 0.5 0.2 0.3\n1 0.7 0.3 0.1 0.2"
            label_path.write_text(label_content)

    # Create data.yaml
    data_yaml = {
        "path": str(dataset_dir),
        "train": "train",
        "val": "valid",
        "test": "test",
        "nc": 2,
        "names": ["class_0", "class_1"],
    }

    import yaml

    with open(dataset_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    return dataset_dir


@pytest.fixture
def sample_checkpoint():
    """Create a sample checkpoint for testing."""
    checkpoint = {
        "epoch": 5,
        "model": {"layer1.weight": torch.randn(10, 10)},
        "optimizer": {"param_groups": [{"lr": 0.01}]},
        "metrics": {"train_loss": 0.5, "val_loss": 0.6},
        "config": {"model_type": "yolov8", "image_size": 640, "batch_size": 16},
    }
    return checkpoint


@pytest.fixture
def mock_model():
    """Create a mock PyTorch model for testing."""
    model = Mock()
    model.parameters.return_value = [torch.randn(10, 10)]
    model.state_dict.return_value = {"layer1.weight": torch.randn(10, 10)}
    model.to.return_value = model
    model.train.return_value = None
    model.eval.return_value = None
    return model


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for testing."""
    optimizer = Mock()
    optimizer.param_groups = [{"lr": 0.01}]
    optimizer.state_dict.return_value = {"param1": "value1"}
    optimizer.load_state_dict.return_value = None
    optimizer.zero_grad.return_value = None
    optimizer.step.return_value = None
    return optimizer


@pytest.fixture
def sample_batch():
    """Create a sample training batch for testing."""
    batch_size = 4
    image_size = 640
    num_classes = 2

    # Sample images (batch_size, channels, height, width)
    images = torch.randn(batch_size, 3, image_size, image_size)

    # Sample labels (batch_size, max_objects, 5) - class, x, y, w, h
    labels = torch.zeros(batch_size, 10, 5)
    labels[:, 0, :] = torch.tensor([0, 0.5, 0.5, 0.2, 0.3])  # First object
    labels[:, 1, :] = torch.tensor([1, 0.7, 0.3, 0.1, 0.2])  # Second object

    # Sample paths
    paths = [f"image_{i}.jpg" for i in range(batch_size)]

    # Sample shapes
    shapes = torch.tensor([[640, 640] for _ in range(batch_size)])

    return images, labels, paths, shapes


@pytest.fixture
def mock_roboflow_response():
    """Create a mock Roboflow API response for testing."""
    response = Mock()
    response.location = "tests/fixtures/dataset"
    response.version = "1"
    response.project = "test_project"
    return response


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create test fixtures directory
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(exist_ok=True)

    # Create logs directory
    logs_dir = fixtures_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    yield

    # Cleanup after tests
    if fixtures_dir.exists():
        shutil.rmtree(fixtures_dir)


@pytest.fixture
def patch_ultralytics():
    """Patch Ultralytics imports for testing."""
    with patch.dict(
        "sys.modules",
        {
            "ultralytics": Mock(),
            "ultralytics.YOLO": Mock(),
            "ultralytics.utils.torch_utils": Mock(),
        },
    ):
        yield


@pytest.fixture
def patch_wandb():
    """Patch Weights & Biases imports for testing."""
    with patch.dict("sys.modules", {"wandb": Mock()}):
        yield


@pytest.fixture
def patch_tensorboard():
    """Patch TensorBoard imports for testing."""
    with patch.dict(
        "sys.modules",
        {
            "torch.utils.tensorboard": Mock(),
            "torch.utils.tensorboard.SummaryWriter": Mock(),
        },
    ):
        yield
