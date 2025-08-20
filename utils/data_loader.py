"""
Data loading utilities for YOLO training.
Handles dataset loading, validation, and augmentation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import yaml

logger = logging.getLogger(__name__)


class YOLODataset(Dataset):
    """Custom Dataset for YOLO training."""

    def __init__(
        self,
        data_path: Path,
        split: str = "train",
        image_size: int = 640,
        augment: bool = True,
        cache: bool = False,
        rect: bool = False,
        single_cls: bool = False,
        stride: int = 32,
        pad: float = 0.0,
        prefix: str = "",
    ):
        """
        Initialize YOLO dataset.

        Args:
            data_path: Path to dataset root directory
            split: Dataset split ('train', 'valid', 'test')
            image_size: Input image size
            augment: Whether to apply augmentations
            cache: Cache images in memory
            rect: Rectangular training
            single_cls: Single class mode
            stride: Model stride
            pad: Padding for rectangular training
            prefix: Prefix for logging
        """
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.cache = cache
        self.rect = rect
        self.single_cls = single_cls
        self.stride = stride
        self.pad = pad
        self.prefix = prefix

        # Validate paths - support both nested and flat structure
        self.images_path = self.data_path / split
        self.labels_path = self.data_path / split / "labels"

        if not self.images_path.exists():
            raise FileNotFoundError(f"Images path not found: {self.images_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels path not found: {self.labels_path}")

        # Load image files
        self.image_files = sorted(
            [
                f
                for f in self.images_path.glob("*")
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ]
        )

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {self.images_path}")

        logger.info(
            f"{self.prefix}Found {len(self.image_files)} images in {split} split"
        )

        # Load class names from data.yaml
        self.class_names = self._load_class_names()

        # Cache images if requested
        self.cached_images = {}
        if self.cache:
            self._cache_images()

    def _load_class_names(self) -> List[str]:
        """Load class names from data.yaml."""
        data_yaml_path = self.data_path / "data.yaml"
        if data_yaml_path.exists():
            with open(data_yaml_path) as f:
                data_yaml = yaml.safe_load(f)
                return data_yaml.get("names", [])
        else:
            logger.warning("data.yaml not found, using default class names")
            return [f"class_{i}" for i in range(1000)]  # Default fallback

    def _cache_images(self):
        """Cache images in memory for faster training."""
        logger.info(f"{self.prefix}Caching {len(self.image_files)} images...")
        for i, img_path in enumerate(self.image_files):
            if i % 100 == 0:
                logger.info(f"{self.prefix}Cached {i}/{len(self.image_files)} images")
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.cached_images[img_path] = img
            except Exception as e:
                logger.warning(f"Failed to cache {img_path}: {e}")
        logger.info(f"{self.prefix}Image caching completed")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        """
        Get dataset item.

        Returns:
            image: Preprocessed image tensor
            labels: Label tensor
            image_path: Path to original image
            original_shape: Original image shape
        """
        img_path = self.image_files[idx]

        # Find corresponding label file by matching the base name
        img_base = img_path.stem  # Remove extension
        label_path = self.labels_path / f"{img_base}.txt"

        # Load image
        if self.cache and img_path in self.cached_images:
            img = self.cached_images[img_path].copy()
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")

        # Load labels
        labels = self._load_labels(label_path, img.shape)

        # Preprocess image and labels
        img, labels, original_shape = self._preprocess(img, labels)

        # Convert to tensors
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        labels_tensor = torch.from_numpy(labels).float()

        return img_tensor, labels_tensor, str(img_path), torch.tensor(original_shape)

    def _load_labels(
        self, label_path: Path, img_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Load and validate labels."""
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        try:
            with open(label_path) as f:
                labels = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if self.single_cls:
                            class_id = 0

                        # Convert normalized coordinates to absolute
                        x_center = float(parts[1]) * img_shape[1]
                        y_center = float(parts[2]) * img_shape[1]
                        width = float(parts[3]) * img_shape[1]
                        height = float(parts[4]) * img_shape[1]

                        labels.append([class_id, x_center, y_center, width, height])

                return (
                    np.array(labels, dtype=np.float32)
                    if labels
                    else np.zeros((0, 5), dtype=np.float32)
                )

        except Exception as e:
            logger.warning(f"Failed to load labels from {label_path}: {e}")
            return np.zeros((0, 5), dtype=np.float32)

    def _preprocess(
        self, img: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Preprocess image and labels."""
        original_shape = img.shape[:2]  # (height, width)

        # Resize image
        if self.rect:
            # Rectangular training
            img, labels = self._resize_rectangular(img, labels)
        else:
            # Square training
            img, labels = self._resize_square(img, labels)

        # Apply augmentations if training
        if self.augment and self.split == "train":
            img, labels = self._apply_augmentations(img, labels)

        return img, labels, original_shape

    def _resize_square(
        self, img: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image to square with padding."""
        h, w = img.shape[:2]
        scale = min(self.image_size / h, self.image_size / w)

        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # Create square image with padding
        img_square = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        y_offset = (self.image_size - new_h) // 2
        x_offset = (self.image_size - new_w) // 2
        img_square[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
            img_resized
        )

        # Adjust labels
        if len(labels) > 0:
            labels[:, 1] = labels[:, 1] * scale + x_offset  # x_center
            labels[:, 2] = labels[:, 2] * scale + y_offset  # y_center
            labels[:, 3] = labels[:, 3] * scale  # width
            labels[:, 4] = labels[:, 4] * scale  # height

        return img_square, labels

    def _resize_rectangular(
        self, img: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image to rectangular shape."""
        h, w = img.shape[:2]

        # Calculate new dimensions maintaining aspect ratio
        scale = min(self.image_size / h, self.image_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Ensure dimensions are multiples of stride
        new_h = (new_h // self.stride) * self.stride
        new_w = (new_w // self.stride) * self.stride

        img_resized = cv2.resize(img, (new_w, new_h))

        # Adjust labels
        if len(labels) > 0:
            labels[:, 1] = labels[:, 1] * scale  # x_center
            labels[:, 2] = labels[:, 2] * scale  # y_center
            labels[:, 3] = labels[:, 3] * scale  # width
            labels[:, 4] = labels[:, 4] * scale  # height

        return img_resized, labels

    def _apply_augmentations(
        self, img: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentations to image and labels."""
        # This is a simplified version - in practice, you'd use more sophisticated augmentation
        # For now, just apply basic augmentations

        # Random horizontal flip
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
            if len(labels) > 0:
                labels[:, 1] = img.shape[1] - labels[:, 1]  # Flip x_center

        # Random brightness/contrast
        if np.random.random() < 0.5:
            alpha = 1.0 + np.random.uniform(-0.1, 0.1)  # Contrast
            beta = np.random.uniform(-10, 10)  # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img, labels


def create_dataloader(
    config: "YOLOConfig",
    split: str = "train",
    augment: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader for YOLO dataset.

    Args:
        config: YOLO configuration
        split: Dataset split ('train', 'valid', 'test')
        augment: Whether to apply augmentations
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    dataset = YOLODataset(
        data_path=Path(config.dataset_config["data_yaml_path"]).parent,
        split=split,
        image_size=config.image_size,
        augment=augment and split == "train",
        cache=config.dataset_config.get("cache", False),
        rect=config.rect,
        single_cls=config.single_cls,
        stride=32,  # Default stride
        prefix=f"[{split.upper()}] ",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=split == "train",
    )

    return dataloader


def collate_fn(batch):
    """Custom collate function for YOLO training."""
    images, labels, paths, shapes = zip(*batch)

    # Stack images
    images = torch.stack(images, 0)

    # Stack labels - pad to max number of objects
    max_objects = max(label.shape[0] for label in labels)
    padded_labels = []
    for label in labels:
        if label.shape[0] < max_objects:
            # Pad with zeros
            padding = torch.zeros(
                max_objects - label.shape[0], label.shape[1], dtype=label.dtype
            )
            padded_label = torch.cat([label, padding], dim=0)
            padded_labels.append(padded_label)
        else:
            padded_labels.append(label)

    labels = torch.stack(padded_labels, 0)

    # Stack shapes
    shapes = torch.stack(shapes, 0)

    return images, labels, paths, shapes


def validate_dataset_structure(data_path: Path) -> bool:
    """
    Validate YOLO dataset structure.

    Args:
        data_path: Path to dataset root directory

    Returns:
        True if structure is valid, False otherwise
    """
    required_dirs = [
        "train",
        "train/labels",
        "valid",
        "valid/labels",
        "test",
        "test/labels",
    ]

    required_files = ["data.yaml"]

    # Check directories
    for dir_path in required_dirs:
        if not (data_path / dir_path).exists():
            logger.error(f"Missing directory: {dir_path}")
            return False

    # Check files
    for file_path in required_files:
        if not (data_path / file_path).exists():
            logger.error(f"Missing file: {file_path}")
            return False

    # Check data.yaml content
    yaml_path = data_path / "data.yaml"
    try:
        with open(yaml_path) as f:
            data_yaml = yaml.safe_load(f)

        required_keys = ["path", "train", "val", "nc", "names"]
        if not all(key in data_yaml for key in required_keys):
            logger.error(f"data.yaml missing required keys: {required_keys}")
            return False

        # Check class count
        if data_yaml["nc"] != len(data_yaml["names"]):
            logger.error("Class count mismatch in data.yaml")
            return False

        logger.info(f"Dataset validation passed. Found {data_yaml['nc']} classes")
        return True

    except Exception as e:
        logger.error(f"Failed to validate data.yaml: {e}")
        return False
