"""
Automated Dataset Preparation System for YOLO Training.
Automatically detects, validates, and prepares datasets for any YOLO model.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml
import json
from dataclasses import dataclass
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about detected dataset."""

    structure_type: str  # 'flat', 'nested', 'mixed'
    has_images: bool
    has_labels: bool
    image_formats: List[str]
    label_formats: List[str]
    class_count: int
    class_names: List[str]
    total_images: int
    splits: Dict[str, int]
    is_yolo_ready: bool
    issues: List[str]


class AutoDatasetPreparer:
    """Automatically prepares datasets for YOLO training."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.dataset_info = None
        self.prepared_path = None

    def prepare_dataset(self, target_format: str = "yolo") -> Path:
        """
        Automatically prepare dataset for YOLO training.

        Args:
            target_format: Target format ('yolo', 'yolov8', 'yolov5', etc.)

        Returns:
            Path to prepared dataset
        """
        logger.info(f"Starting automatic dataset preparation for {target_format}")

        # Step 1: Analyze current dataset structure
        self.dataset_info = self._analyze_dataset_structure()

        # Step 2: Detect and fix issues
        self._detect_and_fix_issues()

        # Step 3: Reorganize to standard YOLO structure
        self.prepared_path = self._reorganize_to_yolo_structure()

        # Step 4: Generate data.yaml
        self._generate_data_yaml(target_format)

        # Step 5: Validate final structure
        if not self._validate_final_structure():
            raise RuntimeError("Dataset preparation failed validation")

        logger.info(f"Dataset preparation completed successfully")
        logger.info(f"Prepared dataset location: {self.prepared_path}")

        return self.prepared_path

    def _analyze_dataset_structure(self) -> DatasetInfo:
        """Analyze current dataset structure and detect format."""
        logger.info("Analyzing dataset structure...")

        # Check for different possible structures
        structures = {
            "flat": self._check_flat_structure(),
            "nested": self._check_nested_structure(),
            "mixed": self._check_mixed_structure(),
        }

        # Determine the actual structure
        structure_type = None
        for struct_name, struct_info in structures.items():
            if struct_info["valid"]:
                structure_type = struct_name
                break

        if not structure_type:
            raise ValueError("Could not determine dataset structure")

        # Analyze content
        image_formats = self._detect_image_formats()
        label_formats = self._detect_label_formats()
        class_info = self._detect_classes()
        splits = self._detect_splits()

        # Check if already YOLO-ready
        is_yolo_ready = self._check_yolo_readiness()

        dataset_info = DatasetInfo(
            structure_type=structure_type,
            has_images=len(image_formats) > 0,
            has_labels=len(label_formats) > 0,
            image_formats=image_formats,
            label_formats=label_formats,
            class_count=class_info["count"],
            class_names=class_info["names"],
            total_images=class_info["total_images"],
            splits=splits,
            is_yolo_ready=is_yolo_ready,
            issues=[],
        )

        logger.info(f"Dataset structure: {structure_type}")
        logger.info(f"Image formats: {image_formats}")
        logger.info(f"Label formats: {label_formats}")
        logger.info(
            f"Classes: {class_info['count']} ({', '.join(class_info['names'])})"
        )

        return dataset_info

    def _check_flat_structure(self) -> Dict[str, Any]:
        """Check if dataset has flat structure (all files in root)."""
        files = list(self.dataset_path.glob("*"))
        images = [
            f
            for f in files
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]
        labels = [
            f
            for f in files
            if f.is_file() and f.suffix.lower() in [".txt", ".json", ".xml"]
        ]

        return {"valid": len(images) > 0, "images": images, "labels": labels}

    def _check_nested_structure(self) -> Dict[str, Any]:
        """Check if dataset has nested structure (train/val/test folders)."""
        expected_splits = ["train", "val", "valid", "test"]
        found_splits = []

        for split in expected_splits:
            split_path = self.dataset_path / split
            if split_path.exists() and split_path.is_dir():
                found_splits.append(split)

        return {"valid": len(found_splits) > 0, "splits": found_splits}

    def _check_mixed_structure(self) -> Dict[str, Any]:
        """Check if dataset has mixed structure (some nested, some flat)."""
        # This is a fallback for datasets that don't fit other patterns
        return {"valid": True, "type": "mixed"}

    def _detect_image_formats(self) -> List[str]:
        """Detect image formats present in dataset."""
        formats = set()
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
            if list(self.dataset_path.rglob(f"*{ext}")):
                formats.add(ext.lstrip("."))
        return list(formats)

    def _detect_label_formats(self) -> List[str]:
        """Detect label formats present in dataset."""
        formats = set()

        # Check for YOLO format
        if list(self.dataset_path.rglob("*.txt")):
            formats.add("yolo")

        # Check for COCO format
        if list(self.dataset_path.rglob("*.json")):
            formats.add("coco")

        # Check for XML format
        if list(self.dataset_path.rglob("*.xml")):
            formats.add("xml")

        return list(formats)

    def _detect_classes(self) -> Dict[str, Any]:
        """Detect classes and count images."""
        classes = set()
        total_images = 0

        # Try to find class information from different sources
        sources = [
            self._detect_classes_from_yolo_labels,
            self._detect_classes_from_coco_annotations,
            self._detect_classes_from_class_mapping,
        ]

        for source_func in sources:
            try:
                result = source_func()
                if result:
                    classes = result["classes"]
                    total_images = result["total_images"]
                    break
            except Exception as e:
                logger.debug(
                    f"Failed to detect classes from {source_func.__name__}: {e}"
                )

        # Fallback: assume single class if nothing detected
        if not classes:
            classes = {"object"}
            total_images = len(list(self.dataset_path.rglob("*.jpg")))

        return {
            "count": len(classes),
            "names": sorted(list(classes)),
            "total_images": total_images,
        }

    def _detect_classes_from_yolo_labels(self) -> Optional[Dict[str, Any]]:
        """Detect classes from YOLO label files."""
        classes = set()
        total_images = 0

        for label_file in self.dataset_path.rglob("*.txt"):
            try:
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            classes.add(class_id)
                total_images += 1
            except Exception:
                continue

        if classes:
            # Convert numeric IDs to names
            class_names = [f"class_{i}" for i in sorted(classes)]
            return {"classes": set(class_names), "total_images": total_images}

        return None

    def _detect_classes_from_coco_annotations(self) -> Optional[Dict[str, Any]]:
        """Detect classes from COCO annotation files."""
        for json_file in self.dataset_path.rglob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                if "categories" in data:
                    classes = {cat["name"] for cat in data["categories"]}
                    total_images = len(data.get("images", []))
                    return {"classes": classes, "total_images": total_images}
            except Exception:
                continue

        return None

    def _detect_classes_from_class_mapping(self) -> Optional[Dict[str, Any]]:
        """Detect classes from class mapping file."""
        mapping_file = self.dataset_path / "class_mapping.json"
        if mapping_file.exists():
            try:
                with open(mapping_file) as f:
                    mapping = json.load(f)

                classes = set(mapping.keys())
                total_images = len(list(self.dataset_path.rglob("*.jpg")))

                return {"classes": classes, "total_images": total_images}
            except Exception:
                pass

        return None

    def _detect_splits(self) -> Dict[str, int]:
        """Detect dataset splits and count images in each."""
        splits = {}

        # Check for standard split directories
        for split in ["train", "val", "valid", "test"]:
            split_path = self.dataset_path / split
            if split_path.exists() and split_path.is_dir():
                # Count images in this split
                image_count = len(list(split_path.rglob("*.jpg")))
                if image_count > 0:
                    splits[split] = image_count

        # If no splits found, assume single split
        if not splits:
            total_images = len(list(self.dataset_path.rglob("*.jpg")))
            splits["all"] = total_images

        return splits

    def _check_yolo_readiness(self) -> bool:
        """Check if dataset is already in YOLO format."""
        # Check for standard YOLO structure
        required_dirs = ["train", "valid", "test"]
        required_files = ["data.yaml"]

        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not (
                dir_path.exists()
                and (dir_path / "images").exists()
                and (dir_path / "labels").exists()
            ):
                return False

        if not (self.dataset_path / "data.yaml").exists():
            return False

        return True

    def _detect_and_fix_issues(self):
        """Detect and fix common dataset issues."""
        logger.info("Detecting and fixing dataset issues...")

        issues = []

        # Check for missing labels
        if self.dataset_info.has_images and not self.dataset_info.has_labels:
            issues.append("Images found but no labels detected")

        # Check for empty splits
        for split_name, count in self.dataset_info.splits.items():
            if count == 0:
                issues.append(f"Split '{split_name}' has no images")

        # Check for class imbalance
        if self.dataset_info.class_count == 0:
            issues.append("No classes detected")

        self.dataset_info.issues = issues

        if issues:
            logger.warning(f"Found {len(issues)} issues: {issues}")
        else:
            logger.info("No major issues detected")

    def _reorganize_to_yolo_structure(self) -> Path:
        """Reorganize dataset to standard YOLO structure."""
        logger.info("Reorganizing dataset to YOLO structure...")

        # Create prepared dataset directory
        prepared_path = self.dataset_path.parent / f"{self.dataset_path.name}_prepared"
        if prepared_path.exists():
            shutil.rmtree(prepared_path)
        prepared_path.mkdir(parents=True, exist_ok=True)

        # Create standard YOLO structure
        for split in ["train", "valid", "test"]:
            (prepared_path / split / "images").mkdir(parents=True, exist_ok=True)
            (prepared_path / split / "labels").mkdir(parents=True, exist_ok=True)

        # Copy and reorganize files based on detected structure
        if self.dataset_info.structure_type == "nested":
            self._reorganize_nested_structure(prepared_path)
        elif self.dataset_info.structure_type == "flat":
            self._reorganize_flat_structure(prepared_path)
        else:
            self._reorganize_mixed_structure(prepared_path)

        return prepared_path

    def _reorganize_nested_structure(self, prepared_path: Path):
        """Reorganize nested structure to YOLO format."""
        # Map detected splits to YOLO splits
        split_mapping = {
            "train": "train",
            "val": "valid",
            "valid": "valid",
            "test": "test",
        }

        for detected_split, yolo_split in split_mapping.items():
            source_path = self.dataset_path / detected_split
            if source_path.exists():
                # Copy images
                images_source = (
                    source_path / "images"
                    if (source_path / "images").exists()
                    else source_path
                )
                if images_source.exists():
                    for img_file in images_source.glob("*.jpg"):
                        shutil.copy2(
                            img_file,
                            prepared_path / yolo_split / "images" / img_file.name,
                        )

                # Copy labels
                labels_source = (
                    source_path / "labels"
                    if (source_path / "labels").exists()
                    else source_path
                )
                if labels_source.exists():
                    for label_file in labels_source.glob("*.txt"):
                        shutil.copy2(
                            label_file,
                            prepared_path / yolo_split / "labels" / label_file.name,
                        )

    def _reorganize_flat_structure(self, prepared_path: Path):
        """Reorganize flat structure to YOLO format."""
        # For flat structure, we need to create splits
        all_images = list(self.dataset_path.glob("*.jpg"))
        all_labels = list(self.dataset_path.glob("*.txt"))

        # Simple split: 80% train, 20% valid
        np.random.shuffle(all_images)
        split_idx = int(len(all_images) * 0.8)

        train_images = all_images[:split_idx]
        valid_images = all_images[split_idx:]

        # Copy training images
        for img_file in train_images:
            shutil.copy2(img_file, prepared_path / "train" / "images" / img_file.name)

            # Copy corresponding label if exists
            label_file = img_file.with_suffix(".txt")
            if label_file.exists():
                shutil.copy2(
                    label_file, prepared_path / "train" / "labels" / label_file.name
                )

        # Copy validation images
        for img_file in valid_images:
            shutil.copy2(img_file, prepared_path / "valid" / "images" / img_file.name)

            # Copy corresponding label if exists
            label_file = img_file.with_suffix(".txt")
            if label_file.exists():
                shutil.copy2(
                    label_file, prepared_path / "valid" / "labels" / label_file.name
                )

    def _reorganize_mixed_structure(self, prepared_path: Path):
        """Reorganize mixed structure to YOLO format."""
        # This is a fallback that tries to make the best of what we have
        logger.warning("Mixed structure detected - using fallback reorganization")
        self._reorganize_flat_structure(prepared_path)

    def _generate_data_yaml(self, target_format: str):
        """Generate data.yaml file for the prepared dataset."""
        logger.info(f"Generating data.yaml for {target_format} format...")

        # Determine paths based on target format
        if target_format in ["yolo", "yolov8", "yolov5"]:
            # Standard YOLO format
            yaml_content = {
                "path": str(self.prepared_path.absolute()),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": self.dataset_info.class_count,
                "names": self.dataset_info.class_names,
            }
        else:
            # Fallback to standard format
            yaml_content = {
                "path": str(self.prepared_path.absolute()),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": self.dataset_info.class_count,
                "names": self.dataset_info.class_names,
            }

        # Write data.yaml
        yaml_path = self.prepared_path / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated data.yaml at {yaml_path}")

    def _validate_final_structure(self) -> bool:
        """Validate the final prepared dataset structure."""
        logger.info("Validating final dataset structure...")

        required_dirs = [
            "train/images",
            "train/labels",
            "valid/images",
            "valid/labels",
            "test/images",
            "test/labels",
        ]

        for dir_path in required_dirs:
            full_path = self.prepared_path / dir_path
            if not full_path.exists():
                logger.error(f"Missing required directory: {dir_path}")
                return False

            # Check if directory has files
            if not list(full_path.glob("*")):
                logger.warning(f"Directory {dir_path} is empty")

        # Check for data.yaml
        if not (self.prepared_path / "data.yaml").exists():
            logger.error("data.yaml not found")
            return False

        logger.info("Dataset structure validation passed")
        return True

    def get_preparation_summary(self) -> Dict[str, Any]:
        """Get summary of dataset preparation."""
        if not self.prepared_path:
            return {"status": "not_prepared"}

        return {
            "status": "prepared",
            "original_structure": self.dataset_info.structure_type,
            "prepared_path": str(self.prepared_path),
            "classes": self.dataset_info.class_names,
            "total_images": self.dataset_info.total_images,
            "splits": self.dataset_info.splits,
            "issues": self.dataset_info.issues,
        }


def auto_prepare_dataset(
    dataset_path: Union[str, Path], target_format: str = "yolo"
) -> Path:
    """
    Convenience function to automatically prepare a dataset.

    Args:
        dataset_path: Path to dataset directory
        target_format: Target YOLO format

    Returns:
        Path to prepared dataset
    """
    preparer = AutoDatasetPreparer(Path(dataset_path))
    return preparer.prepare_dataset(target_format)
