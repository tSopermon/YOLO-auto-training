#!/usr/bin/env python3
"""
Test script for the automated dataset preparation system.
"""

import sys
import pytest
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.auto_dataset_preparer import AutoDatasetPreparer


def test_dataset_analysis():
    """Test dataset structure analysis."""
    print("Testing dataset structure analysis...")

    # Test with current dataset
    dataset_path = Path("dataset")

    if not dataset_path.exists():
        pytest.skip(
            "Dataset directory not found. Please ensure you have a dataset/ folder."
        )

    preparer = AutoDatasetPreparer(dataset_path)
    dataset_info = preparer._analyze_dataset_structure()

    print(f"✅ Dataset analysis successful!")
    print(f"   Structure type: {dataset_info.structure_type}")
    print(f"   Has images: {dataset_info.has_images}")
    print(f"   Has labels: {dataset_info.has_labels}")
    print(f"   Image formats: {dataset_info.image_formats}")
    print(f"   Label formats: {dataset_info.label_formats}")
    print(
        f"   Classes: {dataset_info.class_count} ({', '.join(dataset_info.class_names)})"
    )
    print(f"   Total images: {dataset_info.total_images}")
    print(f"   Splits: {dataset_info.splits}")
    print(f"   Is YOLO ready: {dataset_info.is_yolo_ready}")

    if dataset_info.issues:
        print(f"   Issues: {dataset_info.issues}")

    # Use assertions instead of returning values
    assert dataset_info.structure_type in ["flat", "nested", "mixed"]
    assert dataset_info.has_images or dataset_info.has_labels


def test_dataset_preparation():
    """Test full dataset preparation."""
    print("\nTesting dataset preparation...")

    dataset_path = Path("dataset")

    if not dataset_path.exists():
        pytest.skip("Dataset directory not found. Skipping preparation test.")

    preparer = AutoDatasetPreparer(dataset_path)
    prepared_path = preparer.prepare_dataset("yolo")

    print(f"✅ Dataset preparation successful!")
    print(f"   Prepared dataset location: {prepared_path}")

    # Check if data.yaml was created
    yaml_path = prepared_path / "data.yaml"
    assert yaml_path.exists(), "data.yaml not found"
    print(f"   data.yaml created: {yaml_path}")

    # Check structure
    required_dirs = ["train/images", "train/labels", "valid/images", "valid/labels"]
    for dir_path in required_dirs:
        full_path = prepared_path / dir_path
        assert full_path.exists(), f"Directory {dir_path} not found"
        assert list(full_path.glob("*")), f"Directory {dir_path} is empty"
        print(f"   ✅ {dir_path}: OK")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Automated Dataset System Test")
    print("=" * 60)

    # Test 1: Dataset analysis
    analysis_success = test_dataset_analysis()

    # Test 2: Dataset preparation (only if analysis succeeded)
    preparation_success = False
    if analysis_success:
        preparation_success = test_dataset_preparation()

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"Dataset Analysis: {'✅ PASS' if analysis_success else '❌ FAIL'}")
    print(f"Dataset Preparation: {'✅ PASS' if preparation_success else '❌ FAIL'}")

    if analysis_success and preparation_success:
        print(
            "\n🎉 All tests passed! Your automated dataset system is working correctly."
        )
        print("You can now use:")
        print("  - python train.py (automatic preparation during training)")
        print(
            "  - python utils/prepare_dataset.py dataset/ --format yolov8 (standalone preparation)"
        )
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

    print("=" * 60)


if __name__ == "__main__":
    main()
