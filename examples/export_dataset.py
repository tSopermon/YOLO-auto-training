#!/usr/bin/env python3
"""
YOLO Dataset Export Example from Roboflow

This script demonstrates how to export datasets from Roboflow
for different YOLO model versions.
"""

import os
import sys
from pathlib import Path
from typing import Optional

try:
    from roboflow import Roboflow
except ImportError:
    print("Error: roboflow package not installed.")
    print("Install it with: pip install roboflow")
    sys.exit(1)


def export_dataset_for_yolo(
    api_key: str,
    workspace: str,
    project_id: str,
    version: str,
    yolo_version: str = "yolov8",
) -> Optional[str]:
    """
    Export dataset from Roboflow for a specific YOLO version.

    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project_id: Project ID
        version: Dataset version number
        yolo_version: YOLO version format (yolov8, yolov5, yolo)

    Returns:
        Path to downloaded dataset or None if failed
    """
    try:
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)

        # Get project and version
        project = rf.workspace(workspace).project(project_id)
        dataset = project.version(version).download(yolo_version)

        print(f"✅ Dataset exported successfully!")
        print(f"📁 Location: {dataset.location}")
        print(f"🔢 Format: {yolo_version}")

        return dataset.location

    except Exception as e:
        print(f"❌ Error exporting dataset: {e}")
        return None


def verify_dataset_structure(dataset_path: str) -> bool:
    """
    Verify that the exported dataset has the correct structure.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        True if structure is correct, False otherwise
    """
    required_dirs = [
        "train/images",
        "train/labels",
        "valid/images",
        "valid/labels",
        "test/images",
        "test/labels",
    ]

    required_files = ["data.yaml"]

    dataset_path = Path(dataset_path)

    # Check required directories
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        if not full_path.exists():
            print(f"❌ Missing directory: {dir_path}")
            return False

    # Check required files
    for file_path in required_files:
        full_path = dataset_path / file_path
        if not full_path.exists():
            print(f"❌ Missing file: {file_path}")
            return False

    # Check data.yaml content
    yaml_path = dataset_path / "data.yaml"
    try:
        with open(yaml_path, "r") as f:
            content = f.read()
            if "path:" in content and "train:" in content and "val:" in content:
                print("✅ data.yaml structure looks correct")
            else:
                print("⚠️  data.yaml may have incorrect structure")
                return False
    except Exception as e:
        print(f"❌ Error reading data.yaml: {e}")
        return False

    print("✅ Dataset structure verified successfully!")
    return True


def main():
    """Main function to demonstrate dataset export."""

    # Configuration - Replace with your actual values
    config = {
        "api_key": os.getenv("ROBOFLOW_API_KEY"),
        "workspace": "your_workspace_name",
        "project_id": "your_project_id",
        "version": "your_version_number",
    }

    # Check if API key is available
    if not config["api_key"]:
        print("❌ ROBOFLOW_API_KEY environment variable not set")
        print("Please set it with: export ROBOFLOW_API_KEY='your_api_key'")
        return

    # Check if other config values are set
    if any(v.startswith("your_") for v in config.values()):
        print(
            "⚠️  Please update the configuration in the script with your actual values"
        )
        print("Current config:", config)
        return

    print("🚀 Starting YOLO dataset export...")
    print(f"📊 Project: {config['workspace']}/{config['project_id']}")
    print(f"🔢 Version: {config['version']}")
    print()

    # Export for different YOLO versions
    yolo_versions = ["yolo11", "yolov8", "yolov5", "yolo"]

    for yolo_version in yolo_versions:
        print(f"📦 Exporting for {yolo_version}...")

        dataset_path = export_dataset_for_yolo(
            api_key=config["api_key"],
            workspace=config["workspace"],
            project_id=config["project_id"],
            version=config["version"],
            yolo_version=yolo_version,
        )

        if dataset_path:
            print(f"🔍 Verifying dataset structure for {yolo_version}...")
            verify_dataset_structure(dataset_path)
        else:
            print(f"⚠️  Failed to export for {yolo_version}")

        print("-" * 50)

    print("🎉 Export process completed!")
    print("\n📚 Next steps:")
    print("1. Choose the best export format for your needs")
    print("2. Use the dataset with your preferred YOLO training method")
    print("3. Check the individual YOLO version guides for training instructions")


if __name__ == "__main__":
    main()
