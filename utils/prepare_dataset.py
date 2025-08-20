#!/usr/bin/env python3
"""
Standalone Dataset Preparation Script.
Automatically prepares datasets for YOLO training.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.auto_dataset_preparer import auto_prepare_dataset
from config.config import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main dataset preparation function."""
    parser = argparse.ArgumentParser(
        description="Automatically prepare datasets for YOLO training"
    )
    parser.add_argument(
        "dataset_path", type=str, help="Path to dataset directory to prepare"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="yolo",
        choices=["yolo", "yolov8", "yolov5", "yolo11"],
        help="Target YOLO format (default: yolo)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for prepared dataset (default: dataset_prepared)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Validate input path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    if not dataset_path.is_dir():
        logger.error(f"Dataset path is not a directory: {dataset_path}")
        sys.exit(1)

    try:
        logger.info(f"Starting dataset preparation for {args.format} format...")
        logger.info(f"Input dataset: {dataset_path}")

        # Prepare the dataset
        prepared_path = auto_prepare_dataset(dataset_path, args.format)

        logger.info("=" * 60)
        logger.info("Dataset Preparation Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"Prepared dataset location: {prepared_path}")
        logger.info(f"Target format: {args.format}")
        logger.info(f"data.yaml created at: {prepared_path / 'data.yaml'}")
        logger.info("=" * 60)
        logger.info("You can now use this dataset for YOLO training!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
