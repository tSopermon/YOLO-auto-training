#!/usr/bin/env python3
"""
Convert COCO format annotations to YOLO format for training.
This script converts your COCO JSON annotations to YOLO .txt label files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def convert_bbox_to_yolo(
    bbox: List[float], img_width: int, img_height: int
) -> List[float]:
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height].

    Args:
        bbox: COCO bbox [x, y, width, height]
        img_width: Image width
        img_height: Image height

    Returns:
        YOLO bbox [x_center, y_center, width, height] (normalized)
    """
    x, y, w, h = bbox

    # Convert to center coordinates
    x_center = x + w / 2
    y_center = y + h / 2

    # Normalize to [0, 1]
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height

    return [x_center, y_center, w, h]


def convert_coco_to_yolo(
    coco_file: str, output_dir: str, class_mapping: Dict[str, int]
) -> None:
    """
    Convert COCO annotations to YOLO format.

    Args:
        coco_file: Path to COCO JSON file
        output_dir: Directory to save YOLO labels
        class_mapping: Mapping from class names to class IDs
    """
    # Load COCO annotations
    with open(coco_file, "r") as f:
        coco_data = json.load(f)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create image ID to filename mapping
    image_info = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Process each image
    for img_id, annotations in annotations_by_image.items():
        img_info = image_info[img_id]
        filename = img_info["file_name"]
        img_width = img_info["width"]
        img_height = img_info["height"]

        # Create label filename (replace .jpg with .txt)
        label_filename = filename.replace(".jpg", ".txt")
        label_path = output_path / label_filename

        # Convert annotations to YOLO format
        yolo_labels = []
        for ann in annotations:
            category_id = ann["category_id"]
            category_name = next(
                cat["name"]
                for cat in coco_data["categories"]
                if cat["id"] == category_id
            )

            if category_name in class_mapping:
                class_id = class_mapping[category_name]
                bbox = ann["bbox"]
                yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)

                # Format: class_id x_center y_center width height
                yolo_labels.append(
                    f"{class_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}"
                )

        # Write YOLO labels
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_labels))

        print(f"Converted {filename}: {len(yolo_labels)} labels")


def main():
    parser = argparse.ArgumentParser(description="Convert COCO to YOLO format")
    parser.add_argument("--coco-file", required=True, help="Path to COCO JSON file")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for YOLO labels"
    )
    parser.add_argument(
        "--class-mapping", required=True, help="JSON file with class name to ID mapping"
    )

    args = parser.parse_args()

    # Load class mapping
    with open(args.class_mapping, "r") as f:
        class_mapping = json.load(f)

    print(f"Converting {args.coco_file} to YOLO format...")
    print(f"Class mapping: {class_mapping}")

    convert_coco_to_yolo(args.coco_file, args.output_dir, class_mapping)
    print("Conversion completed!")


if __name__ == "__main__":
    main()
