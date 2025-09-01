#!/usr/bin/env python3
"""
Script to download and manage pretrained YOLO weights.
This script downloads pretrained weights to the pretrained_weights/ folder
and manages their organization.
"""

import os
import sys
from pathlib import Path
import requests
import shutil
from typing import Dict, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# YOLO model weights URLs (from Ultralytics)
YOLO_WEIGHTS = {
    "yolo11": {
        "n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        "m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
        "x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
    },
    "yolov8": {
        "n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
        "m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
        "l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt",
        "x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt",
    },
    "yolov5": {
        "n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5nu.pt",
        "s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5su.pt",
        "m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5mu.pt",
        "l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5lu.pt",
        "x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5xu.pt",
    },
}


def download_file(url: str, filepath: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL to the specified filepath.

    Args:
        url: URL to download from
        filepath: Path to save the file
        chunk_size: Size of chunks to download

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(
                            f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)",
                            end="",
                        )

        print(f"\n✅ Downloaded: {filepath}")
        return True

    except Exception as e:
        print(f"\n❌ Failed to download {url}: {e}")
        return False


def download_weights(model_type: str, size: str, weights_dir: Path) -> bool:
    """
    Download specific YOLO weights.

    Args:
        model_type: Type of YOLO (yolo11, yolov8, yolov5)
        size: Model size (n, s, m, l, x)
        weights_dir: Directory to save weights

    Returns:
        True if successful, False otherwise
    """
    if model_type not in YOLO_WEIGHTS:
        print(f"❌ Unknown model type: {model_type}")
        return False

    if size not in YOLO_WEIGHTS[model_type]:
        print(f"❌ Unknown size: {size}")
        return False

    url = YOLO_WEIGHTS[model_type][size]
    # Extract the actual filename from the URL to preserve the full name
    filename = url.split("/")[-1]
    filepath = weights_dir / filename

    # Check if file already exists
    if filepath.exists():
        print(f"⚠️  Weights already exist: {filepath}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            print("Skipping download.")
            return True

    # Create directory if it doesn't exist
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Download the weights
    return download_file(url, filepath)


def list_available_weights() -> None:
    """List all available YOLO weights."""
    print("Available YOLO weights:")
    print("=" * 60)

    for model_type, sizes in YOLO_WEIGHTS.items():
        print(f"\n{model_type.upper()}:")
        for size, url in sizes.items():
            print(f"  {size}: {url.split('/')[-1]}")


def list_downloaded_weights(weights_dir: Path) -> None:
    """List all downloaded weights."""
    if not weights_dir.exists():
        print("No weights directory found.")
        return

    weights = list(weights_dir.glob("*.pt"))
    if not weights:
        print("No weights downloaded yet.")
        return

    print(f"Downloaded weights in {weights_dir}:")
    print("=" * 60)

    for weight in weights:
        size_mb = weight.stat().st_size / (1024 * 1024)
        print(f"  {weight.name} ({size_mb:.1f} MB)")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and manage pretrained YOLO weights")
    parser.add_argument("--model", choices=["yolo11", "yolov8", "yolov5"], 
                       help="Model type to download")
    parser.add_argument("--size", choices=["n", "s", "m", "l", "x"], 
                       help="Model size to download")
    parser.add_argument("--all", action="store_true", 
                       help="Download all available weights")
    parser.add_argument("--list", action="store_true", 
                       help="List available weights")
    parser.add_argument("--downloaded", action="store_true", 
                       help="List downloaded weights")
    
    args = parser.parse_args()
    
    print("🚀 YOLO Pretrained Weights Downloader")
    print("=" * 60)

    # Create weights directory
    weights_dir = Path(__file__).parent.parent / "pretrained_weights"
    weights_dir.mkdir(exist_ok=True)
    print(f"Weights directory: {weights_dir}")

    # Handle command line arguments
    if args.list:
        list_available_weights()
        return
    
    if args.downloaded:
        list_downloaded_weights(weights_dir)
        return
    
    if args.all:
        print("Downloading all available weights...")
        success_count = 0
        total_count = 0

        for model_type, sizes in YOLO_WEIGHTS.items():
            for size in sizes:
                total_count += 1
                if download_weights(model_type, size, weights_dir):
                    success_count += 1

        print(f"\n📊 Download Summary:")
        print(f"Successfully downloaded: {success_count}/{total_count}")
        return
    
    if args.model and args.size:
        if download_weights(args.model, args.size, weights_dir):
            print(f"✅ Successfully downloaded {args.model}{args.size}.pt")
        else:
            print(f"❌ Failed to download {args.model}{args.size}.pt")
        return

    # Check for legacy command format (for backwards compatibility)
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "list":
            list_available_weights()
            return
        elif command == "downloaded":
            list_downloaded_weights(weights_dir)
            return
        elif command == "download" and len(sys.argv) >= 4:
            model_type = sys.argv[2].lower()
            size = sys.argv[3].lower()
            if download_weights(model_type, size, weights_dir):
                print(f"✅ Successfully downloaded {model_type}{size}.pt")
            else:
                print(f"❌ Failed to download {model_type}{size}.pt")
            return
        elif command == "all":
            print("Downloading all available weights...")
            success_count = 0
            total_count = 0

            for model_type, sizes in YOLO_WEIGHTS.items():
                for size in sizes:
                    total_count += 1
                    if download_weights(model_type, size, weights_dir):
                        success_count += 1

            print(f"\n📊 Download Summary:")
            print(f"Successfully downloaded: {success_count}/{total_count}")
            return


if __name__ == "__main__":
    main()
