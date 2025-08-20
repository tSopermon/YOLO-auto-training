#!/usr/bin/env python3
"""
Script to export existing trained YOLO models to various deployment formats.
This script will convert your trained models (yolo11n.pt, yolov8n.pt) to ONNX, TorchScript, etc.
"""

import os
import sys
import glob
from pathlib import Path
import torch
from ultralytics import YOLO
import shutil
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def export_model_to_formats(model_path: Path, export_dir: Path):
    """
    Export a trained YOLO model to multiple formats.

    Args:
        model_path: Path to the trained model (.pt file)
        export_dir: Directory to save exported models
    """
    print(f"\n{'='*60}")
    print(f"Exporting model: {model_path.name}")
    print(f"{'='*60}")

    try:
        # Load the model
        model = YOLO(str(model_path))

        # Create model-specific export directory
        model_name = model_path.stem  # Remove .pt extension
        model_export_dir = export_dir / model_name
        model_export_dir.mkdir(parents=True, exist_ok=True)

        print(f"Model loaded successfully")
        print(f"Export directory: {model_export_dir}")

        # Export to ONNX
        print("\n1. Exporting to ONNX format...")
        onnx_path = model_export_dir / f"{model_name}.onnx"
        model.export(format="onnx", simplify=True, half=False)  # Disable half for CPU
        # Find and move the exported ONNX file
        onnx_files = glob.glob(f"{model_name}.onnx")
        if onnx_files:
            shutil.move(onnx_files[0], onnx_path)
            print(f"   ✅ ONNX exported: {onnx_path}")
        else:
            print(f"   ❌ ONNX export failed - file not found")

        # Export to TorchScript
        print("\n2. Exporting to TorchScript format...")
        torchscript_path = model_export_dir / f"{model_name}_torchscript.pt"
        model.export(format="torchscript", half=False)  # Disable half for CPU
        # Find and move the exported TorchScript file
        torchscript_files = glob.glob(f"{model_name}.torchscript")
        if torchscript_files:
            shutil.move(torchscript_files[0], torchscript_path)
            print(f"   ✅ TorchScript exported: {torchscript_path}")
        else:
            print(f"   ❌ TorchScript export failed - file not found")

        # Export to OpenVINO
        print("\n3. Exporting to OpenVINO format...")
        openvino_path = model_export_dir / f"{model_name}_openvino"
        model.export(format="openvino", half=False)  # Disable half for CPU
        # Find and move the exported OpenVINO directory
        openvino_dirs = glob.glob(f"{model_name}_openvino_model")
        if openvino_dirs:
            if openvino_path.exists():
                shutil.rmtree(openvino_path)
            shutil.move(openvino_dirs[0], openvino_path)
            print(f"   ✅ OpenVINO exported: {openvino_path}")
        else:
            print(f"   ❌ OpenVINO export failed - directory not found")

        # Export to CoreML
        print("\n4. Exporting to CoreML format...")
        coreml_path = model_export_dir / f"{model_name}.mlpackage"
        model.export(format="coreml", half=False)  # Disable half for CPU
        # Find and move the exported CoreML file (note: .mlpackage extension)
        coreml_files = glob.glob(f"{model_name}.mlpackage")
        if coreml_files:
            shutil.move(coreml_files[0], coreml_path)
            print(f"   ✅ CoreML exported: {coreml_path}")
        else:
            print(f"   ❌ CoreML export failed - file not found")

        # Export to TensorRT (if CUDA available)
        if torch.cuda.is_available():
            print("\n5. Exporting to TensorRT format...")
            tensorrt_path = model_export_dir / f"{model_name}.engine"
            model.export(format="engine", half=True)
            # Find and move the exported TensorRT file
            engine_files = glob.glob(f"{model_name}.engine")
            if engine_files:
                shutil.move(engine_files[0], tensorrt_path)
                print(f"   ✅ TensorRT exported: {tensorrt_path}")
            else:
                print(f"   ❌ TensorRT export failed - file not found")
        else:
            print("\n5. Skipping TensorRT export (CUDA not available)")

        # Create export metadata
        metadata = {
            "model_name": model_name,
            "original_path": str(model_path),
            "export_timestamp": str(datetime.now()),
            "exported_formats": ["onnx", "torchscript", "openvino", "coreml"],
            "export_directory": str(model_export_dir),
            "model_info": {
                "type": str(type(model)),
                "parameters": (
                    sum(p.numel() for p in model.parameters())
                    if hasattr(model, "parameters")
                    else "Unknown"
                ),
            },
        }

        if torch.cuda.is_available():
            metadata["exported_formats"].append("tensorrt")

        # Save metadata
        import json

        metadata_path = model_export_dir / "export_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\n   ✅ Export metadata saved: {metadata_path}")
        print(f"\n🎉 Model {model_name} exported successfully!")
        print(f"📁 All exports saved to: {model_export_dir}")

        return True

    except Exception as e:
        print(f"❌ Failed to export {model_path.name}: {e}")
        return False


def main():
    """Main export function."""
    print("🚀 YOLO Model Export Script")
    print("=" * 60)
    print("This script will export your trained models to multiple deployment formats.")
    print("=" * 60)

    # Check if we're in the right directory
    root_dir = Path.cwd()
    export_dir = root_dir / "exported_models"

    # Find trained models
    trained_models = list(root_dir.glob("*.pt"))

    if not trained_models:
        print("❌ No trained models (.pt files) found in current directory")
        print(
            "   Make sure you're running this script from the directory containing your trained models"
        )
        return

    print(f"\n📋 Found {len(trained_models)} trained model(s):")
    for model in trained_models:
        print(f"   • {model.name}")

    # Create export directory
    export_dir.mkdir(exist_ok=True)
    print(f"\n📁 Export directory: {export_dir}")

    # Export each model
    success_count = 0
    for model_path in trained_models:
        if export_model_to_formats(model_path, export_dir):
            success_count += 1

    # Summary
    print(f"\n{'='*60}")
    print("📊 EXPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Total models found: {len(trained_models)}")
    print(f"Successfully exported: {success_count}")
    print(f"Failed: {len(trained_models) - success_count}")

    if success_count > 0:
        print(f"\n🎯 Your models are now ready for deployment!")
        print(f"📁 Check the 'exported_models/' folder for all exported formats")
        print(f"\n💡 Usage examples:")
        print(f"   • ONNX: Use with ONNX Runtime for cross-platform deployment")
        print(f"   • TorchScript: Use with PyTorch for production deployment")
        print(f"   • OpenVINO: Use with Intel OpenVINO for CPU optimization")
        print(f"   • CoreML: Use with Apple devices (iOS/macOS)")
        if torch.cuda.is_available():
            print(f"   • TensorRT: Use with NVIDIA GPUs for maximum performance")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
