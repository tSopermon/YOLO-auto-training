#!/usr/bin/env python3
"""
Main YOLO training script.
Supports YOLOv8, YOLOv5, and YOLO11 training with configuration management.
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config import get_config, setup_logging, YOLOConfig
from utils.data_loader import YOLODataset
from utils.model_loader import load_yolo_model
from utils.training_utils import train_model, validate_model
from utils.checkpoint_manager import CheckpointManager
from utils.training_monitor import TrainingMonitor
from utils.auto_dataset_preparer import auto_prepare_dataset
from utils.training_callbacks import register_training_callbacks

logger = logging.getLogger(__name__)


def get_custom_results_folder() -> str:
    """
    Prompt user for a custom results folder name.

    Returns:
        str: Custom folder name for training results
    """
    print("\n" + "=" * 60)
    print("YOLO Training Results Folder")
    print("=" * 60)
    print("Enter a custom name for your training results folder.")
    print("This will create a folder like: logs/your_custom_name/")
    print("Examples: experiment_1, car_parts_v1, test_run_2024")
    print("=" * 60)

    while True:
        folder_name = input("\nEnter results folder name: ").strip()

        if not folder_name:
            print("❌ Folder name cannot be empty. Please try again.")
            continue

        # Clean the folder name (remove invalid characters)
        import re

        clean_name = re.sub(r'[<>:"/\\|?*]', "_", folder_name)
        clean_name = clean_name.replace(" ", "_")

        if clean_name != folder_name:
            print(f"⚠️  Folder name cleaned to: {clean_name}")

        # Check if folder already exists
        full_path = Path("logs") / clean_name
        if full_path.exists():
            response = (
                input(f"⚠️  Folder '{clean_name}' already exists. Overwrite? (y/N): ")
                .strip()
                .lower()
            )
            if response in ["y", "yes"]:
                print(f"✅ Using existing folder: {clean_name}")
                return clean_name
            else:
                print("Please choose a different name.")
                continue
        else:
            print(f"✅ Results will be saved to: logs/{clean_name}/")
            return clean_name


def get_interactive_yolo_version() -> str:
    """
    Get interactive YOLO version selection from user input.

    Returns:
        str: Selected YOLO version (yolo11, yolov8, yolov5)
    """
    print("\n" + "=" * 60)
    print("YOLO Version Selection")
    print("=" * 60)
    print("Choose which YOLO version you want to train:")
    print("=" * 60)

    versions = [
        ("yolo11", "YOLO11 - Latest version with best performance"),
        ("yolov8", "YOLOv8 - Stable, well-tested, recommended"),
        ("yolov5", "YOLOv5 - Classic version, very stable"),
    ]

    for i, (version, description) in enumerate(versions, 1):
        print(f"{i}. {version.upper()}: {description}")

    print("=" * 60)

    while True:
        choice = input("Select YOLO version (1-3, default: 2): ").strip()

        if not choice:
            print("Using default: YOLOv8")
            return "yolov8"

        try:
            choice_num = int(choice)
            if 1 <= choice_num <= 3:
                selected_version = versions[choice_num - 1][0]
                print(f"Selected: {selected_version.upper()}")
                return selected_version
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number (1-3).")


def get_interactive_config(model_type: str) -> dict:
    """
    Get interactive configuration from user input.

    Args:
        model_type: Type of YOLO model (yolo11, yolov8, yolov5)

    Returns:
        Dict containing user-specified configuration
    """
    print("\n" + "=" * 60)
    print("YOLO Training Configuration")
    print("=" * 60)
    print("Configure your training parameters. Press Enter to use defaults.")
    print("=" * 60)

    config = {}

    # Model size selection
    print(f"\nModel Size Selection ({model_type.upper()})")
    print("-" * 40)
    if model_type == "yolo11":
        sizes = ["n", "s", "m", "l", "x"]
        print("Available sizes: n (nano), s (small), m (medium), l (large), x (xlarge)")
        print("n = fastest, x = most accurate")
    elif model_type == "yolov8":
        sizes = ["n", "s", "m", "l", "x"]
        print("Available sizes: n (nano), s (small), m (medium), l (large), x (xlarge)")
        print("n = fastest, x = most accurate")
    else:  # yolov5
        sizes = ["n", "s", "m", "l", "x"]
        print("Available sizes: n (nano), s (small), m (medium), l (large), x (xlarge)")
        print("n = fastest, x = most accurate")

    while True:
        size_input = input(f"Model size (default: n): ").strip().lower()
        if not size_input:
            config["model_size"] = "n"
            break
        elif size_input in sizes:
            config["model_size"] = size_input
            break
        else:
            print(f"Invalid size. Choose from: {', '.join(sizes)}")

    # Training epochs
    print(f"\nTraining Duration")
    print("-" * 40)
    epochs_input = input("Number of epochs (default: 100): ").strip()
    if epochs_input:
        try:
            config["epochs"] = int(epochs_input)
        except ValueError:
            print("Invalid number. Using default: 100")
            config["epochs"] = 100
    else:
        config["epochs"] = 100

    # Batch size
    print(f"\nBatch Size")
    print("-" * 40)
    print("Larger batch size = faster training but more memory")
    print("Recommended: 8-32 (depending on GPU memory)")
    batch_input = input("Batch size (default: 8): ").strip()
    if batch_input:
        try:
            config["batch_size"] = int(batch_input)
        except ValueError:
            print("Invalid number. Using default: 8")
            config["batch_size"] = 8
    else:
        config["batch_size"] = 8

    # Image size
    print(f"\nImage Size")
    print("-" * 40)
    print("Larger images = better accuracy but slower training")
    print("Recommended: 640, 1024, or 1280")
    size_input = input("Image size (default: 1024): ").strip()
    if size_input:
        try:
            config["image_size"] = int(size_input)
        except ValueError:
            print("Invalid number. Using default: 1024")
            config["image_size"] = 1024
    else:
        config["image_size"] = 1024

    # Learning rate
    print(f"\nLearning Rate")
    print("-" * 40)
    print("Higher LR = faster training but may be unstable")
    print("Lower LR = more stable but slower training")
    lr_input = input("Learning rate (default: 0.01): ").strip()
    if lr_input:
        try:
            config["learning_rate"] = float(lr_input)
        except ValueError:
            print("Invalid number. Using default: 0.01")
            config["learning_rate"] = 0.01
    else:
        config["learning_rate"] = 0.01

    # Device selection
    print(f"\nTraining Device")
    print("-" * 40)
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print("cuda = GPU training (recommended)")
        print("cpu = CPU training (slower)")
        device_input = input("Device (default: cuda): ").strip().lower()
        if not device_input:
            config["device"] = "cuda"
        elif device_input in ["cuda", "cpu"]:
            config["device"] = device_input
        else:
            print("Invalid device. Using default: cuda")
            config["device"] = "cuda"
    else:
        print("No GPU available. Using CPU.")
        config["device"] = "cpu"

    # Advanced options
    print(f"\nAdvanced Options")
    print("-" * 40)
    print("Press Enter to use defaults for all advanced options")

    # Patience (early stopping)
    patience_input = input("Early stopping patience (default: 50): ").strip()
    if patience_input:
        try:
            config["patience"] = int(patience_input)
        except ValueError:
            config["patience"] = 50
    else:
        config["patience"] = 50

    # Data augmentation
    print(f"\nData Augmentation")
    print("-" * 40)
    mosaic_input = input(
        "Mosaic augmentation probability 0.0-1.0 (default: 1.0): "
    ).strip()
    if mosaic_input:
        try:
            config["mosaic"] = float(mosaic_input)
        except ValueError:
            config["mosaic"] = 1.0
    else:
        config["mosaic"] = 1.0

    # Validation frequency
    val_input = input("Validate every N epochs (default: 1): ").strip()
    if val_input:
        try:
            config["val_freq"] = int(val_input)
        except ValueError:
            config["val_freq"] = 1
    else:
        config["val_freq"] = 1

    print(f"\nConfiguration complete!")
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Model Training")

    parser.add_argument(
        "--model-type",
        type=str,
        default="yolov8",
        choices=["yolo11", "yolov8", "yolov5"],
        help="YOLO model type to train (will be prompted interactively if not specified)",
    )

    parser.add_argument("--config", type=str, help="Path to custom configuration file")

    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs (overrides config)"
    )

    parser.add_argument("--batch-size", type=int, help="Batch size (overrides config)")

    parser.add_argument(
        "--image-size", type=int, help="Input image size (overrides config)"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        help="Device to use for training (overrides config)",
    )

    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the model, don't train",
    )

    parser.add_argument(
        "--export", action="store_true", help="Export model after training"
    )

    parser.add_argument(
        "--results-folder",
        type=str,
        help="Custom name for results folder (skips interactive prompt)",
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip interactive configuration prompts (use defaults)",
    )

    return parser.parse_args()


def update_config_from_args(config: YOLOConfig, args: argparse.Namespace) -> YOLOConfig:
    """Update configuration with command line arguments."""
    if args.epochs is not None:
        config.epochs = args.epochs
        logger.info(f"Overriding epochs: {args.epochs}")

    if args.batch_size is not None:
        config.batch_size = args.batch_size
        logger.info(f"Overriding batch size: {args.batch_size}")

    if args.image_size is not None:
        config.image_size = args.image_size
        logger.info(f"Overriding image size: {args.image_size}")

    if args.device is not None:
        if args.device == "auto":
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            config.device = args.device
        logger.info(f"Overriding device: {config.device}")

    return config


def update_config_from_interactive(
    config: YOLOConfig, interactive_config: Dict[str, Any]
) -> YOLOConfig:
    """Update configuration with interactive user input."""
    # Update model weights based on size
    if "model_size" in interactive_config:
        size = interactive_config["model_size"]
        if config.model_type == "yolo11":
            config.weights = f"yolo11{size}.pt"
        elif config.model_type == "yolov8":
            config.weights = f"yolov8{size}.pt"
        elif config.model_type == "yolov5":
            config.weights = f"yolov5{size}.pt"
        logger.info(f"Model weights: {config.weights}")

    # Update other parameters
    if "epochs" in interactive_config:
        config.epochs = interactive_config["epochs"]
        logger.info(f"Epochs: {config.epochs}")

    if "batch_size" in interactive_config:
        config.batch_size = interactive_config["batch_size"]
        logger.info(f"Batch size: {config.batch_size}")

    if "image_size" in interactive_config:
        config.image_size = interactive_config["image_size"]
        logger.info(f"Image size: {config.image_size}")

    if "device" in interactive_config:
        config.device = interactive_config["device"]
        logger.info(f"Device: {config.device}")

    if "patience" in interactive_config:
        config.patience = interactive_config["patience"]
        logger.info(f"Patience: {config.patience}")

    if "learning_rate" in interactive_config:
        config.model_config["learning_rate"] = interactive_config["learning_rate"]
        logger.info(f"Learning rate: {interactive_config['learning_rate']}")

    if "mosaic" in interactive_config:
        config.augmentation_config["mosaic"] = interactive_config["mosaic"]
        logger.info(f"Mosaic probability: {interactive_config['mosaic']}")

    if "val_freq" in interactive_config:
        config.eval_config["val_freq"] = interactive_config["val_freq"]
        logger.info(f"Validation frequency: {interactive_config['val_freq']}")

    return config


def auto_prepare_dataset_if_needed(model_type: str) -> Path:
    """
    Automatically prepare dataset for training if needed.

    Args:
        model_type: YOLO model type (yolo11, yolov8, yolov5)

    Returns:
        Path to prepared dataset
    """
    # Check if dataset is already prepared
    dataset_yaml = Path("dataset/data.yaml")
    if dataset_yaml.exists():
        dataset_dir = dataset_yaml.parent
        if (dataset_dir / "train" / "images").exists() and (
            dataset_dir / "valid" / "images"
        ).exists():
            logger.info("Dataset already prepared for YOLO training")
            return dataset_dir

    # Check if we have a dataset directory to prepare
    dataset_root = Path("dataset")
    if not dataset_root.exists():
        raise FileNotFoundError(
            "No dataset directory found. Please create a 'dataset' folder with your data."
        )

    logger.info(
        "Dataset not prepared for YOLO training. Starting automatic preparation..."
    )

    try:
        # Auto-prepare the dataset
        prepared_path = auto_prepare_dataset(dataset_root, model_type)
        logger.info(f"Dataset prepared successfully at: {prepared_path}")
        return prepared_path
    except Exception as e:
        logger.error(f"Failed to prepare dataset automatically: {e}")
        logger.info("Please prepare your dataset manually or check the error above.")
        raise


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    try:
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = YOLOConfig.load(args.config)
        else:
            logger.info(f"Creating configuration for: {args.model_type}")
            config = get_config(args.model_type)

        # Add auto_prepare_dataset_if_needed function
        def auto_prepare_dataset_if_needed(model_type: str) -> Path:
            """
            Automatically prepare dataset for YOLO training if needed.
            """
            dataset_path = Path("dataset")
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
            
            # Always prepare the dataset to ensure correct configuration
            logger.info("Preparing dataset for YOLO training...")
            prepared_path = auto_prepare_dataset(dataset_path, model_type)
            logger.info("Dataset prepared successfully")
            
            return prepared_path

        # Update config with command line arguments
        config = update_config_from_args(config, args)

        # Get interactive configuration if not skipped
        if not args.non_interactive:
            # If no model type specified, get it interactively
            if args.model_type == "yolov8" and not args.config:
                print(f"\nInteractive YOLO Training Setup")
                interactive_model_type = get_interactive_yolo_version()
                # Update the config with the selected model type
                config = get_config(interactive_model_type)
                args.model_type = interactive_model_type
                logger.info(f"Selected model type: {interactive_model_type}")

            print(f"\nInteractive Configuration for {args.model_type.upper()}")
            interactive_config = get_interactive_config(args.model_type)
            config = update_config_from_interactive(config, interactive_config)
        else:
            logger.info("Skipping interactive configuration (using defaults)")

        # Get custom results folder name
        if args.results_folder:
            results_folder = args.results_folder
            logger.info(f"Using custom results folder: {results_folder}")
        else:
            results_folder = get_custom_results_folder()
            logger.info(f"Selected results folder: {results_folder}")

        # Update the logging configuration with custom results folder
        config.logging_config["log_dir"] = str(Path("logs") / results_folder)
        config.logging_config["project_name"] = results_folder

        # Set up logging
        setup_logging(config)

        # Auto-prepare dataset if needed
        logger.info("Checking dataset preparation...")
        prepared_dataset_path = auto_prepare_dataset_if_needed(args.model_type)

        # Update config with prepared dataset path
        config.dataset_config["data_yaml_path"] = str(
            prepared_dataset_path / "data.yaml"
        )

        # Now load the data.yaml configuration
        config.load_data_yaml()

        # Log configuration
        logger.info("=" * 60)
        logger.info("YOLO Training Configuration")
        logger.info("=" * 60)
        logger.info(f"Model Type: {config.model_type}")
        logger.info(f"Model Weights: {config.weights}")
        logger.info(f"Training Device: {config.device}")
        logger.info(f"Batch Size: {config.batch_size}")
        logger.info(f"Image Size: {config.image_size}")
        logger.info(f"Epochs: {config.epochs}")
        logger.info(f"Dataset: {config.dataset_config['data_yaml_path']}")
        logger.info(f"Results Folder: {results_folder}")
        logger.info(f"Full Log Path: {config.logging_config['log_dir']}")
        logger.info("=" * 60)

        # Save configuration
        config.save()

        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(
            save_dir=Path(config.logging_config["log_dir"]) / "checkpoints",
            max_checkpoints=config.logging_config["num_checkpoint_keep"],
        )

        # Initialize training monitor
        monitor = TrainingMonitor(
            config=config, log_dir=config.logging_config["log_dir"]
        )

        if args.validate_only:
            # Validation only mode
            logger.info("Running validation only...")
            model = load_yolo_model(config, checkpoint_manager, args.resume)
            validate_model(model, config, monitor)
        else:
            # Training mode
            logger.info("Starting training...")

            try:
                # Create Ultralytics YOLO instance for training
                from ultralytics import YOLO

                # Choose model weights - resume checkpoint takes priority
                model_weights = args.resume if args.resume else config.weights
                logger.info(f"Using model weights: {model_weights}")
                
                yolo_model = YOLO(model_weights)

                # Start training
                logger.info("Starting Ultralytics training...")
                
                # Setup training configuration
                cfg = {
                    'data': str(config.dataset_config["data_yaml_path"]),
                    'epochs': config.epochs,
                    'imgsz': config.image_size,
                    'batch': config.batch_size,
                    'device': config.device,
                    'workers': config.num_workers,
                    'patience': config.patience,
                    'project': config.logging_config["log_dir"],
                    'name': results_folder,
                    'exist_ok': True,
                    'lr0': config.model_config.get("learning_rate", 0.01),
                    'resume': bool(args.resume),  # Enable resume if checkpoint provided
                }
                
                try:
                    # Train model - Ultralytics automatically enables TensorBoard
                    results = yolo_model.train(**cfg)
                    
                    # Log results
                    logger.info(f"Training completed. Results saved to {cfg['project']}/{cfg['name']}")
                    logger.info("You can view training metrics in TensorBoard")
                    
                    # Log best metrics from the results object
                    if hasattr(results, 'results_dict'):
                        logger.info("Best metrics achieved:")
                        best_map50 = results.results_dict.get('metrics/mAP50(B)', 'N/A')
                        best_map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 'N/A')
                        logger.info(f"Best mAP50: {best_map50}")
                        logger.info(f"Best mAP50-95: {best_map50_95}")
                    
                    logger.info("Training completed successfully!")
                    logger.info("=" * 60)
                    logger.info("Training Complete! Your TensorBoard is still running.")
                    logger.info("View your training results at: http://localhost:6006")
                    logger.info("You can analyze training metrics, loss curves, and model performance.")
                    logger.info("")
                    logger.info("TensorBoard Management Commands:")
                    logger.info("  python -m utils.tensorboard_manager status    # Check status & open")
                    logger.info("  python -m utils.tensorboard_manager stop      # Stop TensorBoard")
                    logger.info("  python -m utils.tensorboard_manager list      # List experiments")
                    logger.info(f"  python -m utils.tensorboard_manager launch {results_folder}  # Relaunch this experiment")
                    logger.info("=" * 60)
                except Exception as e:
                    logger.error(f"Training failed with error: {e}")
                    raise
                finally:
                    # Keep TensorBoard running after training completes
                    monitor.close(keep_tensorboard=True)

            except ImportError:
                logger.error(
                    "Ultralytics not available. Please install with: pip install ultralytics"
                )
                raise
            except Exception as e:
                logger.error(f"Training failed: {e}")
                raise

            # Export model if requested
            if args.export:
                logger.info("Exporting model...")
                from utils.export_utils import export_model

                export_model(model, config)

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
