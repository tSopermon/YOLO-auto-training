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
from utils.gpu_memory_manager import GPUMemoryManager, clear_gpu_memory, optimize_gpu_for_training

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
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Model Training")

    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
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
        # Load configuration first if provided
        config = None
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = YOLOConfig.load(args.config)
        
        # Determine model type (from args, config, or interactive)
        model_type = args.model_type
        if model_type is None and config:
            # Get model type from config file
            model_type = config.model_type
            logger.info(f"Using model type from config: {model_type}")
        
        if model_type is None:
            if args.non_interactive:
                # Use default when non-interactive and no model type specified
                model_type = "yolov8"
                logger.info("Using default model type: yolov8 (non-interactive mode)")
            else:
                print(f"\nInteractive YOLO Training Setup")
                model_type = get_interactive_yolo_version()
                logger.info(f"Selected model type: {model_type}")
        
        # Load or create configuration
        if not config:
            logger.info(f"Creating configuration for: {model_type}")
            config = get_config(model_type)

        # Update model type in args for consistency
        args.model_type = model_type

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

        # Get interactive configuration if not skipped and no config file provided
        if not args.non_interactive and not args.config and not args.resume:
            # Check if essential training parameters are provided via command line
            # If epochs and batch_size are provided, skip interactive mode
            if args.epochs is not None and args.batch_size is not None:
                logger.info("Key training parameters provided via command line (skipping interactive configuration)")
                # Use default values for missing parameters
                if args.image_size is None:
                    args.image_size = 640  # Set default image size
                if args.device is None:
                    args.device = "auto"  # Set default device
                # Update config again with any new defaults
                config = update_config_from_args(config, args)
            else:
                print(f"\nInteractive Configuration for {args.model_type.upper()}")
                interactive_config = get_interactive_config(args.model_type)
                config = update_config_from_interactive(config, interactive_config)
        else:
            if args.non_interactive:
                logger.info("Skipping interactive configuration (non-interactive mode)")
            elif args.config:
                logger.info("Skipping interactive configuration (using config file)")
            elif args.resume:
                logger.info("Skipping interactive configuration (resuming from checkpoint)")

        # Get custom results folder name
        if args.results_folder:
            results_folder = args.results_folder
            logger.info(f"Using custom results folder: {results_folder}")
        elif args.resume:
            # Extract folder from resume path (e.g., logs/my_experiment/my_experiment/weights/last.pt -> my_experiment)
            resume_path = Path(args.resume)
            if "logs" in resume_path.parts:
                # Find the experiment folder name from the path
                logs_index = resume_path.parts.index("logs")
                if len(resume_path.parts) > logs_index + 1:
                    results_folder = resume_path.parts[logs_index + 1]
                    logger.info(f"Resuming training in existing folder: {results_folder}")
                else:
                    results_folder = "resumed_training"
                    logger.warning(f"Could not extract folder from resume path, using: {results_folder}")
            else:
                results_folder = "resumed_training"
                logger.warning(f"Could not extract folder from resume path, using: {results_folder}")
        elif args.non_interactive:
            # Generate default folder name in non-interactive mode
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_folder = f"{args.model_type}_{timestamp}"
            logger.info(f"Auto-generated results folder: {results_folder}")
        elif args.config:
            # Generate default folder name when using config file
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_folder = f"{model_type}_config_{timestamp}"
            logger.info(f"Auto-generated results folder for config: {results_folder}")
        else:
            # Check if we have enough CLI args to skip interactive folder selection
            if (args.model_type and args.epochs and args.batch_size and 
                (args.image_size and args.device)):
                # Generate a descriptive folder name based on parameters
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                results_folder = f"{args.model_type}_e{args.epochs}_b{args.batch_size}_{timestamp}"
                logger.info(f"Auto-generated results folder: {results_folder}")
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

        # Initialize GPU memory manager
        gpu_manager = GPUMemoryManager()
        if gpu_manager.cuda_available:
            logger.info("GPU Memory Manager initialized successfully")
            
            # Check if training configuration will fit in memory
            # Extract model size from weights filename (e.g., yolov8l.pt -> l)
            model_size = "m"  # Default fallback
            if config.weights:
                import re
                size_match = re.search(r'yolo(?:v5|v8|11)?([nslmx])', config.weights.lower())
                if size_match:
                    model_size = size_match.group(1)
            
            logger.info(f"Checking memory requirements for {config.model_type}{model_size} with {config.image_size}px images and batch size {config.batch_size}")
            
            memory_check = gpu_manager.estimate_training_memory_usage(
                model_size=model_size,
                image_size=config.image_size,
                batch_size=config.batch_size,
                model_version=config.model_type
            )
            
            if memory_check["gpu_status"]["will_fit"]:
                logger.info(f"✅ Configuration will fit in GPU memory ({memory_check['gpu_status']['utilization_pct']:.1f}% utilization)")
                logger.info(f"Estimated memory usage: {memory_check['estimated_usage']['total_estimated_gb']:.2f} GB")
            else:
                logger.warning(f"⚠️  Configuration may exceed GPU memory ({memory_check['gpu_status']['utilization_pct']:.1f}% over capacity)")
                logger.warning(f"Estimated memory needed: {memory_check['estimated_usage']['total_estimated_gb']:.2f} GB")
                logger.warning(f"Available memory: {memory_check['gpu_status']['available_memory_gb']:.2f} GB")
                logger.info(f"💡 Recommended batch size: {memory_check['recommendations']['recommended_batch_size']}")
                
                if memory_check['recommendations']['alternative_configs']:
                    logger.info("🔧 Alternative configurations:")
                    for alt in memory_check['recommendations']['alternative_configs'][:2]:
                        logger.info(f"   • {alt['change']} (Est: {alt['estimated_memory_gb']:.2f} GB)")
            
            # Optimize GPU for training
            optimization_result = optimize_gpu_for_training(config.batch_size)
            if "error" not in optimization_result:
                logger.info("GPU optimizations applied for training")
                recs = optimization_result["recommendations"]
                logger.info(f"Available GPU memory: {recs['available_memory_gb']:.2f} GB")
            else:
                logger.warning("Could not optimize GPU settings")
        else:
            logger.info("GPU not available - skipping GPU optimizations")

        if args.validate_only:
            # Validation only mode
            logger.info("Running validation only...")
            model = load_yolo_model(config, checkpoint_manager, args.resume)
            
            try:
                validate_model(model, config, monitor)
            finally:
                # Clear GPU memory after validation
                if gpu_manager.cuda_available:
                    logger.info("Clearing GPU memory after validation...")
                    cleanup_result = clear_gpu_memory(aggressive=True)
                    if "error" not in cleanup_result:
                        freed_gb = cleanup_result.get("freed_gb", 0)
                        logger.info(f"GPU memory cleanup completed. Freed: {freed_gb:.2f} GB")
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
                }
                
                # Add resume parameter if provided
                if args.resume:
                    # First, check if the checkpoint has completed training
                    try:
                        import torch
                        # Use weights_only=False for YOLO checkpoints (they contain model class definitions)
                        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
                        ckpt_epoch = ckpt.get('epoch', -1)  # -1 means not started/completed
                        ckpt_train_args = ckpt.get('train_args', {})
                        ckpt_target_epochs = ckpt_train_args.get('epochs', 0)
                        
                        # Calculate actual completed epochs (epoch is 0-indexed during training, -1 when completed)
                        if ckpt_epoch == -1:
                            # Training was completed (epoch resets to -1 after completion)
                            completed_epochs = ckpt_target_epochs
                        else:
                            # Training was interrupted (epoch shows last completed epoch)
                            completed_epochs = ckpt_epoch + 1
                        
                        logger.info(f"Checkpoint inspection: epoch={ckpt_epoch}, target_epochs={ckpt_target_epochs}, completed={completed_epochs}")
                        
                        # If user specified epochs, check if they want to extend training
                        if args.epochs and args.epochs > 0:
                            if args.epochs > completed_epochs:
                                logger.info(f"Extending training from checkpoint (completed {completed_epochs} epochs) to {args.epochs} total epochs")
                                logger.info("Note: This will start fresh training using the checkpoint as starting weights")
                                # Don't use resume mode, use the checkpoint as initial weights instead
                            else:
                                logger.warning(f"Requested {args.epochs} epochs, but checkpoint already completed {completed_epochs} epochs")
                                logger.info("Using checkpoint as starting weights for fresh training")
                        else:
                            # No epochs specified - check if training can be resumed
                            if ckpt_epoch == -1:  # Training was completed
                                logger.error("═════════════════════════════════════════════════════════════════════════════════")
                                logger.error("CANNOT RESUME: CHECKPOINT TRAINING ALREADY COMPLETED")
                                logger.error("═════════════════════════════════════════════════════════════════════════════════")
                                logger.error(f"The checkpoint has already completed {completed_epochs} epochs of training.")
                                logger.error("To extend training beyond the original epochs, you must specify --epochs:")
                                logger.error("")
                                logger.error("Examples:")
                                logger.error(f"  python train.py --model-type {args.model_type} --epochs {completed_epochs + 50} --resume {args.resume}")
                                logger.error(f"  python train.py --model-type {args.model_type} --epochs 200 --resume {args.resume}")
                                logger.error("")
                                logger.error("Or use the checkpoint as initial weights for new training:")
                                logger.error(f"  python train.py --model-type {args.model_type} --epochs 100 {args.resume}")
                                logger.error("  (Remove --resume and use checkpoint path as model)")
                                logger.error("═════════════════════════════════════════════════════════════════════════════════")
                                return
                            else:
                                logger.info(f"Resuming training from checkpoint (completed {completed_epochs} epochs)")
                                cfg['resume'] = args.resume
                    except Exception as e:
                        logger.warning(f"Could not inspect checkpoint: {e}")
                        logger.info("Proceeding with resume attempt...")
                        cfg['resume'] = args.resume
                
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
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Training failed with error: {error_msg}")
                    
                    # Special handling for "nothing to resume" error
                    if "training to" in error_msg and "epochs is finished, nothing to resume" in error_msg:
                        logger.error("═" * 80)
                        logger.error("RESUME TRAINING ERROR")
                        logger.error("═" * 80)
                        logger.error("The checkpoint has already completed its original training epochs.")
                        logger.error("To extend training beyond the original epochs, use one of these options:")
                        logger.error("")
                        logger.error("1. Continue training with additional epochs (recommended):")
                        logger.error(f"   python train.py --model-type {args.model_type} --epochs {args.epochs or 'NEW_EPOCH_COUNT'} {args.resume}")
                        logger.error("   (Remove --resume and use the checkpoint path as the model)")
                        logger.error("")
                        logger.error("2. Or use the checkpoint as initial weights for new training:")
                        checkpoint_path = args.resume
                        logger.error(f"   python train.py --model-type {args.model_type} --epochs {args.epochs or 'NEW_EPOCH_COUNT'} --config custom_config.yaml")
                        logger.error(f"   (Copy {checkpoint_path} to pretrained_weights/ and reference it)")
                        logger.error("")
                        logger.error("The --resume flag is only for continuing interrupted training,")
                        logger.error("not for extending completed training sessions.")
                        logger.error("═" * 80)
                    
                    logger.error(f"Training failed with error: {error_msg}")
                    raise
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
                    
                    # Check if it's a GPU memory error
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        logger.error("GPU memory error detected. Performing emergency cleanup...")
                        if gpu_manager.cuda_available:
                            emergency_result = gpu_manager.emergency_cleanup()
                            if emergency_result.get("success", False):
                                logger.info("Emergency GPU cleanup completed successfully")
                            else:
                                logger.error("Emergency GPU cleanup failed")
                    raise
                finally:
                    # Keep TensorBoard running after training completes
                    monitor.close(keep_tensorboard=True)
                    
                    # Clear GPU memory after training
                    if gpu_manager.cuda_available:
                        logger.info("Performing final GPU memory cleanup...")
                        cleanup_result = clear_gpu_memory(aggressive=True)
                        if "error" not in cleanup_result:
                            freed_gb = cleanup_result.get("freed_gb", 0)
                            logger.info(f"GPU memory cleanup completed. Freed: {freed_gb:.2f} GB")
                            
                            # Show final memory stats
                            final_stats = gpu_manager.get_memory_stats()
                            if "error" not in final_stats:
                                gpu_usage = final_stats["gpu"]["reserved_pct"]
                                logger.info(f"Final GPU memory usage: {gpu_usage:.1f}%")
                        else:
                            logger.warning("Could not perform final GPU cleanup")

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
