# Training Workflows

## What This System Does

The training workflows are like a well-orchestrated symphony where every component plays its part in perfect harmony. Think of it as a complete journey from raw dataset to trained model, where each step builds upon the previous one to create something amazing.

## Complete Training Workflow Overview

### **The Big Picture**

The training workflow follows this high-level sequence:

```
Raw Dataset → Auto Preparation → Model Selection → Configuration → Training → Results → Export
     ↓              ↓              ↓              ↓           ↓         ↓         ↓
  dataset/    prepared/      model_type    config.py    training   results/  exported/
```

### **Workflow Phases**

1. **Dataset Input Phase** - Raw data enters the system
2. **Preparation Phase** - Automatic dataset organization and conversion
3. **Configuration Phase** - Model and training parameters setup
4. **Training Phase** - Model training execution
5. **Results Phase** - Training outcomes and evaluation
6. **Export Phase** - Model conversion for deployment

## Phase 1: Dataset Input Phase

### **How Data Enters the System**

The system accepts datasets in multiple ways:

#### **Method 1: Direct Placement (Recommended)**
```bash
# Simply place your dataset in the dataset/ folder
# Any structure, any format - the system handles everything!

dataset/
├── images/          # Your images
├── labels/          # Your labels (any format)
├── annotations.json # COCO format
├── labels.xml       # XML format
└── ...              # Any other files
```

#### **Method 2: Roboflow Export**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="your_key")
project = rf.workspace("workspace").project("project_id")
dataset = project.version("version_number").download("yolov8")
```

#### **Method 3: Command Line Import**
```bash
# Import from external source
python utils/import_dataset.py --source /path/to/external/dataset

# Download from URL
python utils/import_dataset.py --url https://example.com/dataset.zip
```

### **Dataset Detection Logic**

The system automatically detects what you've provided:

```python
def auto_prepare_dataset_if_needed(model_type: str) -> Path:
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

    logger.info("Dataset not prepared for YOLO training. Starting automatic preparation...")
    
    try:
        # Auto-prepare the dataset
        prepared_path = auto_prepare_dataset(dataset_root, model_type)
        logger.info(f"Dataset prepared successfully at: {prepared_path}")
        return prepared_path
    except Exception as e:
        logger.error(f"Failed to prepare dataset automatically: {e}")
        raise
```

## Phase 2: Preparation Phase

### **Automatic Dataset Preparation**

The system automatically prepares your dataset through these steps:

#### **Step 1: Structure Analysis**
```python
def _analyze_dataset_structure(self) -> DatasetInfo:
    """Analyze the current dataset structure."""
    logger.info("Analyzing dataset structure...")
    
    # Detect structure type
    if self._is_flat_structure():
        structure_type = "flat"
    elif self._is_nested_structure():
        structure_type = "nested"
    elif self._is_mixed_structure():
        structure_type = "mixed"
    else:
        structure_type = "unknown"
    
    # Count classes and images
    class_names = self._detect_classes()
    total_images = self._count_images()
    
    return DatasetInfo(
        structure_type=structure_type,
        class_names=class_names,
        class_count=len(class_names),
        total_images=total_images,
        splits=self._detect_splits()
    )
```

#### **Step 2: Issue Detection and Fixing**
```python
def _detect_and_fix_issues(self):
    """Detect and fix common dataset issues."""
    logger.info("Detecting and fixing dataset issues...")
    
    # Check for missing labels
    missing_labels = self._find_missing_labels()
    if missing_labels:
        logger.warning(f"Found {len(missing_labels)} images without labels")
        self._fix_missing_labels(missing_labels)
    
    # Check for empty labels
    empty_labels = self._find_empty_labels()
    if empty_labels:
        logger.warning(f"Found {len(empty_labels)} empty label files")
        self._fix_empty_labels(empty_labels)
    
    # Check for format inconsistencies
    format_issues = self._detect_format_inconsistencies()
    if format_issues:
        logger.warning(f"Found {len(format_issues)} format inconsistencies")
        self._fix_format_inconsistencies(format_issues)
```

#### **Step 3: Smart Reorganization**
```python
def _reorganize_to_yolo_structure(self) -> Path:
    """Reorganize dataset to YOLO standard structure."""
    logger.info("Reorganizing dataset to YOLO structure...")
    
    # Create YOLO directory structure
    prepared_path = self.dataset_path / "prepared"
    for split in ["train", "valid", "test"]:
        for subdir in ["images", "labels"]:
            (prepared_path / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Move files based on detected structure
    if self.dataset_info.structure_type == "flat":
        self._reorganize_flat_structure(prepared_path)
    elif self.dataset_info.structure_type == "nested":
        self._reorganize_nested_structure(prepared_path)
    else:
        self._reorganize_mixed_structure(prepared_path)
    
    return prepared_path
```

#### **Step 4: Format Conversion**
```python
def _convert_annotations(self, source_path: Path, target_path: Path):
    """Convert annotations to YOLO format."""
    logger.info("Converting annotations to YOLO format...")
    
    # Detect annotation format
    if self._is_coco_format(source_path):
        self._convert_coco_to_yolo(source_path, target_path)
    elif self._is_xml_format(source_path):
        self._convert_xml_to_yolo(source_path, target_path)
    elif self._is_custom_format(source_path):
        self._convert_custom_to_yolo(source_path, target_path)
    else:
        logger.warning("Unknown annotation format, attempting generic conversion")
        self._convert_generic_to_yolo(source_path, target_path)
```

#### **Step 5: Configuration Generation**
```python
def _generate_data_yaml(self, target_format: str):
    """Generate data.yaml file for the prepared dataset."""
    logger.info(f"Generating data.yaml for {target_format} format...")
    
    # Create YOLO configuration
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
```

#### **Step 6: Final Validation**
```python
def _validate_final_structure(self) -> bool:
    """Validate the final prepared dataset structure."""
    logger.info("Validating final dataset structure...")
    
    required_dirs = [
        "train/images", "train/labels",
        "valid/images", "valid/labels",
        "test/images", "test/labels",
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
```

## Phase 3: Configuration Phase

### **Interactive Configuration Workflow**

The system guides users through configuration with multiple interaction levels:

#### **Full Interactive Mode (Default)**
```python
def get_interactive_config(model_type: str) -> Dict[str, Any]:
    """Get interactive configuration from user."""
    config = {}
    
    print(f"\n=== Interactive Configuration for {model_type.upper()} ===")
    
    # Model size selection
    print("\n1. Model Size Selection:")
    sizes = ["n", "s", "m", "l", "x"]
    for i, size in enumerate(sizes, 1):
        print(f"   {i}. {size} (nano, small, medium, large, xlarge)")
    
    size_choice = input("Select model size (1-5): ").strip()
    if size_choice.isdigit() and 1 <= int(size_choice) <= 5:
        config["model_size"] = sizes[int(size_choice) - 1]
    
    # Training parameters
    print("\n2. Training Parameters:")
    epochs = input("Number of epochs (default: 100): ").strip()
    if epochs.isdigit():
        config["epochs"] = int(epochs)
    
    batch_size = input("Batch size (default: 16): ").strip()
    if batch_size.isdigit():
        config["batch_size"] = int(batch_size)
    
    image_size = input("Image size (default: 640): ").strip()
    if image_size.isdigit():
        config["image_size"] = int(image_size)
    
    return config
```

#### **Partial Interactive Mode**
```python
def get_interactive_yolo_version() -> str:
    """Get YOLO version interactively."""
    print("\n=== YOLO Version Selection ===")
    print("1. YOLO11 (Latest, most advanced)")
    print("2. YOLOv8 (Fast, accurate)")
    print("3. YOLOv5 (Stable, well-tested)")
    
    choice = input("Select YOLO version (1-3): ").strip()
    
    if choice == "1":
        return "yolo11"
    elif choice == "2":
        return "yolov8"
    elif choice == "3":
        return "yolov5"
    else:
        print("Invalid choice, defaulting to YOLOv8")
        return "yolov8"
```

#### **Non-Interactive Mode**
```bash
# Use all defaults, no prompts
python train.py --model-type yolov8 --non-interactive --results-folder my_experiment

# This will:
# 1. Use YOLOv8 with nano size
# 2. Use default training parameters
# 3. Create results folder 'my_experiment'
# 4. Start training immediately
```

### **Configuration Sources and Priority**

The system merges configuration from multiple sources:

```python
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
```

**Configuration Priority (highest to lowest):**
1. **Command line arguments** - Direct overrides
2. **Interactive input** - User choices
3. **Configuration file** - Saved settings
4. **Default values** - System defaults

## Phase 4: Training Phase

### **Training Execution Workflow**

The training phase orchestrates all components:

#### **Step 1: Model Loading**
```python
def load_yolo_model(config: YOLOConfig, checkpoint_manager: CheckpointManager, resume_path: Optional[str] = None):
    """Load YOLO model for training."""
    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        model = load_model_from_checkpoint(resume_path)
    else:
        logger.info(f"Loading model: {config.weights}")
        model = load_model_from_weights(config.weights)
    
    return model
```

#### **Step 2: Training Initialization**
```python
# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(
    save_dir=Path(config.logging_config["log_dir"]) / "checkpoints",
    max_checkpoints=config.logging_config["num_checkpoint_keep"],
)

# Initialize training monitor with TensorBoard integration
monitor = TrainingMonitor(
    config=config, 
    log_dir=config.logging_config["log_dir"]
)
# Training monitor automatically:
# - Launches TensorBoard in browser
# - Provides real-time training visualization
# - Keeps TensorBoard running after training completion
```

#### **Step 3: TensorBoard Integration**
The system provides automatic TensorBoard integration for real-time monitoring:

**Automatic Features:**
- **Browser Launch**: TensorBoard opens automatically during training
- **Real-time Metrics**: Live loss curves, accuracy plots, mAP tracking
- **Model Visualization**: Network architecture and computational graphs
- **Persistent Access**: TensorBoard remains running after training completes
- **Smart Directory Detection**: Automatically finds Ultralytics log structure

**Manual Management:**
```bash
# Check TensorBoard status and open browser
python -m utils.tensorboard_manager status

# Launch TensorBoard for specific experiment
python -m utils.tensorboard_manager launch experiment_name

# List all experiments with TensorBoard data
python -m utils.tensorboard_manager list

# Stop TensorBoard when done
python -m utils.tensorboard_manager stop
```

#### **Step 4: Training Execution**
```python
# Create Ultralytics YOLO instance for training
from ultralytics import YOLO
yolo_model = YOLO(config.weights)

# Start training
logger.info("Starting Ultralytics training...")
results = yolo_model.train(
    data=str(config.dataset_config["data_yaml_path"]),
    epochs=config.epochs,
    imgsz=config.image_size,
    batch=config.batch_size,
    device=config.device,
    workers=config.num_workers,
    patience=config.patience,
    save_period=config.logging_config.get("save_period", -1),
    project=config.logging_config["log_dir"],
    name=results_folder,
    exist_ok=True,
    pretrained=config.pretrained,
    optimizer="auto",
    lr0=config.model_config.get("learning_rate", 0.01),
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    close_mosaic=10,
    verbose=True,
)
```

#### **Step 4: Training Monitoring**
```python
class TrainingMonitor:
    def __init__(self, config: YOLOConfig, log_dir: str):
        self.config = config
        self.log_dir = log_dir
        self.metrics = {}
        self.start_time = time.time()
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics."""
        self.metrics[epoch] = metrics
        
        # Log to TensorBoard
        if self.config.logging_config.get("enable_tensorboard", True):
            self._log_to_tensorboard(epoch, metrics)
        
        # Log to Weights & Biases
        if self.config.logging_config.get("enable_wandb", False):
            self._log_to_wandb(epoch, metrics)
        
        # Save to local file
        self._save_metrics(epoch, metrics)
```

### **Training Modes**

#### **Full Training Mode**
```bash
# Complete training with all features
python train.py

# This will:
# 1. Guide through YOLO version selection
# 2. Configure model size and parameters
# 3. Prepare dataset automatically
# 4. Train the model
# 5. Save results and checkpoints
```

#### **Validation Only Mode**
```bash
# Only validate existing model
python train.py --validate-only --resume path/to/checkpoint.pt

# This will:
# 1. Load the specified model
# 2. Run validation on test set
# 3. Generate evaluation metrics
# 4. Skip training entirely
```

#### **Resume Training Mode**
```bash
# Resume from checkpoint
python train.py --resume logs/training_run/weights/last.pt

# This will:
# 1. Load model state from checkpoint
# 2. Continue training from last epoch
# 3. Maintain all previous progress
# 4. Save new checkpoints
```

## Phase 5: Results Phase

### **Training Results Management**

#### **Results Directory Structure**
```
logs/
└── training_run/
    ├── weights/
    │   ├── best.pt          # Best model weights
    │   ├── last.pt          # Last epoch weights
    │   └── epoch_50.pt      # Specific epoch weights
    ├── checkpoints/
    │   ├── checkpoint_epoch_25.pt
    │   └── checkpoint_epoch_50.pt
    ├── events.out.tfevents  # TensorBoard logs
    ├── results.csv          # Training metrics
    ├── confusion_matrix.png # Confusion matrix
    ├── labels_correlogram.jpg # Label correlation
    └── results.png          # Training curves
```

#### **Results Analysis**
```python
def analyze_training_results(results_dir: Path):
    """Analyze training results."""
    logger.info("Analyzing training results...")
    
    # Load training metrics
    results_csv = results_dir / "results.csv"
    if results_csv.exists():
        metrics = pd.read_csv(results_csv)
        
        # Plot training curves
        plot_training_curves(metrics)
        
        # Generate confusion matrix
        generate_confusion_matrix(results_dir)
        
        # Calculate final metrics
        final_map50 = metrics['metrics/mAP50(B)'].iloc[-1]
        final_map50_95 = metrics['metrics/mAP50-95(B)'].iloc[-1]
        
        logger.info(f"Final mAP50: {final_map50:.3f}")
        logger.info(f"Final mAP50-95: {final_map50_95:.3f}")
```

## Phase 6: Export Phase

### **Model Export Workflow**

#### **Automatic Export During Training**
```python
# Export model if requested
if args.export:
    logger.info("Exporting model...")
    from utils.export_utils import export_model
    
    export_model(model, config)
```

#### **Manual Export After Training**
```bash
# Export existing trained models
python utils/export_existing_models.py

# This will:
# 1. Find all .pt files in your project
# 2. Export each to multiple formats
# 3. Organize by model name
# 4. Validate exports
```

## Complete Workflow Examples

### **Example 1: Beginner's First Training**

```bash
# 1. Place dataset in dataset/ folder
# 2. Run training
python train.py

# System will:
# - Detect dataset automatically
# - Convert to YOLO format
# - Guide through configuration
# - Train the model
# - Save results
```

### **Example 2: Production Training**

```bash
# 1. Prepare configuration file
cp config/production_config.yaml my_config.yaml

# 2. Run non-interactive training
python train.py \
  --config my_config.yaml \
  --non-interactive \
  --results-folder production_run_v1 \
  --export

# System will:
# - Use production configuration
# - Skip all prompts
# - Train with optimized settings
# - Export to deployment formats
```

### **Example 3: Research Experiment**

```bash
# 1. Run with custom parameters
python train.py \
  --model-type yolo11 \
  --epochs 200 \
  --batch-size 32 \
  --image-size 1024 \
  --device cuda \
  --results-folder research_experiment_1

# System will:
# - Use YOLO11 with custom settings
# - Optimize for research needs
# - Provide detailed logging
# - Enable advanced monitoring
```

## Workflow Integration Points

### **Component Interactions**

#### **Dataset → Training Integration**
```python
# Dataset preparation automatically updates training configuration
prepared_dataset = auto_prepare_dataset_if_needed(model_type)
config.dataset_config["data_yaml_path"] = prepared_dataset / "data.yaml"
config.dataset_config["num_classes"] = len(detected_classes)
```

#### **Configuration → Training Integration**
```python
# Configuration drives all training parameters
training_params = {
    "epochs": config.epochs,
    "batch_size": config.batch_size,
    "image_size": config.image_size,
    "device": config.device,
    "learning_rate": config.model_config.get("learning_rate", 0.01)
}
```

#### **Training → Monitoring Integration**
```python
# Training progress feeds monitoring system
monitor.log_metrics(epoch, {
    "loss": current_loss,
    "mAP50": current_map50,
    "mAP50-95": current_map50_95
})
```

#### **Training → Export Integration**
```python
# Training completion triggers export
if training_completed and args.export:
    export_trained_model(training_results, config)
```

## Error Handling and Recovery

### **Workflow Resilience**

#### **Dataset Preparation Failures**
```python
try:
    prepared_path = auto_prepare_dataset(dataset_root, model_type)
    logger.info(f"Dataset prepared successfully at: {prepared_path}")
    return prepared_path
except Exception as e:
    logger.error(f"Failed to prepare dataset automatically: {e}")
    logger.info("Please prepare your dataset manually or check the error above.")
    raise
```

#### **Training Interruptions**
```python
try:
    # Training code
    results = yolo_model.train(...)
except KeyboardInterrupt:
    logger.info("Training interrupted by user")
    # Save current state
    save_interrupted_state(model, config)
    sys.exit(1)
except Exception as e:
    logger.error(f"Training failed with error: {e}")
    logger.exception("Full traceback:")
    # Attempt recovery
    attempt_training_recovery(model, config)
    sys.exit(1)
```

#### **Configuration Validation**
```python
def validate_configuration(config: YOLOConfig) -> bool:
    """Validate training configuration."""
    errors = []
    
    # Check required fields
    if not config.model_type:
        errors.append("Model type is required")
    
    if not config.weights:
        errors.append("Model weights are required")
    
    if config.epochs <= 0:
        errors.append("Epochs must be positive")
    
    if config.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    # Report errors
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        return False
    
    return True
```

## Performance Optimization

### **Workflow-Level Optimization**

#### **Parallel Processing**
```python
# Enable parallel data loading
config.num_workers = min(8, os.cpu_count())

# Enable mixed precision training
config.training_config["enable_amp"] = True

# Enable gradient accumulation
config.training_config["accumulate"] = 4
```

#### **Memory Management**
```python
# Optimize batch size for available memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory < 8:
        config.batch_size = 8
    elif gpu_memory < 16:
        config.batch_size = 16
    else:
        config.batch_size = 32
```

#### **Caching Strategies**
```python
# Enable dataset caching
config.dataset_config["enable_cache"] = True
config.dataset_config["cache_size"] = 10  # GB

# Enable model caching
config.model_config["enable_model_cache"] = True
```

## Best Practices for Workflows

### **For Beginners**
1. **Start simple**: Use default settings first
2. **Test workflow**: Run with small dataset first
3. **Monitor progress**: Watch training logs
4. **Save checkpoints**: Enable automatic checkpointing

### **For Intermediate Users**
1. **Customize configuration**: Modify training parameters
2. **Optimize performance**: Adjust batch size and workers
3. **Experiment tracking**: Use TensorBoard or wandb
4. **Model comparison**: Train multiple models

### **For Advanced Users**
1. **Custom workflows**: Create specialized training loops
2. **Advanced monitoring**: Implement custom metrics
3. **Performance tuning**: Profile and optimize bottlenecks
4. **Production deployment**: Implement CI/CD pipelines

---

**Next**: We'll explore [Data Flow Integration](02-data-flow-integration.md) to understand how data moves through the entire system.
