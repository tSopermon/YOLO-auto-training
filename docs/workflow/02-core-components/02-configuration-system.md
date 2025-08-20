# Configuration System

## What This System Does

The configuration system is like a smart settings manager that keeps track of all the parameters needed for training YOLO models. Think of it as a recipe book where each recipe (configuration) tells the system exactly how to cook (train) your model.

## Configuration Files Overview

### **File Structure**
```
config/
├── __init__.py      # Makes the folder work as a package
├── config.py        # Main configuration manager
└── constants.py     # All the default values and settings
```

### **File Purposes**
- **`__init__.py`**: Package initialization and exports
- **`config.py`**: Configuration class and management functions
- **`constants.py`**: Default values and preset configurations

## Main Configuration Class: YOLOConfig

### **What It Is**
`YOLOConfig` is a Python dataclass that holds all your training settings in one organized place. It's like a smart container that automatically validates your settings and sets up everything correctly.

### **Key Configuration Sections**

#### **1. Model Configuration**
```python
model_type: str        # Which YOLO version (yolo11, yolov8, yolov5)
weights: str           # Path to pre-trained weights
pretrained: bool       # Whether to use pre-trained weights
```

#### **2. Training Configuration**
```python
epochs: int            # How many times to train on the data
batch_size: int        # How many images to process at once
image_size: int        # Input image resolution
num_workers: int       # How many processes to use for data loading
device: str            # CPU or GPU (cuda)
seed: int              # Random seed for reproducible results
patience: int          # Early stopping patience
```

#### **3. Dataset Configuration**
```python
dataset_config: Dict   # Paths to train/valid/test folders
                       # Dataset settings and options
```

#### **4. Augmentation Configuration**
```python
augmentation_config: Dict  # Data augmentation settings
                           # Mosaic, rotation, scaling, etc.
```

#### **5. Evaluation Configuration**
```python
eval_config: Dict      # How to evaluate the model
                       # Metrics, validation frequency, etc.
```

#### **6. Logging Configuration**
```python
logging_config: Dict   # Where to save logs and results
                       # Checkpoint settings, project names
```

#### **7. Export Configuration**
```python
export_config: Dict    # How to export the trained model
                       # Formats, optimization settings
```

## Default Values and Constants

### **Where Defaults Come From**
All default values are stored in `constants.py` and organized into logical groups:

#### **Common Training Parameters**
```python
COMMON_TRAINING = {
    "seed": 42,              # Random seed for reproducibility
    "epochs": 100,           # Default training duration
    "patience": 50,          # Early stopping patience
    "batch_size": 8,         # Default batch size
    "image_size": 1024,      # Default image resolution
    "num_workers": 4,        # Data loading processes
    "device": "cuda",        # Use GPU if available
    "pretrained": True,      # Use pre-trained weights
    "deterministic": True,   # Reproducible results
    "single_cls": False,     # Multi-class training
    "rect": False,           # Rectangular training
    "cos_lr": True,          # Cosine learning rate schedule
    "close_mosaic": 10,      # Disable mosaic in last epochs
    "resume": False,         # Don't resume by default
}
```

#### **YOLO Version-Specific Settings**
Each YOLO version has its own optimized settings:

**YOLOv8 Configuration**
```python
YOLOV8_CONFIG = {
    "model_type": "yolov8",
    "weights": "yolov8n.pt",     # Default to nano model
    "learning_rate": 0.01,       # Default learning rate
    "optimizer": "auto",         # Ultralytics auto-optimizer
    "lr_scheduler": "cosine",    # Cosine decay schedule
    "warmup_epochs": 3.0,       # Gradual warmup
    "box": 7.5,                 # Box loss weight
    "cls": 0.5,                 # Classification loss weight
    "dfl": 1.5,                 # Distribution focal loss weight
}
```

**YOLOv5 Configuration**
```python
YOLOV5_CONFIG = {
    "model_type": "yolov5",
    "weights": "yolov5nu.pt",    # Default to nano model
    "learning_rate": 0.01,       # Default learning rate
    "optimizer": "SGD",          # Stochastic gradient descent
    "lr_scheduler": "cosine",    # Cosine decay schedule
    "warmup_epochs": 3.0,       # Gradual warmup
    "box": 0.05,                # Box loss weight
    "cls": 0.5,                 # Classification loss weight
    "obj": 1.0,                 # Objectness loss weight
}
```

**YOLO11 Configuration**
```python
YOLO11_CONFIG = {
    "model_type": "yolo11",
    "weights": "yolo11n.pt",     # Default to nano model
    "learning_rate": 0.01,       # Default learning rate
    "optimizer": "auto",         # Auto-optimizer selection
    "lr_scheduler": "cosine",    # Cosine decay schedule
    "warmup_epochs": 3.0,       # Gradual warmup
    # ... other YOLO11-specific settings
}
```

## How Configuration Works

### **1. Configuration Creation**
When you start training, the system:

```python
# 1. Creates base configuration from constants
config = YOLOConfig(model_type="yolov8")

# 2. Applies model-specific settings
config.model_config.update(YOLOV8_CONFIG)

# 3. Applies user overrides (command line or interactive)
config.epochs = 200  # User wants 200 epochs instead of 100
```

### **2. Automatic Validation**
The system automatically checks if your settings make sense:

```python
def _validate_config(self):
    # Check model type is valid
    if self.model_type not in ["yolo11", "yolov8", "yolov5"]:
        raise ValueError(f"Unsupported model type: {self.model_type}")
    
    # Check image size is valid
    if not isinstance(self.image_size, (int, list, tuple)):
        raise ValueError(f"Invalid image size: {self.image_size}")
    
    # Check batch size is positive
    if self.batch_size < 1:
        raise ValueError(f"Invalid batch size: {self.batch_size}")
    
    # Check patience is non-negative
    if self.patience < 0:
        raise ValueError(f"Invalid patience value: {self.patience}")
```

### **3. Automatic Setup**
The system automatically sets up everything you need:

```python
def _setup_paths(self):
    # Create necessary folders
    # Set up logging paths
    # Prepare dataset paths

def _setup_device(self):
    # Check if GPU is available
    # Set device automatically if needed
    # Validate device choice

def _setup_wandb(self):
    # Set up Weights & Biases logging if enabled
    # Configure project names and tags
```

## Configuration Sources and Priority

### **Priority Order (Highest to Lowest)**
1. **Command Line Arguments** - `--epochs 200`
2. **Interactive User Input** - User chooses during prompts
3. **Configuration File** - Custom config file
4. **Model-Specific Defaults** - YOLOv8, YOLOv5, YOLO11 settings
5. **Common Defaults** - General training defaults

### **Example of Priority in Action**
```python
# Default value in constants.py
COMMON_TRAINING["epochs"] = 100

# Model-specific override
YOLOV8_CONFIG["epochs"] = 150

# User interactive choice
interactive_config["epochs"] = 200

# Command line argument
args.epochs = 300

# Final result: 300 epochs (command line wins)
```

## How to Customize Configuration

### **Method 1: Interactive Mode**
```bash
python train.py
# System will ask for each parameter
# Your choices override defaults
```

### **Method 2: Command Line Arguments**
```bash
python train.py --epochs 200 --batch-size 16 --image-size 640
# These values override all defaults
```

### **Method 3: Custom Configuration File**
```python
# Create custom_config.py
CUSTOM_CONFIG = {
    "epochs": 500,
    "batch_size": 32,
    "learning_rate": 0.001,
}

# Use in training
python train.py --config custom_config.py
```

### **Method 4: Modify Constants**
```python
# Edit config/constants.py
COMMON_TRAINING["epochs"] = 200  # Change default epochs
COMMON_TRAINING["batch_size"] = 16  # Change default batch size
```

## Configuration Validation and Safety

### **What Gets Validated**
- **Model type**: Must be supported YOLO version
- **Image size**: Must be valid integer or list
- **Batch size**: Must be positive number
- **Epochs**: Must be positive number
- **Device**: Must be available (GPU exists if cuda selected)
- **Paths**: Must be accessible and writable

### **Automatic Fixes**
- **Device selection**: Auto-falls back to CPU if GPU unavailable
- **Path creation**: Automatically creates missing folders
- **File permissions**: Checks and reports permission issues
- **Resource validation**: Warns about insufficient memory/disk space

### **Error Messages**
The system provides clear error messages when something goes wrong:

```python
# Example error messages
"Unsupported model type: yolov6"  # Only yolo11, yolov8, yolov5 supported
"Invalid image size: -640"        # Image size must be positive
"Invalid batch size: 0"           # Batch size must be at least 1
"GPU not available, falling back to CPU"  # Automatic fallback
```

## Configuration Persistence

### **What Gets Saved**
- **Training configuration**: All parameters used for training
- **Dataset paths**: Where your data is located
- **Model settings**: Which model and weights were used
- **Results location**: Where training results are saved

### **Where It Gets Saved**
```python
# Configuration saved to:
config.logging_config["log_dir"] / "config.yaml"

# Example path:
logs/my_experiment/config.yaml
```

### **What the Saved Config Contains**
```yaml
# Example saved configuration
model_type: yolov8
weights: yolov8n.pt
epochs: 200
batch_size: 16
image_size: 640
device: cuda
dataset_config:
  data_yaml_path: dataset_prepared/data.yaml
logging_config:
  log_dir: logs/my_experiment
  project_name: my_experiment
```

## Advanced Configuration Features

### **1. Environment Variable Support**
```python
# Roboflow API key from environment
ROBOFLOW_CONFIG = {
    "api_key": None,  # Will be loaded from ROBOFLOW_API_KEY env var
}
```

### **2. Automatic Path Resolution**
```python
# Paths automatically resolved relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
EXPORT_ROOT = PROJECT_ROOT / "exported_models"
```

### **3. Dynamic Configuration Loading**
```python
# Load data.yaml after dataset preparation
def load_data_yaml(self):
    if Path(self.dataset_config["data_yaml_path"]).exists():
        with open(self.dataset_config["data_yaml_path"], "r") as f:
            data_yaml = yaml.safe_load(f)
        # Update configuration with dataset info
```

### **4. Configuration Inheritance**
```python
# Base configuration with overrides
config = YOLOConfig(model_type="yolov8")
config.model_config.update(YOLOV8_CONFIG)  # Apply YOLOv8 defaults
config.epochs = 200  # Override with user choice
```

## Best Practices for Configuration

### **For Beginners**
1. **Use defaults**: Start with default values
2. **Interactive mode**: Let the system guide you
3. **Small changes**: Modify one parameter at a time
4. **Save configurations**: Keep track of what works

### **For Intermediate Users**
1. **Understand defaults**: Know what each setting does
2. **Experiment systematically**: Change one thing at a time
3. **Use command line**: Specify common parameters directly
4. **Monitor results**: See how changes affect training

### **For Advanced Users**
1. **Create presets**: Build configuration files for common scenarios
2. **Automate workflows**: Use non-interactive mode with custom configs
3. **Optimize parameters**: Fine-tune based on your specific dataset
4. **Version control**: Track configuration changes with your code

## Troubleshooting Configuration Issues

### **Common Problems and Solutions**

#### **"Invalid model type" Error**
- **Problem**: Unsupported YOLO version specified
- **Solution**: Use only yolo11, yolov8, or yolov5

#### **"Invalid image size" Error**
- **Problem**: Image size is not a valid number
- **Solution**: Use positive integers like 640, 1024, 1280

#### **"GPU not available" Warning**
- **Problem**: CUDA selected but no GPU available
- **Solution**: System automatically falls back to CPU

#### **"Path not found" Error**
- **Problem**: Dataset or output folder doesn't exist
- **Solution**: System automatically creates missing folders

#### **"Permission denied" Error**
- **Problem**: Can't write to specified location
- **Solution**: Check folder permissions or use different location

---

**Next**: We'll explore the [Utility Modules](03-utility-modules.md) to understand all the specialized tools that make the system work.
