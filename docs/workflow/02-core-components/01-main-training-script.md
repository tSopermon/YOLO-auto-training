# Main Training Script (train.py)

## What This File Does

`train.py` is the heart of the entire system.

## File Overview

**File**: `train.py`  
**Purpose**: Main training script that coordinates the entire training process  
**Size**: 635 lines of code  
**Dependencies**: Multiple utility modules and configuration systems

## Main Functions Breakdown

### **1. Main Entry Point (`main()`)**

This is where everything starts. The main function:

```python
def main():
    # 1. Parse command line arguments
    # 2. Set up configuration
    # 3. Prepare dataset automatically
    # 4. Initialize training components
    # 5. Start training or validation
    # 6. Handle results and export
```

**What it does:**
- Coordinates the entire training workflow
- Handles both interactive and non-interactive modes
- Manages errors and user interruptions
- Ensures everything is set up correctly before training

### **2. Interactive User Interface Functions**

#### **`get_custom_results_folder()`**
- **Purpose**: Gets a custom name for your training results folder
- **What it does**: 
  - Prompts you to name your results folder
  - Cleans the name (removes invalid characters)
  - Checks if folder already exists
  - Asks for permission to overwrite if needed
- **Example output**:
  ```
  ============================================================
  YOLO Training Results Folder
  ============================================================
  Enter a custom name for your training results folder.
  This will create a folder like: logs/your_custom_name/
  Examples: experiment_1, car_parts_v1, test_run_2024
  ============================================================
  ```

#### **`get_interactive_yolo_version()`**
- **Purpose**: Lets you choose which YOLO version to train
- **Options**:
  1. **YOLO11** - Latest version with best performance
  2. **YOLOv8** - Stable, well-tested, recommended
  3. **YOLOv5** - Classic version, very stable
- **Default**: YOLOv8 (if you just press Enter)

#### **`get_interactive_config(model_type)`**
- **Purpose**: Guides you through all training parameters
- **Parameters it asks for**:
  - **Model Size**: n (nano), s (small), m (medium), l (large), x (xlarge)
  - **Training Epochs**: How long to train (default: 100)
  - **Batch Size**: How many images per batch (default: 8)
  - **Image Size**: Input resolution (default: 1024)
  - **Learning Rate**: How fast to learn (default: 0.01)
  - **Device**: CPU or GPU (auto-detects)
  - **Advanced Options**: Early stopping, data augmentation, validation frequency

### **3. Command Line Interface (`parse_args()`)**

This function handles all the command-line options you can use:

#### **Basic Options**
- `--model-type`: Choose YOLO version (yolo11, yolov8, yolov5)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--image-size`: Input image resolution
- `--device`: Training device (cpu, cuda, auto)

#### **Advanced Options**
- `--resume`: Resume from a checkpoint
- `--validate-only`: Only test the model, don't train
- `--export`: Export model after training
- `--results-folder`: Custom results folder name
- `--non-interactive`: Skip all prompts, use defaults
- `--config`: Use custom configuration file

#### **Example Usage**:
```bash
# Fully automated training
python train.py --model-type yolov8 --non-interactive --results-folder my_experiment

# Custom parameters
python train.py --model-type yolo11 --epochs 200 --batch-size 16 --image-size 640

# Resume training
python train.py --resume logs/previous_run/checkpoints/last.pt
```

### **4. Configuration Management Functions**

#### **`update_config_from_args(config, args)`**
- **Purpose**: Applies command-line arguments to configuration
- **What it does**: Overrides default settings with your command-line choices
- **Example**: If you use `--epochs 200`, it sets training to 200 epochs

#### **`update_config_from_interactive(config, interactive_config)`**
- **Purpose**: Applies interactive user choices to configuration
- **What it does**: Takes your interactive input and updates the training settings
- **Example**: If you choose "large" model size, it sets the right weight file

## How the Training Process Works

### **Step 1: Setup and Initialization**
```python
# Parse arguments and get configuration
args = parse_args()
config = get_config()

# Set up logging and results folder
results_folder = get_custom_results_folder()
setup_logging(config)
```

### **Step 2: Dataset Preparation**
```python
# Automatically prepare dataset
prepared_dataset_path = auto_prepare_dataset_if_needed(args.model_type)
config.dataset_config["data_yaml_path"] = str(prepared_dataset_path / "data.yaml")
```

**What happens**:
- System checks if dataset needs preparation
- Automatically converts formats if needed
- Creates proper folder structure
- Generates `data.yaml` configuration file

### **Step 3: Component Initialization**
```python
# Set up checkpoint management
checkpoint_manager = CheckpointManager(...)

# Set up training monitoring
monitor = TrainingMonitor(...)

# Load the model
model = load_yolo_model(config, checkpoint_manager, args.resume)
```

**What happens**:
- Creates systems to save training progress
- Sets up monitoring and logging
- Loads the YOLO model with correct weights

### **Step 4: Training Execution**
```python
# Create Ultralytics YOLO instance
yolo_model = YOLO(config.weights)

# Start training
results = yolo_model.train(
    data=str(config.dataset_config["data_yaml_path"]),
    epochs=config.epochs,
    imgsz=config.image_size,
    batch=config.batch_size,
    device=config.device,
    # ... other parameters
)
```

**What happens**:
- Creates the actual training instance
- Starts the training loop
- Monitors progress and saves checkpoints
- Handles early stopping if configured

### **Step 5: Results and Export**
```python
# Log results
logger.info(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")

# Export if requested
if args.export:
    export_model(model, config)
```

**What happens**:
- Records training performance metrics
- Exports model to different formats if requested
- Organizes all results in the specified folder

## Training Modes

### **1. Interactive Mode (Default)**
- **What it does**: Guides you through every decision step-by-step
- **Best for**: Beginners, learning, one-time training
- **How to use**: Just run `python train.py`
- **What happens**: System asks for each parameter with explanations

### **2. Non-Interactive Mode**
- **What it does**: Uses all default values automatically
- **Best for**: Automation, scripting, repeated training
- **How to use**: Add `--non-interactive` flag
- **What happens**: No prompts, training starts immediately

### **3. Custom Configuration Mode**
- **What it does**: Uses your specific parameters
- **Best for**: Experienced users, production training
- **How to use**: Specify parameters on command line
- **What happens**: Your settings override defaults

### **4. Validation Only Mode**
- **What it does**: Tests existing model without training
- **Best for**: Evaluating pre-trained models
- **How to use**: Add `--validate-only` flag
- **What happens**: Model is tested on validation data

## Error Handling and Recovery

### **What the System Handles Automatically**
- **Dataset issues**: Detects and fixes common problems
- **Configuration errors**: Validates parameters and suggests fixes
- **Resource problems**: Checks GPU memory, disk space
- **Training failures**: Saves progress and allows resuming

### **What You Need to Handle**
- **Keyboard interrupts**: Press Ctrl+C to stop training
- **Hardware failures**: System will save progress before crashing
- **Permission issues**: Make sure you can write to the logs folder

### **Recovery Options**
- **Resume training**: Use `--resume` with checkpoint file
- **Start over**: Delete results folder and run again
- **Debug mode**: Check logs for detailed error information

## Integration with Other Components

### **Configuration System**
- **`config/config.py`**: Provides training configuration
- **`config/constants.py`**: Defines default values
- **`config/__init__.py`**: Makes configuration importable

### **Utility Modules**
- **`utils/auto_dataset_preparer.py`**: Handles dataset preparation
- **`utils/model_loader.py`**: Loads YOLO models
- **`utils/training_utils.py`**: Core training functions
- **`utils/checkpoint_manager.py`**: Manages training progress
- **`utils/training_monitor.py`**: Tracks training metrics

### **External Dependencies**
- **Ultralytics**: The actual YOLO training engine
- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **YAML**: Configuration file handling

## Best Practices for Using train.py

### **For Beginners**
1. **Start simple**: Just run `python train.py` and follow prompts
2. **Use defaults**: Press Enter to accept default values
3. **Read explanations**: Each prompt explains what the setting does
4. **Start small**: Use nano models for first experiments

### **For Intermediate Users**
1. **Customize parameters**: Adjust epochs, batch size, image size
2. **Use command line**: Specify common parameters directly
3. **Monitor training**: Check logs and progress bars
4. **Experiment**: Try different model sizes and configurations

### **For Advanced Users**
1. **Automate workflows**: Use `--non-interactive` mode
2. **Custom configurations**: Create configuration files
3. **Script integration**: Call from other Python scripts
4. **Production use**: Set up automated training pipelines

## Common Use Cases

### **Quick Training Session**
```bash
python train.py
# Follow prompts for quick setup
```

### **Production Training**
```bash
python train.py \
  --model-type yolov8 \
  --epochs 300 \
  --batch-size 16 \
  --image-size 640 \
  --non-interactive \
  --results-folder production_run_v1
```

### **Resume Interrupted Training**
```bash
python train.py --resume logs/my_experiment/checkpoints/last.pt
```

### **Test Existing Model**
```bash
python train.py --validate-only --resume logs/my_experiment/weights/best.pt
```

## Troubleshooting Common Issues

### **"No module named 'ultralytics'"**
- **Solution**: Install with `pip install ultralytics`
- **Why it happens**: Ultralytics is required for YOLO training

### **"CUDA out of memory"**
- **Solution**: Reduce batch size or image size
- **Why it happens**: GPU doesn't have enough memory

### **"Dataset not found"**
- **Solution**: Place dataset in `dataset/` folder
- **Why it happens**: System can't find your training data

### **"Permission denied"**
- **Solution**: Check folder permissions, use different results folder
- **Why it happens**: Can't write to specified location

---

**Next**: We'll explore the [Configuration System](02-configuration-system.md) to understand how all the training parameters are managed.
