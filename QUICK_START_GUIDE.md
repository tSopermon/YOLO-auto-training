# Quick Start Guide - Train Your First YOLO Model

This guide will get you training a YOLO model with your dataset in minutes. Choose your experience level and follow the corresponding path.

## Prerequisites

- Python 3.8+ installed
- Git (to clone the repository)
- Basic command line knowledge

## Environment Setup

### 1. Clone and Navigate
```bash
git clone <your-repository-url>
cd YOLO-auto-training
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration (Optional)
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your settings (optional)
# Most settings have sensible defaults
```

## Beginner Path (Zero Experience)

### Step 1: Prepare Your Dataset
Place your dataset in the `dataset/` folder. The system automatically detects and converts any format:
- **YOLO format**: `dataset/train/images/`, `dataset/train/labels/`
- **COCO format**: `dataset/annotations.json`
- **XML format**: `dataset/annotations/`
- **Mixed formats**: Any combination of the above

### Step 2: Run Training
```bash
# If you have a virtual environment (.venv), use:
.venv/bin/python train.py              # Linux/Mac
# .venv\Scripts\python.exe train.py    # Windows

# Or if using system Python:
python train.py
```

The system will guide you through:
1. YOLO version selection (YOLO11, YOLOv8, YOLOv5)
2. Model size (nano, small, medium, large, xlarge)
3. Training parameters (epochs, batch size, image size)
4. Results folder naming

### Step 3: Monitor Progress
Training automatically:
- **Launches TensorBoard** in your browser for real-time monitoring
- Shows live training metrics, loss curves, and model performance
- Saves checkpoints every 10 epochs
- Creates organized results in `logs/` folder
- **Keeps TensorBoard running** after training for result analysis

**TensorBoard Features:**
- Real-time loss curves and accuracy plots
- Model architecture visualization
- Training image samples with augmentations
- Hyperparameter tracking
- Performance metrics (mAP50, mAP50-95, precision, recall)

### Step 4: Manage TensorBoard (Optional)
```bash
# Use your Python executable (adjust path as needed)
python -m utils.tensorboard_manager status    # Check status & open browser
python -m utils.tensorboard_manager list      # View all your experiments
python -m utils.tensorboard_manager stop      # Stop TensorBoard when done
```

### Step 5: Get Your Model
After training, find your trained model in:
```
logs/your_experiment_name/weights/best.pt
```

## Intermediate Path (Some ML Experience)

### Step 1: Dataset Preparation
```bash
# Place dataset in dataset/ folder
# Or use the automated preparation utility
python utils/prepare_dataset.py dataset/ --format yolov8
```

### Step 2: Custom Training
```bash
# Train with specific parameters
python train.py --model-type yolov8 --epochs 200 --batch-size 16

# Use custom results folder
python train.py --model-type yolov8 --results-folder my_experiment

# Resume interrupted training (only works if training was interrupted)
python train.py --model-type yolov8 --resume logs/previous_run/weights/last.pt

# Extend completed training beyond original epochs
python train.py --model-type yolov8 --epochs 200 --resume logs/previous_run/weights/last.pt
```

**TensorBoard automatically launches** during training for real-time monitoring.

### Step 3: TensorBoard Management
```bash
# View training progress (opens browser)
python -m utils.tensorboard_manager status

# Launch TensorBoard for specific experiment
python -m utils.tensorboard_manager launch my_experiment

# List all experiments and their data status
python -m utils.tensorboard_manager list
```

### Step 4: Export for Deployment
```bash
# Export to multiple formats after training
python utils/export_existing_models.py logs/your_experiment/weights/best.pt

# Or export during training
python train.py --model-type yolov8 --export
```

## Advanced Path (ML Expert)

### Step 1: Custom Configuration
```bash
# Create custom config YAML file (NOT Python file)
cp config/advanced_solar_defect.yaml config/my_config.yaml
# Edit my_config.yaml with your parameters

# Use custom config (completely non-interactive)
python train.py --config config/my_config.yaml

# Override specific config values via command line
python train.py --config config/my_config.yaml --epochs 200 --batch-size 16
```

**Important**: Configuration files must be in YAML format, not Python. See `config/advanced_solar_defect.yaml` for an example.

**New Feature**: Configuration files now run completely non-interactively - no prompts for model selection, results folder, or any other parameters. The system auto-generates appropriate folder names and respects all config values.

**Note**: Replace `python` with your virtual environment path if needed:
- Linux/Mac: `.venv/bin/python` or `venv/bin/python`  
- Windows: `.venv\Scripts\python.exe` or `venv\Scripts\python.exe`

### Step 2: Advanced Training
```bash
# GPU training (use 'cuda', not specific device numbers)
python train.py --model-type yolov8 --device cuda

# Large model with high epochs  
python train.py --model-type yolov8 --epochs 300 --batch-size 32

# Custom image size - use 'cuda' or 'cpu', not 'auto'
python train.py --model-type yolov8 --image-size 1024 --device cuda
```

**GPU Memory Note**: If you get CUDA out of memory errors, reduce batch size:
```bash
python train.py --model-type yolov8 --batch-size 8 --device cuda
```

### Step 3: Custom Export Pipeline
```python
from utils.export_utils import YOLOExporter

exporter = YOLOExporter("path/to/model.pt")
formats = ["onnx", "torchscript", "openvino", "coreml"]
exported = exporter.export_all_formats(formats=formats)
```

## Environment Files

### .env File (Optional)
Create a `.env` file in the root directory for custom settings:

```bash
# Copy example file
cp env.example .env
```

Key settings in `.env`:
```bash
# Dataset paths
DATASET_ROOT=./dataset
EXPORT_ROOT=./exported_models

# Training defaults
DEFAULT_EPOCHS=100
DEFAULT_BATCH_SIZE=8
DEFAULT_IMAGE_SIZE=640

# Hardware settings
DEFAULT_DEVICE=cuda
NUM_WORKERS=4

# Export settings
EXPORT_FORMATS=onnx,torchscript,openvino
```

### Example YAML Configuration

Here's an example custom configuration file (`config/my_config.yaml`):

```yaml
# Model Configuration
model_type: "yolo11"
weights: "pretrained_weights/yolo11m.pt"

# Training Parameters  
epochs: 100
batch_size: 16
image_size: 640
device: "cuda"

# Model Configuration
model_config:
  dropout: 0.2

# Augmentation
augmentation_config:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 10
  translate: 0.1
  scale: 0.5
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.1

# Export Configuration (REQUIRED)
export_config:
  export_formats: ["onnx", "torchscript", "openvino"]
  export_dir: "exported_models"
  include_nms: true
  batch_size: 1
```

### Environment Variables (Alternative)
Set environment variables directly:

```bash
export DATASET_ROOT=./dataset
export DEFAULT_DEVICE=cuda
export DEFAULT_EPOCHS=200
```

## Resume vs Extend Training

Understanding the difference between resuming and extending training:

### Resume Training (Interrupted Training)
Use when training was stopped mid-process (power outage, interruption, etc.):
```bash
# Resume from where training left off
python train.py --model-type yolov8 --resume logs/experiment/weights/last.pt
```

### Extend Training (Add More Epochs)
Use when you want to train a completed model for additional epochs:
```bash
# Extend from 100 epochs to 200 epochs
python train.py --model-type yolov8 --epochs 200 --resume logs/experiment/weights/last.pt
```

### What Happens Automatically
- **Smart Detection**: The system automatically detects whether training was completed or interrupted
- **Helpful Guidance**: If you use the wrong approach, you'll get clear instructions with examples
- **No Interactive Prompts**: When you provide command-line arguments or config files, the system respects them completely
- **Fixed Resume Logic**: Recent improvements ensure resume functionality works correctly in all scenarios

### Example Error Guidance
If you try to resume completed training without specifying epochs:
```
CANNOT RESUME: CHECKPOINT TRAINING ALREADY COMPLETED
The checkpoint has already completed 10 epochs of training.
To extend training beyond the original epochs, you must specify --epochs:

Examples:
  python train.py --model-type yolov8 --epochs 60 --resume logs/experiment/weights/last.pt
  python train.py --model-type yolov8 --epochs 200 --resume logs/experiment/weights/last.pt
```

## Common Commands Reference

### Training Commands
```bash
# Full interactive (beginner)
python train.py

# Partial interactive
python train.py --model-type yolov8

# Fully automated
python train.py --model-type yolov8 --non-interactive

# GPU training (recommended)
python train.py --model-type yolov8 --device cuda

# CPU training (if no GPU)
python train.py --model-type yolov8 --device cpu

# Custom configuration file (completely non-interactive)
python train.py --config config/my_config.yaml

# Override config values via command line
python train.py --config config/my_config.yaml --epochs 150

# Resume interrupted training (training was stopped mid-process)
python train.py --model-type yolov8 --resume logs/experiment/weights/last.pt

# Extend completed training (add more epochs to finished training)
python train.py --model-type yolov8 --epochs 200 --resume logs/experiment/weights/last.pt

# Validation only
python train.py --model-type yolov8 --validate-only
```

### Utility Commands
```bash
# Test dataset preparation (shows what would be done)
python utils/prepare_dataset.py dataset/ --format yolov8 --verbose

# Create a test dataset for experimentation
python examples/create_test_dataset.py --output test_dataset --images 20

# Download pretrained weights (interactive)
python utils/download_pretrained_weights.py

# Download specific weights (command line)
python utils/download_pretrained_weights.py --model yolov8 --size n

# Export existing model to multiple formats
python utils/export_existing_models.py path/to/model.pt

# Run all tests to verify system
python -m pytest tests/

# TensorBoard management
python -m utils.tensorboard_manager status  # Check if running
python -m utils.tensorboard_manager list    # List experiments
python -m utils.tensorboard_manager stop    # Stop TensorBoard
```

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size with `--batch-size 4` or `--batch-size 8`
2. **Dataset not found**: Ensure dataset is in `dataset/` folder
3. **Import errors**: Activate virtual environment with `source .venv/bin/activate`
4. **CUDA not working**: Use `--device cuda` instead of `--device auto` or specific device numbers
5. **Resume not working**: 
   - For interrupted training: `python train.py --model-type yolov8 --resume path/to/last.pt`
   - For extending completed training: `python train.py --model-type yolov8 --epochs NEW_COUNT --resume path/to/last.pt`
   - The system will guide you with specific examples if you use the wrong approach

### GPU Memory Issues
If you have CUDA available but training fails:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Check if your configuration will fit before training
python gpu_memory_cli.py check --model l --image-size 1280 --batch-size 4 --version yolov8

# Use smaller batch size if memory issues
python train.py --device cuda --batch-size 4

# Clear GPU memory if needed
python gpu_memory_cli.py clear
```

### Configuration File Issues
- **Use YAML format**, not Python files: `config/my_config.yaml`
- **Required fields**: `model_type`, `weights`, `export_config` with `export_formats`
- **Example**: See `config/advanced_solar_defect.yaml`
- **No duplicate keys**: Ensure YAML file has no duplicate entries (like `device` appearing twice)
- **Non-interactive**: Config files now run completely without prompts - all values are respected

### Getting Help
- Check the comprehensive documentation in `docs/workflow/`
- Review error messages in the terminal
- Check the `logs/` folder for detailed training logs

## Next Steps

After your first successful training:
1. **Experiment**: Try different model sizes and parameters
2. **Optimize**: Use the performance optimization guide
3. **Deploy**: Export models for production use
4. **Scale**: Set up automated training pipelines

## File Structure After Training

```
YOLO-auto-training/
├── logs/                           # Training results
│   └── your_experiment/
│       ├── weights/
│       │   ├── best.pt            # Best model
│       │   └── last.pt            # Latest checkpoint
│       └── training_log.txt       # Training history
├── exported_models/                # Deployed models
│   └── your_experiment/
│       ├── model.onnx
│       ├── model.torchscript
│       └── model.coreml
└── dataset/                        # Your dataset
    ├── train/
    ├── valid/
    └── test/
```

---

**Start Training**: `python train.py`

**Need Help?**: Check `docs/workflow/README.md` for comprehensive documentation.
