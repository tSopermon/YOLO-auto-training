# YOLO Model Training with Automated Dataset System

This project provides comprehensive documentation and examples for training various YOLO model versions with **zero dataset preparation required**. The automated dataset system handles any dataset format and structure automatically.

## Zero Dataset Preparation Required!**

### **Simply Place Your Dataset and Train!**
```bash
# 1. Place ANY dataset in dataset/ folder (any structure/format)
# 2. Run training - everything happens automatically!
python train.py
```

**The system automatically:**
- Detects dataset structure (flat, nested, mixed)
- Converts any format (YOLO, COCO, XML, custom)
- Reorganizes to YOLO standard
- Generates `data.yaml` configuration
- Starts training immediately

**Supported Sources:**
- Roboflow exports (any format)
- Kaggle datasets
- Custom annotations
- Mixed sources
- Any organization structure

## Quick Start

**NEW: [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - Get training in minutes with your dataset!

### 1. Setup Environment
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Roboflow API Key (Optional)
```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

### 3. Prepare Dataset (Automatic!)
```bash
# Option A: Automatic (Recommended)
# Just place your dataset in dataset/ folder and run training!

# Option B: Manual preparation (if needed)
python utils/prepare_dataset.py dataset/ --format yolov8

# Option C: Roboflow export
from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key")
project = rf.workspace("workspace").project("project_id")
dataset = project.version("version_number").download("yolov8")
```

### 4. Train YOLO Model (Any Version!)

#### Full Interactive Experience (Recommended for Beginners)
```bash
python train.py
```
The system will guide you through:
- YOLO version selection (YOLO11, YOLOv8, YOLOv5)
- Model size selection (nano to xlarge)
- Training parameters (epochs, batch size, image size)
- Advanced options and results folder naming
- **Automatic TensorBoard launch** for real-time monitoring

#### Partial Interactive (Skip YOLO Selection)
```bash
python train.py --model-type yolov8
```
Uses specified YOLO version, prompts for other parameters

#### Fully Automated (No Prompts)
```bash
python train.py --model-type yolov8 --non-interactive --results-folder my_experiment
```
Uses all defaults, creates organized results folder

#### Custom Configuration (No Prompts)
```bash
python train.py \
  --model-type yolov8 \
  --epochs 100 \
  --batch-size 16 \
  --image-size 640 \
  --device cuda \
  --results-folder production_run \
  --non-interactive
```

### 5. Monitor Training with TensorBoard

**Automatic Monitoring** (Recommended):
- TensorBoard launches automatically during training
- Opens in your browser with real-time metrics
- Continues running after training for result analysis

**Manual TensorBoard Management**:
```bash
# Check TensorBoard status and open in browser
python -m utils.tensorboard_manager status

# List all experiments with TensorBoard data
python -m utils.tensorboard_manager list

# Launch TensorBoard for specific experiment
python -m utils.tensorboard_manager launch experiment_name

# Stop TensorBoard when done
python -m utils.tensorboard_manager stop
```

### 6. Test Automated Dataset System (Optional)
```bash
# Test the automated dataset system
python tests/test_auto_dataset.py

# Run comprehensive YOLO testing
python tests/test_comprehensive_yolo.py

# Run standard tests
python -m pytest tests/ -v
```

## **Automated Dataset System Features**

### **Smart Detection & Conversion**
- **Structure Detection**: Automatically identifies flat, nested, or mixed dataset structures
- **Format Conversion**: Handles YOLO, COCO, XML, and custom annotation formats
- **Class Detection**: Automatically detects classes from labels, annotations, or mapping files
- **Split Management**: Creates optimal train/validation/test splits automatically

### **Zero Configuration Required**
- **Automatic Organization**: Converts any structure to YOLO standard
- **Smart Validation**: Detects and reports dataset issues
- **Error Recovery**: Handles corrupted files and missing labels gracefully
- **YOLO Compatibility**: Works with YOLOv8, YOLOv5, and YOLO11

### **Production Ready**
- **Comprehensive Testing**: 100% test coverage for all YOLO versions
- **Error Handling**: Robust error handling and recovery
- **Performance Optimized**: Fast dataset preparation and validation
- **Integration Ready**: Seamlessly integrates with training pipeline

## Documentation Structure

### **Complete Documentation System**
- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - **NEW!** Get training in minutes
- **[docs/README.md](docs/README.md)** - Main documentation hub
- **[docs/workflow/README.md](docs/workflow/README.md)** - Comprehensive workflow documentation

### **Quick Start with Documentation**
1. **Start here**: [docs/README.md](docs/README.md) - Main documentation hub
2. **System overview**: [docs/workflow/01-system-overview/01-system-overview.md](docs/workflow/01-system-overview/01-system-overview.md)
3. **Training workflows**: [docs/workflow/04-integration-workflows/01-training-workflows.md](docs/workflow/04-integration-workflows/01-training-workflows.md)

### **Complete Documentation Coverage**
The documentation covers **ALL files in ALL repository directories**:
- **System Overview** - What the system does and why it matters
- **Core Components** - Main training script, configuration, utilities, dataset system
- **Supporting Systems** - Examples, testing, export, environment setup
- **Integration Workflows** - Complete training processes, data flow, error handling
- **Validation & Testing** - Quality assurance and maintenance procedures

### **Examples & Scripts**
- **[examples/export_dataset.py](examples/export_dataset.py)** - Practical export script

## Supported YOLO Versions

| Version | Status | Export Format | Training Method |
|---------|--------|---------------|-----------------|
| **YOLO11** | New Latest | `yolo11` | Repository/Ultralytics |
| **YOLOv8** | Recommended | `yolov8` | Ultralytics |
| **YOLOv5** | Stable | `yolo` | Repository/Ultralytics |
| **YOLOv6** | Limited | `yolo` | Repository |
| **YOLOv7** | Limited | `yolo` | Repository |
| **YOLOv9** | Experimental | `yolo` | Repository |

## Training Command Line Options

### Basic Training Commands
```bash
# Full interactive experience - selects YOLO version and all parameters
python train.py

# Train YOLOv8 with interactive configuration (recommended for beginners)
python train.py --model-type yolov8

# Train YOLOv8 with custom parameters (no prompts)
python train.py --model-type yolov8 --epochs 200 --batch-size 16 --image-size 640

# Train with custom results folder (no folder prompt)
python train.py --model-type yolov8 --results-folder experiment_2024

# Skip all interactive prompts (use defaults)
python train.py --model-type yolov8 --non-interactive

# Resume training from checkpoint
python train.py --model-type yolov8 --resume logs/previous_run/weights/last.pt

# Validate only (no training)
python train.py --model-type yolov8 --validate-only

# Export model after training
python train.py --model-type yolov8 --export
```

### Command Line Arguments
- `--model-type`: Choose between `yolo11`, `yolov8`, `yolov5`
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--image-size`: Input image size
- `--device`: Training device (`cpu`, `cuda`, `auto`)
- `--results-folder`: Custom folder name for results (skips interactive prompt)
- `--non-interactive`: Skip all interactive configuration prompts (use defaults)
- `--resume`: Path to checkpoint for resuming training
- `--validate-only`: Only validate, don't train
- `--export`: Export model after training

### Interactive Configuration
When you run training without `--non-interactive`, the system will prompt you for:

1. **YOLO Version**: Choose between YOLO11, YOLOv8, YOLOv5 (if not specified)
2. **Model Size**: Choose between n (nano), s (small), m (medium), l (large), x (xlarge)
3. **Training Duration**: Number of epochs (default: 100)
4. **Batch Size**: Training batch size (default: 8)
5. **Image Size**: Input resolution (default: 1024)
6. **Learning Rate**: Training learning rate (default: 0.01)
7. **Device**: GPU or CPU training (default: cuda if available)
8. **Advanced Options**: Early stopping patience, augmentation, validation frequency

**Pro Tips:**
- Press Enter to accept default values
- Use `--non-interactive` for automated scripts
- Combine with `--results-folder` to skip folder naming prompt

## TensorBoard Integration

### Automatic Training Monitoring
The system includes **automatic TensorBoard integration** that provides real-time training visualization:

#### During Training
- **Auto-launch**: TensorBoard opens automatically in your browser
- **Real-time metrics**: Live loss curves, accuracy plots, and training progress
- **Model visualization**: Network architecture and computational graphs
- **Persistent access**: TensorBoard remains running after training completes

#### After Training
- **Result analysis**: Continue viewing training metrics and model performance
- **Experiment comparison**: Compare different training runs and experiments
- **Easy management**: Simple commands to control TensorBoard sessions

#### TensorBoard Management Commands
```bash
# Check if TensorBoard is running and open in browser
python -m utils.tensorboard_manager status

# List all experiments with their TensorBoard data status
python -m utils.tensorboard_manager list

# Launch TensorBoard for a specific experiment
python -m utils.tensorboard_manager launch experiment_name

# Launch on custom port
python -m utils.tensorboard_manager launch experiment_name --port 6007

# Stop all TensorBoard processes
python -m utils.tensorboard_manager stop
```

#### TensorBoard Features
- **Training Metrics**: Loss curves (box, classification, DFL losses)
- **Validation Metrics**: mAP50, mAP50-95, precision, recall
- **Model Architecture**: Visual representation of YOLO network structure
- **Training Images**: Sample batches with augmentations and predictions
- **Hyperparameters**: Complete training configuration tracking
- **System Metrics**: GPU utilization, memory usage, training speed

#### Manual Access
If you need to manually access TensorBoard for any experiment:
```bash
# For current training
http://localhost:6006

# View experiment logs directly
tensorboard --logdir logs/experiment_name/experiment_name
```

## Custom Results Folder Feature
- When you run training, the system prompts for a custom folder name
- Results are organized in `logs/your_custom_name/` instead of the default `logs/yolo_training/`
- Each training run gets its own organized folder with weights, plots, and logs
- Folder names are automatically cleaned of invalid characters
- Existing folders can be reused or new names can be chosen

## Interactive Configuration Feature
- **Beginner-Friendly**: Step-by-step prompts for all major training parameters
- **Smart Defaults**: Press Enter to accept recommended values
- **Model Selection**: Choose from nano (n) to xlarge (x) model sizes
- **Parameter Guidance**: Helpful explanations for each setting
- **Validation**: Input validation with helpful error messages
- **Flexible**: Use `--non-interactive` to skip prompts for automation

## Robust System Architecture

The system is built with enterprise-grade reliability:

### Core Components
- **Configuration Management**: Centralized config with validation and environment overrides
- **Data Pipeline**: Robust dataset handling with automatic validation and preprocessing
- **Training Engine**: Comprehensive training loop with checkpoint management and monitoring
- **Evaluation System**: Multi-metric evaluation with visualization and reporting
- **Export Utilities**: Multi-format model export (ONNX, TorchScript, CoreML, TensorRT)

### Reliability Features
- **Comprehensive Testing**: 98 tests covering all major components
- **Error Handling**: Graceful failure handling with detailed logging
- **Data Validation**: Automatic dataset structure and format validation
- **Checkpoint Management**: Robust save/load with automatic cleanup
- **Real-time Monitoring**: Automatic TensorBoard integration with persistent access
- **Training Visualization**: Live metrics, loss curves, and model performance tracking

## Prerequisites

- **Python 3.8+** with pip
- **Roboflow account** with annotated dataset
- **API key** from Roboflow
- **GPU** (recommended for training)

## Installation

### Option 1: Full Installation
```bash
pip install -r requirements.txt
```

### Option 2: Minimal Installation
```bash
pip install ultralytics roboflow torch torchvision
```

### Option 3: GPU Support
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics roboflow
```

## Learning Path

### Beginner (Start Here)
1. **Read** [docs/README.md](docs/README.md) - Main documentation hub
2. **Try** [docs/workflow/01-system-overview/01-system-overview.md](docs/workflow/01-system-overview/01-system-overview.md) - System overview
3. **Follow** [docs/workflow/04-integration-workflows/01-training-workflows.md](docs/workflow/04-integration-workflows/01-training-workflows.md) - Training workflows
4. **Run** the example script
5. **Train** your first model with `python train.py`

### Intermediate
1. **Explore** other YOLO versions
2. **Customize** training parameters
3. **Experiment** with different architectures
4. **Optimize** for your use case

### Advanced
1. **Research** latest YOLO versions
2. **Contribute** to the community
3. **Deploy** models to production
4. **Optimize** for edge devices

## Common Issues

### Export Problems
- **Format not found**: Use `yolo` format as fallback
- **API key errors**: Verify environment variable is set
- **Permission denied**: Check Roboflow project access

### Training Problems
- **Memory errors**: Reduce batch size and image size
- **CUDA issues**: Verify PyTorch CUDA installation
- **Path errors**: Check `data.yaml` file paths

### Performance Issues
- **Slow training**: Use smaller model variants
- **Low accuracy**: Increase dataset size and quality
- **Overfitting**: Add more augmentation and regularization

## Testing & Quality Assurance

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_config.py -v
python -m pytest tests/test_data_loader.py -v
python -m pytest tests/test_training.py -v
```

## Demo and Examples

### Interactive Demo
```bash
python demo_interactive_training.py
```
Shows all available training modes and options

### Quick Examples
```bash
# Interactive training with YOLO version selection
python train.py

# Non-interactive training with defaults
python train.py --model-type yolov8 --non-interactive --results-folder quick_test

# Custom configuration
python train.py --model-type yolov8 --epochs 50 --batch-size 4 --image-size 640
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the docs/ folder for detailed guides
- **Examples**: See the examples/ folder for practical usage

## Acknowledgments

- Ultralytics team for YOLOv8 implementation
- Roboflow for dataset management tools
- PyTorch community for deep learning framework
- Contributors and users of this project
