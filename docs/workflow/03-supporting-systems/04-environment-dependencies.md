# Environment & Dependencies

## What This System Needs

The environment and dependencies system is like the foundation of a house - it provides everything needed to run the YOLO training system. Think of it as a checklist that ensures your computer has all the right tools, libraries, and settings to train models successfully.

## System Requirements Overview

### **Hardware Requirements**

#### **Minimum Requirements**
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for basic training
- **GPU**: Optional but highly recommended for faster training

#### **Recommended Requirements**
- **CPU**: Intel i7/AMD Ryzen 7 or better
- **RAM**: 32GB or more for large datasets
- **Storage**: 50GB+ free space for extensive training
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)

#### **GPU Requirements**
- **NVIDIA**: GTX 1060 6GB or better (CUDA 11.8+)
- **AMD**: RX 580 8GB or better (ROCm support)
- **Apple**: M1/M2 chip with Metal Performance Shaders

### **Software Requirements**

#### **Operating System**
- **Linux**: Ubuntu 20.04+, CentOS 8+, Debian 11+
- **Windows**: Windows 10/11 (64-bit)
- **macOS**: macOS 10.15+ (Catalina or later)

#### **Python Version**
- **Required**: Python 3.8 or higher
- **Recommended**: Python 3.9 or 3.10
- **Latest**: Python 3.11+ (with compatibility notes)

## Core Dependencies

### **Primary Dependencies**

The system requires these essential packages to function:

```txt
# Core dependencies
torch>=2.0.0              # PyTorch deep learning framework
torchvision>=0.15.0       # Computer vision utilities
ultralytics>=8.0.0        # YOLO training and inference
numpy>=1.21.0             # Numerical computing
Pillow>=9.0.0             # Image processing
PyYAML>=6.0               # Configuration file handling
opencv-python>=4.8.0      # Computer vision operations
```

#### **PyTorch (torch)**
- **Purpose**: Core deep learning framework
- **Version**: 2.0.0 or higher required
- **Installation**: `pip install torch torchvision`
- **GPU Support**: CUDA 11.8+ for NVIDIA GPUs

#### **Ultralytics**
- **Purpose**: YOLO model training and management
- **Version**: 8.0.0 or higher required
- **Installation**: `pip install ultralytics`
- **Features**: YOLOv5, YOLOv8, YOLO11 support

#### **OpenCV (opencv-python)**
- **Purpose**: Image processing and computer vision
- **Version**: 4.8.0 or higher required
- **Installation**: `pip install opencv-python`
- **Features**: Image loading, preprocessing, augmentation

### **Data Handling Dependencies**

```txt
# Data handling and evaluation
scikit-learn>=1.3.0       # Machine learning utilities
matplotlib>=3.7.0          # Plotting and visualization
tqdm>=4.65.0               # Progress bars
```

#### **Scikit-learn**
- **Purpose**: Data preprocessing and evaluation metrics
- **Version**: 1.3.0 or higher required
- **Installation**: `pip install scikit-learn`
- **Features**: Data splitting, metrics calculation

#### **Matplotlib**
- **Purpose**: Training visualization and plotting
- **Version**: 3.7.0 or higher required
- **Installation**: `pip install matplotlib`
- **Features**: Training curves, confusion matrices

### **Logging and Monitoring**

```txt
# Logging and monitoring
tensorboard>=2.13.0        # Training visualization
wandb>=0.15.0              # Experiment tracking
```

#### **TensorBoard**
- **Purpose**: Real-time training monitoring
- **Version**: 2.13.0 or higher required
- **Installation**: `pip install tensorboard`
- **Features**: Training curves, model graphs, hyperparameters

#### **Weights & Biases (wandb)**
- **Purpose**: Experiment tracking and collaboration
- **Version**: 0.15.0 or higher required
- **Installation**: `pip install wandb`
- **Features**: Cloud-based experiment tracking

### **Export Format Dependencies**

```txt
# Export formats
onnx>=1.14.0               # ONNX model format
onnxsim>=0.4.0             # ONNX model optimization

# Optional dependencies for advanced export
# coremltools>=7.0         # CoreML export (Apple)
# tensorrt>=8.6.0          # TensorRT export (NVIDIA)
```

#### **ONNX (Open Neural Network Exchange)**
- **Purpose**: Cross-platform model deployment
- **Version**: 1.14.0 or higher required
- **Installation**: `pip install onnx onnxsim`
- **Features**: Model conversion, optimization

#### **CoreML Tools**
- **Purpose**: Apple device deployment
- **Version**: 7.0 or higher (optional)
- **Installation**: `pip install coremltools`
- **Features**: iOS/macOS model conversion

#### **TensorRT**
- **Purpose**: NVIDIA GPU optimization
- **Version**: 8.6.0 or higher (optional)
- **Installation**: Complex - requires NVIDIA account
- **Features**: Maximum GPU performance

## Testing Dependencies

### **Core Testing Framework**

```txt
# Core testing framework
pytest>=7.0.0              # Testing framework
pytest-cov>=4.0.0          # Coverage reporting
pytest-mock>=3.10.0        # Mocking utilities
pytest-xdist>=3.0.0        # Parallel test execution
```

#### **Pytest**
- **Purpose**: Main testing framework
- **Version**: 7.0.0 or higher required
- **Installation**: `pip install pytest`
- **Features**: Test discovery, execution, reporting

#### **Pytest-cov**
- **Purpose**: Code coverage measurement
- **Version**: 4.0.0 or higher required
- **Installation**: `pip install pytest-cov`
- **Features**: Coverage reports, HTML output

### **Code Quality Tools**

```txt
# Code quality and linting
flake8>=5.0.0              # Code style checking
black>=22.0.0               # Code formatting
isort>=5.10.0               # Import sorting
mypy>=1.0.0                # Type checking
```

#### **Black**
- **Purpose**: Automatic code formatting
- **Version**: 22.0.0 or higher required
- **Installation**: `pip install black`
- **Features**: PEP 8 compliance, consistent style

#### **Flake8**
- **Purpose**: Code style and error detection
- **Version**: 5.0.0 or higher required
- **Installation**: `pip install flake8`
- **Features**: Style checking, error detection

### **Performance Testing**

```txt
# Performance testing
pytest-benchmark>=4.0.0    # Performance benchmarking
pytest-profiling>=1.7.0    # Performance profiling
pytest-memray>=0.3.0       # Memory usage tracking
```

#### **Pytest-benchmark**
- **Purpose**: Performance measurement
- **Version**: 4.0.0 or higher required
- **Installation**: `pip install pytest-benchmark`
- **Features**: Timing measurements, statistical analysis

## Environment Configuration

### **Environment Variables**

The system uses environment variables for configuration:

```bash
# Required: Roboflow Configuration
ROBOFLOW_API_KEY=your_roboflow_api_key_here
ROBOFLOW_WORKSPACE=your_workspace_name
ROBOFLOW_PROJECT_ID=your_project_id
ROBOFLOW_VERSION=1

# Optional: Training Configuration
DEVICE=auto                    # auto, cuda, cpu
NUM_WORKERS=8                  # Data loading workers
MAX_GPU_MEMORY=0              # GPU memory limit (GB)
RANDOM_SEED=42                # Reproducibility

# Optional: Logging and Monitoring
ENABLE_WANDB=false            # Weights & Biases
ENABLE_TENSORBOARD=true       # TensorBoard logging
LOG_LEVEL=INFO                # Logging level
```

### **Configuration File (.env)**

Create a `.env` file from the template:

```bash
# Copy environment template
cp env.example .env

# Edit with your values
nano .env
```

#### **Required Configuration**
```bash
# Roboflow API key (required for dataset export)
ROBOFLOW_API_KEY=your_actual_api_key_here

# Workspace and project information
ROBOFLOW_WORKSPACE=your_username
ROBOFLOW_PROJECT_ID=your_project_id
```

#### **Optional Configuration**
```bash
# Training optimization
ENABLE_AMP=true               # Mixed precision training
GRADIENT_CLIP_NORM=10.0      # Gradient clipping
WARMUP_EPOCHS=3              # Learning rate warmup

# Export settings
ENABLE_ONNX_EXPORT=true      # ONNX export
ENABLE_TENSORRT_EXPORT=false # TensorRT export
ENABLE_COREML_EXPORT=false   # CoreML export
```

## Installation Methods

### **Method 1: Complete Installation (Recommended)**

Install all dependencies at once:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install all dependencies
pip install -r requirements.txt
```

### **Method 2: Core Installation Only**

Install only essential dependencies:

```bash
# Install core packages
pip install torch torchvision ultralytics
pip install numpy Pillow PyYAML opencv-python
pip install scikit-learn matplotlib tqdm
```

### **Method 3: Development Installation**

Install with testing and development tools:

```bash
# Install core dependencies
pip install -r requirements.txt

# Install testing dependencies
pip install -r requirements-test.txt

# Install development tools
pip install black flake8 isort mypy
```

### **Method 4: GPU-Specific Installation**

Install with GPU support:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install TensorRT (optional)
pip install tensorrt
```

## Virtual Environment Management

### **Creating Virtual Environment**

```bash
# Create new virtual environment
python -m venv .venv

# Create with specific Python version
python3.9 -m venv .venv

# Create with specific name
python -m venv my_yolo_env
```

### **Activating Virtual Environment**

```bash
# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# PowerShell
.venv\Scripts\Activate.ps1
```

### **Managing Dependencies**

```bash
# Install package
pip install package_name

# Install specific version
pip install package_name==1.2.3

# Install from requirements
pip install -r requirements.txt

# Upgrade package
pip install --upgrade package_name

# Uninstall package
pip uninstall package_name
```

### **Environment Export**

```bash
# Export current environment
pip freeze > requirements-current.txt

# Export with specific packages
pip freeze | grep -E "(torch|ultralytics|numpy)" > core-requirements.txt
```

## System Compatibility

### **Operating System Support**

#### **Linux (Ubuntu/Debian)**
```bash
# Update package manager
sudo apt update
sudo apt upgrade

# Install system dependencies
sudo apt install python3 python3-pip python3-venv
sudo apt install build-essential cmake
sudo apt install libgl1-mesa-glx libglib2.0-0

# Install CUDA (if using NVIDIA GPU)
sudo apt install nvidia-cuda-toolkit
```

#### **Windows**
```bash
# Install Python from python.org
# Download and install Python 3.9+ for Windows

# Install Visual Studio Build Tools
# Required for some Python packages

# Use Windows Subsystem for Linux (WSL) for better compatibility
```

#### **macOS**
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Install Xcode Command Line Tools
xcode-select --install
```

### **GPU Support**

#### **NVIDIA CUDA**
```bash
# Check CUDA installation
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

#### **AMD ROCm**
```bash
# Install ROCm (Linux only)
sudo apt install rocm-hip-sdk

# Install PyTorch with ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
```

#### **Apple Metal**
```bash
# macOS with Apple Silicon (M1/M2)
# PyTorch automatically uses Metal Performance Shaders

# Verify Metal support
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Dependency Management

### **Version Pinning**

#### **Exact Versions**
```txt
# Pin to exact versions
torch==2.0.1
ultralytics==8.0.196
numpy==1.24.3
```

#### **Version Ranges**
```txt
# Allow compatible updates
torch>=2.0.0,<3.0.0
ultralytics>=8.0.0,<9.0.0
numpy>=1.21.0,<2.0.0
```

#### **Flexible Versions**
```txt
# Allow any compatible version
torch>=2.0.0
ultralytics>=8.0.0
numpy>=1.21.0
```

### **Dependency Conflicts**

#### **Common Conflicts**
```bash
# PyTorch version conflicts
pip install torch==2.0.1
pip install torchvision==0.15.2

# OpenCV conflicts
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

#### **Resolution Strategies**
```bash
# Check for conflicts
pip check

# Install compatible versions
pip install --upgrade --force-reinstall package_name

# Use conda for complex dependencies
conda install pytorch torchvision -c pytorch
```

## Troubleshooting Common Issues

### **Installation Problems**

#### **"Permission Denied" Errors**
```bash
# Use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate
pip install package_name

# Or use user installation
pip install --user package_name
```

#### **"Build Failed" Errors**
```bash
# Install build tools
sudo apt install build-essential  # Linux
# Install Visual Studio Build Tools  # Windows

# Use pre-built wheels
pip install --only-binary=all package_name
```

#### **"Version Conflict" Errors**
```bash
# Check current versions
pip list | grep package_name

# Uninstall conflicting packages
pip uninstall package_name

# Install specific version
pip install package_name==version
```

### **Runtime Problems**

#### **"CUDA Not Available" Errors**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### **"Out of Memory" Errors**
```bash
# Reduce batch size
python train.py --batch-size 8

# Use CPU training
python train.py --device cpu

# Limit GPU memory
export CUDA_VISIBLE_DEVICES=0
export CUDA_MEM_FRACTION=0.8
```

#### **"Import Error" Problems**
```bash
# Check virtual environment
which python
pip list

# Reinstall package
pip uninstall package_name
pip install package_name

# Check Python path
python -c "import sys; print(sys.path)"
```

## Performance Optimization

### **System-Level Optimization**

#### **Linux Optimization**
```bash
# Set CPU governor to performance
sudo cpupower frequency-set -g performance

# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize swap usage
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

#### **Windows Optimization**
```bash
# Disable Windows Defender real-time protection (temporarily)
# Set power plan to High Performance
# Disable unnecessary startup programs
```

#### **macOS Optimization**
```bash
# Disable App Nap for terminal applications
# Set energy saver to prevent sleep
# Close unnecessary applications
```

### **Python-Level Optimization**

#### **Environment Variables**
```bash
# Optimize PyTorch
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Optimize NumPy
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Optimize OpenCV
export OPENCV_OPENCL_RUNTIME=1
```

#### **Package Optimization**
```bash
# Install optimized packages
pip install numpy --upgrade
pip install opencv-python-headless  # Lighter OpenCV

# Use conda for scientific packages
conda install numpy scipy scikit-learn
```

## Security Considerations

### **API Key Security**

#### **Environment Variables**
```bash
# Never hardcode API keys
# Use environment variables
export ROBOFLOW_API_KEY="your_key_here"

# Or use .env file (never commit to git)
echo "ROBOFLOW_API_KEY=your_key_here" > .env
echo ".env" >> .gitignore
```

#### **Secure Storage**
```bash
# Use keyring for secure storage
pip install keyring
python -c "import keyring; keyring.set_password('roboflow', 'username', 'api_key')"

# Or use system keychain
# macOS: Keychain Access
# Linux: GNOME Keyring
# Windows: Credential Manager
```

### **Package Security**

#### **Vulnerability Scanning**
```bash
# Check for known vulnerabilities
pip install safety
safety check

# Update vulnerable packages
pip install --upgrade package_name
```

#### **Source Verification**
```bash
# Install from trusted sources
pip install package_name --index-url https://pypi.org/simple/

# Verify package signatures
pip install package_name --require-hashes
```

## Maintenance and Updates

### **Regular Updates**

#### **Update Dependencies**
```bash
# Check for updates
pip list --outdated

# Update all packages
pip install --upgrade -r requirements.txt

# Update specific packages
pip install --upgrade torch ultralytics
```

#### **Update Python**
```bash
# Check Python version
python --version

# Update Python (system-dependent)
# Linux: Use package manager or pyenv
# Windows: Download from python.org
# macOS: Use Homebrew or pyenv
```

### **Cleanup and Maintenance**

#### **Remove Unused Packages**
```bash
# List installed packages
pip list

# Remove unused packages
pip uninstall package_name

# Clean pip cache
pip cache purge
```

#### **Environment Cleanup**
```bash
# Remove virtual environment
rm -rf .venv

# Create fresh environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

**Next**: We'll move to [Phase 4: Integration & Workflows](04-integration-workflows/01-training-workflows.md) to understand how all components work together in complete training workflows.
