# Export & Model Management

## What This System Does

The export and model management system is like a model conversion factory that takes your trained YOLO models and transforms them into formats that can be used in production environments. Think of it as having a team of specialists who know exactly how to optimize your model for different platforms, from mobile phones to cloud servers.

## Model Export System Overview

### **What Gets Exported**
- **Trained YOLO Models**: Your .pt files from training
- **Multiple Formats**: ONNX, TorchScript, OpenVINO, CoreML, TensorRT
- **Optimized Versions**: Hardware-specific optimizations
- **Deployment Ready**: Models ready for production use

### **Why Export Models**
- **Platform Compatibility**: Use models on different devices
- **Performance Optimization**: Hardware-specific acceleration
- **Production Deployment**: Ready for real-world applications
- **Integration**: Easy to use in existing systems

## Export Formats and Use Cases

### **1. ONNX Format (.onnx)**

#### **What It Is**
ONNX (Open Neural Network Exchange) is a universal format that works across different frameworks and platforms.

#### **Use Cases**
- **Cross-platform deployment**: Windows, Linux, macOS
- **Web services**: REST APIs, microservices
- **Cloud inference**: AWS, Google Cloud, Azure
- **Framework integration**: TensorFlow, PyTorch, Caffe2

#### **Benefits**
- **Hardware agnostic**: Works on any platform
- **Wide support**: Many frameworks and tools
- **Optimized runtime**: ONNX Runtime for performance
- **Standard format**: Industry standard for model exchange

#### **Example Usage**
```python
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
result = session.run([output_name], {input_name: input_data})
```

### **2. TorchScript Format (.pt)**

#### **What It Is**
TorchScript is PyTorch's production-ready format that optimizes models for deployment.

#### **Use Cases**
- **PyTorch production**: Native PyTorch applications
- **C++ integration**: Embed models in C++ applications
- **TorchServe**: Production model serving
- **Mobile deployment**: Android/iOS with PyTorch Mobile

#### **Benefits**
- **Native performance**: Optimized PyTorch execution
- **Easy integration**: Seamless PyTorch workflow
- **Production ready**: Optimized for deployment
- **C++ support**: Can be used in C++ applications

#### **Example Usage**
```python
import torch

# Load TorchScript model
model = torch.jit.load("model_torchscript.pt")

# Run inference
with torch.no_grad():
    result = model(input_tensor)
```

### **3. OpenVINO Format (.xml + .bin)**

#### **What It Is**
OpenVINO is Intel's toolkit for optimizing deep learning models on Intel hardware.

#### **Use Cases**
- **Intel CPU optimization**: Maximum CPU performance
- **Edge devices**: IoT, embedded systems
- **Intel GPUs**: Integrated and discrete graphics
- **Server deployment**: Intel-based servers

#### **Benefits**
- **CPU acceleration**: Optimized for Intel processors
- **Hardware specific**: Intel optimization
- **Edge deployment**: Lightweight for IoT devices
- **Open source**: Free Intel toolkit

#### **Example Usage**
```python
from openvino.runtime import Core

# Load OpenVINO model
ie = Core()
model = ie.read_model("model.xml")
compiled_model = ie.compile_model(model)

# Run inference
result = compiled_model(input_data)
```

### **4. CoreML Format (.mlpackage)**

#### **What It Is**
CoreML is Apple's framework for deploying machine learning models on Apple devices.

#### **Use Cases**
- **iOS applications**: iPhone and iPad apps
- **macOS applications**: Desktop applications
- **watchOS**: Apple Watch apps
- **Apple Silicon**: M1/M2 chip optimization

#### **Benefits**
- **Native Apple performance**: Optimized for Apple hardware
- **Easy integration**: Xcode and Swift/Objective-C
- **Privacy focused**: On-device inference
- **Automatic optimization**: Apple handles optimization

#### **Example Usage**
```swift
import CoreML

// Load CoreML model
guard let model = try? YOLOModel() else { return }

// Run inference
let prediction = try model.prediction(input: input_data)
```

### **5. TensorRT Format (.engine)**

#### **What It Is**
TensorRT is NVIDIA's high-performance deep learning inference library.

#### **Use Cases**
- **NVIDIA GPU acceleration**: Maximum GPU performance
- **Real-time applications**: Video processing, gaming
- **High-throughput**: Batch processing, server deployment
- **Jetson devices**: NVIDIA edge devices

#### **Benefits**
- **Maximum GPU performance**: Optimized for NVIDIA GPUs
- **Real-time inference**: Ultra-low latency
- **High throughput**: Process many images quickly
- **Hardware specific**: NVIDIA optimization

#### **Example Usage**
```python
import tensorrt as trt

# Load TensorRT engine
with open("model.engine", "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

# Run inference
context = engine.create_execution_context()
# ... inference code
```

## Export Process and Workflow

### **Automatic Export During Training**

When you train with the `--export` flag, models are automatically exported:

```bash
# Train and export automatically
python train.py --model-type yolov8 --export

# This will:
# 1. Train your model
# 2. Export to all supported formats
# 3. Save in organized folder structure
```

### **Manual Export of Existing Models**

Export already-trained models using the export script:

```bash
# Export existing models
python utils/export_existing_models.py

# This will:
# 1. Find all .pt files in your project
# 2. Export each to multiple formats
# 3. Organize by model name
```

### **Export Workflow Steps**

#### **Step 1: Model Loading**
```python
def export_model_to_formats(model_path: Path, export_dir: Path):
    # Load the trained model
    model = YOLO(str(model_path))
    
    # Create export directory
    model_name = model_path.stem
    model_export_dir = export_dir / model_name
    model_export_dir.mkdir(parents=True, exist_ok=True)
```

#### **Step 2: Format Conversion**
```python
# Export to ONNX
onnx_path = model_export_dir / f"{model_name}.onnx"
model.export(format="onnx", simplify=True, half=False)

# Export to TorchScript
torchscript_path = model_export_dir / f"{model_name}_torchscript.pt"
model.export(format="torchscript", half=False)

# Export to OpenVINO
openvino_path = model_export_dir / f"{model_name}_openvino"
model.export(format="openvino", half=False)
```

#### **Step 3: Organization and Validation**
```python
# Move exported files to organized structure
if onnx_files:
    shutil.move(onnx_files[0], onnx_path)
            print(f"ONNX exported: {onnx_path}")

# Validate exports
validate_exported_model(onnx_path)
```

## Export Configuration and Options

### **Export Settings**

The system provides configurable export options:

```python
def export_all_formats(
    self,
    formats: Optional[List[str]] = None,
    include_nms: bool = True,
    half_precision: bool = True,
    int8_quantization: bool = False,
    simplify: bool = True,
    dynamic: bool = False,
) -> Dict[str, Path]:
```

#### **Format Selection**
- **formats**: Choose which formats to export
- **include_nms**: Include Non-Maximum Suppression in model
- **half_precision**: Use FP16 for smaller file sizes
- **int8_quantization**: Use INT8 for maximum compression
- **simplify**: Optimize ONNX models
- **dynamic**: Support variable batch sizes

### **Hardware-Specific Optimization**

#### **CPU Optimization**
```python
# CPU-optimized export
model.export(format="onnx", half=False)  # Use FP32 for CPU
model.export(format="openvino", half=False)  # Intel optimization
```

#### **GPU Optimization**
```python
# GPU-optimized export
if torch.cuda.is_available():
    model.export(format="engine", half=True)  # TensorRT with FP16
    model.export(format="onnx", half=True)    # ONNX with FP16
```

#### **Mobile Optimization**
```python
# Mobile-optimized export
model.export(format="coreml", half=True)      # Apple optimization
model.export(format="onnx", simplify=True)    # Simplified ONNX
```

## Model Organization and Management

### **Export Directory Structure**

Exported models are organized in a clear, logical structure:

```
exported_models/
├── yolo11n/                    # YOLO11 nano model
│   ├── yolo11n.onnx           # ONNX format
│   ├── yolo11n_torchscript.pt # TorchScript format
│   ├── yolo11n_openvino/      # OpenVINO format
│   │   ├── yolo11n.xml        # Model definition
│   │   ├── yolo11n.bin        # Model weights
│   │   └── metadata.yaml      # Model metadata
│   ├── yolo11n.mlpackage      # CoreML format
│   └── export_metadata.json   # Export information
├── yolov8n/                    # YOLOv8 nano model
│   ├── yolov8n.onnx           # ONNX format
│   ├── yolov8n_torchscript.pt # TorchScript format
│   ├── yolov8n_openvino/      # OpenVINO format
│   ├── yolov8n.mlpackage      # CoreML format
│   └── export_metadata.json   # Export information
└── README.md                   # Usage instructions
```

### **Export Metadata**

Each export includes detailed metadata:

```json
{
  "export_info": {
    "model_name": "yolo11n",
    "export_date": "2024-01-15T10:30:00",
    "original_model": "logs/training_run/weights/best.pt",
    "export_formats": ["onnx", "torchscript", "openvino", "coreml"]
  },
  "model_info": {
    "model_type": "yolo11",
    "input_shape": [1, 3, 640, 640],
    "output_shape": [1, 25200, 85],
    "num_classes": 80
  },
  "export_settings": {
    "include_nms": true,
    "half_precision": true,
    "simplify": true,
    "dynamic": false
  }
}
```

## Performance Optimization

### **Model Size Optimization**

#### **Precision Reduction**
```python
# FP16 (half precision) - 2x smaller
model.export(format="onnx", half=True)

# INT8 quantization - 4x smaller
model.export(format="onnx", int8=True)
```

#### **Model Simplification**
```python
# Simplify ONNX model
if simplify:
    try:
        self._simplify_onnx(onnx_path)
    except Exception as e:
        logger.warning(f"Failed to simplify ONNX model: {e}")
```

### **Inference Speed Optimization**

#### **Batch Processing**
```python
# Support dynamic batch sizes
dynamic_axes = {
    "images": {0: "batch_size"},
    "output": {0: "batch_size"}
} if dynamic else None
```

#### **Hardware Acceleration**
```python
# GPU acceleration
if torch.cuda.is_available():
    model.export(format="engine")  # TensorRT for NVIDIA
    
# CPU acceleration
model.export(format="openvino")   # OpenVINO for Intel
```

## Quality Assurance and Validation

### **Export Validation**

Each exported model is validated for correctness:

```python
def _validate_onnx(self, onnx_path: Path):
    """Validate ONNX model."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.info(f"ONNX model validation passed: {onnx_path}")
    except Exception as e:
        logger.error(f"ONNX model validation failed: {e}")
```

### **Performance Testing**

Exported models are tested for performance:

```python
def test_exported_model_performance(model_path: Path):
    """Test exported model performance."""
    # Load model
    model = load_exported_model(model_path)
    
    # Test inference speed
    start_time = time.time()
    for _ in range(100):
        result = model.infer(test_input)
    avg_time = (time.time() - start_time) / 100
    
    # Verify performance requirements
    assert avg_time < 0.1  # Should be faster than 100ms
```

## Integration and Deployment

### **Production Deployment**

#### **Web Services**
```python
# Flask API with ONNX model
from flask import Flask, request
import onnxruntime as ort

app = Flask(__name__)
session = ort.InferenceSession("model.onnx")

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json['image']
    result = session.run(['output'], {'input': input_data})
    return {'prediction': result[0].tolist()}
```

#### **Mobile Applications**
```swift
// iOS app with CoreML model
import CoreML

class YOLOPredictor {
    let model: YOLOModel
    
    init() throws {
        self.model = try YOLOModel()
    }
    
    func predict(image: CVPixelBuffer) throws -> YOLOModelOutput {
        return try model.prediction(input: image)
    }
}
```

#### **Edge Devices**
```python
# Raspberry Pi with OpenVINO
from openvino.runtime import Core

ie = Core()
model = ie.read_model("model.xml")
compiled_model = ie.compile_model(model)

def predict_edge(image):
    result = compiled_model(image)
    return process_results(result)
```

### **Cloud Deployment**

#### **AWS Lambda**
```python
# Lambda function with ONNX model
import onnxruntime as ort
import json

def lambda_handler(event, context):
    # Load model (cached between invocations)
    session = ort.InferenceSession("model.onnx")
    
    # Process input
    input_data = event['image']
    result = session.run(['output'], {'input': input_data})
    
    return {
        'statusCode': 200,
        'body': json.dumps({'predictions': result[0].tolist()})
    }
```

#### **Google Cloud Functions**
```python
# Cloud Function with TensorRT model
import tensorrt as trt

def predict_cloud(request):
    # Load TensorRT engine
    engine = load_tensorrt_engine("model.engine")
    
    # Process request
    input_data = request.get_json()['image']
    result = run_inference(engine, input_data)
    
    return {'predictions': result}
```

## Troubleshooting Export Issues

### **Common Export Problems**

#### **"Export failed" Errors**
- **Problem**: Export process failed
- **Solution**: Check model compatibility, try different formats
- **Alternative**: Use manual export with specific settings

#### **"Format not supported" Errors**
- **Problem**: Requested format not available
- **Solution**: Check format support, install required dependencies
- **Alternative**: Use supported formats only

#### **"Performance degradation" Issues**
- **Problem**: Exported model is slower than original
- **Solution**: Check optimization settings, use hardware-specific formats
- **Alternative**: Profile and optimize export settings

### **Getting Help with Exports**

#### **Check Export Logs**
```bash
# Run export with verbose output
python utils/export_existing_models.py --verbose

# Check specific format export
python -c "from utils.export_utils import YOLOExporter; print(YOLOExporter.supported_formats)"
```

#### **Validate Exported Models**
```python
# Test exported model functionality
def test_exported_model(model_path):
    try:
        model = load_exported_model(model_path)
        result = model.infer(test_input)
        print(f"Model {model_path} works correctly")
        return True
    except Exception as e:
        print(f"Model {model_path} failed: {e}")
        return False
```

## Best Practices for Model Export

### **For Beginners**
1. **Start simple**: Export to ONNX first (most compatible)
2. **Use defaults**: Let system choose optimal settings
3. **Test exports**: Verify models work before deployment
4. **Keep organized**: Use organized folder structure

### **For Intermediate Users**
1. **Choose formats wisely**: Select based on deployment target
2. **Optimize settings**: Adjust precision and optimization
3. **Validate performance**: Test speed and accuracy
4. **Document exports**: Keep track of export settings

### **For Advanced Users**
1. **Custom optimization**: Fine-tune for specific hardware
2. **Batch processing**: Optimize for production workloads
3. **Integration testing**: Test in deployment environment
4. **Performance profiling**: Measure and optimize inference

---

**Next**: We'll explore the [Environment & Dependencies](04-environment-dependencies.md) system to understand setup requirements and system dependencies.
