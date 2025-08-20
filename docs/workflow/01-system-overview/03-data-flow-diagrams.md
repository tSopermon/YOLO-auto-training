# Data Flow Diagrams

## How Data Moves Through the System

This document shows you exactly how data flows from the moment you place your dataset in the folder until you have a trained model. Think of it as following a package through a delivery system - we'll trace every step.

## Main Data Flow Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│   INPUT     │───▶│  PROCESSING  │───▶│   OUTPUT    │───▶│   RESULTS   │
│             │    │              │    │             │    │             │
│ Your Dataset│    │ Auto Prep    │    │ Training    │    │ Trained     │
│             │    │ + Config     │    │ Execution   │    │ Model       │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
```

## Detailed Data Flow Steps

### **Step 1: Dataset Input and Detection**

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATASET INPUT                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AUTO-DETECTION SYSTEM                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐    │
│  │ Structure   │ │ Format      │ │ Content                 │    │
│  │ Detection   │ │ Detection   │ │ Analysis                │    │
│  │             │ │             │ │                         │    │
│  • Flat folders│ • YOLO (.txt)   │ • Image count           │    │
│  • Nested      │ • COCO (.json)  │ • Label count           │    │
│  • Mixed       │ • XML (.xml)    │ • Class names           │    │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION RESULTS                            │
│  • Structure type identified                                    │
│  • Format type identified                                       │
│  • Issues found (if any)                                        │
│  • Action plan created                                          │
└─────────────────────────────────────────────────────────────────┘
```

**What happens in this step:**
1. **Input**: You place dataset in `dataset/` folder
2. **Structure Analysis**: System examines folder organization
3. **Format Detection**: Identifies annotation format (YOLO, COCO, XML)
4. **Content Analysis**: Counts images, labels, classes
5. **Issue Detection**: Finds problems that need fixing
6. **Planning**: Creates roadmap for preparation

### **Step 2: Dataset Preparation and Conversion**

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREPARATION PLANNING                         │
│  Based on detection results, system decides what to do:         │
│  • Convert format if needed                                     │
│  • Reorganize structure if needed                               │
│  • Fix issues found                                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FORMAT CONVERSION                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐    │
│  │ COCO to     │ │ XML to      │ │ Custom to               │    │
│  │ YOLO        │ │ YOLO        │ │ YOLO                    │    │
│  │             │ │             │ │                         │    │
│  • JSON parse  │ • XML parse   │ • Format detection        │    │
│  • Coord conv  │ • Coord conv  │ • Custom parser           │    │
│  • Label map   │ • Label map   │ • Validation              │    │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STRUCTURE REORGANIZATION                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐    │
│  │ Create      │ │ Organize    │ │ Generate                │    │
│  │ Folders     │ │ Images      │ │ Config                  │    │
│  │             │ │ & Labels    │ │                         │    │
│  • train/      │ • Copy files  │ • data.yaml               │    │
│  • valid/      │ • Rename      │ • Class names             │    │
│  • test/       │ • Validate    │ • Paths                   │    │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPARED DATASET                             │
│  • All images in YOLO format                                    │
│  • All labels in YOLO format                                    │
│  • Proper train/valid/test structure                            │
│  • data.yaml configuration file                                 │
│  • Ready for training                                           │
└─────────────────────────────────────────────────────────────────┘
```

**What happens in this step:**
1. **Planning**: System creates action plan based on detection
2. **Conversion**: Changes annotation format to YOLO standard
3. **Reorganization**: Creates proper folder structure
4. **Configuration**: Generates `data.yaml` file
5. **Validation**: Ensures everything is correct

### **Step 3: Model Selection and Loading**

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER MODEL CHOICE                            │
│  • YOLO11 (latest)                                              │
│  • YOLOv8 (stable)                                              │
│  • YOLOv5 (classic)                                             │
│  • Other versions                                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL LOADING SYSTEM                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐    │
│  │ Architecture│ │ Pre-trained │ │ Model                   │    │
│  │ Selection   │ │ Weights     │ │ Preparation             │    │
│  │             │ │             │ │                         │    │
│  • Load model  │ • Download    │ • Set device              │    │
│  • Set config  │ • Cache       │ • Configure               │    │
│  • Validate    │ • Verify      │ • Ready flag              │    │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    READY MODEL                                  │
│  • Model architecture loaded                                    │
│  • Pre-trained weights applied                                  │
│  • Device configured (CPU/GPU)                                  │
│  • Ready for training                                           │
└─────────────────────────────────────────────────────────────────┘
```

**What happens in this step:**
1. **User Choice**: You select which YOLO version to use
2. **Architecture Loading**: System loads the right model structure
3. **Weight Download**: Gets pre-trained weights if needed
4. **Configuration**: Sets up model for training
5. **Validation**: Ensures model is ready

### **Step 4: Training Configuration**

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER PARAMETER INPUT                         │
│  • Training epochs                                              │
│  • Batch size                                                   │
│  • Image size                                                   │
│  • Device selection                                             │
│  • Results folder name                                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION VALIDATION                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐    │
│  │ Parameter   │ │ Resource    │ │ Path                    │    │
│  │ Validation  │ │ Check       │ │ Creation                │    │
│  │             │ │             │ │                         │    │
│  • Valid range │ • GPU memory  │ • Create logs folder      │    │
│  • Type check  │ • CPU cores   │ • Set up monitoring       │    │
│  • Defaults    │ • Disk space  │ • Initialize logging      │    │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING CONFIG                              │
│  • All parameters validated                                     │
│  • Resources checked                                            │
│  • Logging system ready                                         │
│  • Ready to start training                                      │
└─────────────────────────────────────────────────────────────────┘
```

**What happens in this step:**
1. **User Input**: You provide training parameters
2. **Validation**: System checks if parameters make sense
3. **Resource Check**: Ensures you have enough resources
4. **Setup**: Creates logging and monitoring systems
5. **Ready**: Everything configured for training

### **Step 5: Training Execution**

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING INITIALIZATION                      │
│  • Data loaders created                                         │
│  • Model moved to device                                        │
│  • Optimizer configured                                         │
│  • Loss function set                                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐    │
│  │ Data        │ │ Forward     │ │ Backward                │    │
│  │ Loading     │ │ Pass        │ │ Pass                    │    │
│  │             │ │             │ │                         │    │
│  • Load batch  │ • Model       │ • Calculate               │    │
│  • Preprocess  │ • prediction  │ • gradients               │    │
│  • To device   │ • Loss calc   │ • Update                  │    │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROGRESS MONITORING                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐    │
│  │ Metrics     │ │ Checkpoint  │ │ Logging                 │    │
│  │ Tracking    │ │ Saving      │ │                         │    │
│  │             │ │             │ │                         │    │
│  • Loss values │ • Save model  │ • Console output          │    │
│  • Accuracy    │ • Save state  │ • File logging            │    │
│  • Learning    │ • Resume      │ • Progress bars           │    │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING COMPLETE                            │
│  • All epochs finished                                          │
│  • Best model saved                                             │
│  • Training metrics recorded                                    │
│  • Ready for evaluation                                         │
└─────────────────────────────────────────────────────────────────┘
```

**What happens in this step:**
1. **Initialization**: Sets up training environment
2. **Training Loop**: Repeats training for each epoch
3. **Monitoring**: Tracks progress and saves checkpoints
4. **Completion**: Finishes training and saves results

### **Step 6: Results and Export**

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL EVALUATION                             │
│  • Test on validation set                                       │
│  • Calculate metrics (mAP, precision, recall)                   │
│  • Generate performance report                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL EXPORT                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐    │
│  │ ONNX        │ │ TorchScript │ │ Other                   │    │
│  │ Format      │ │ Format      │ │ Formats                 │    │
│  │             │ │             │ │                         │    │
│  • Optimized   │ • Scripted    │ • CoreML                  │    │
│  • Deployable  │ • Traced      │ • TensorRT                │    │
│  • Cross-platform│ • Portable  │ • Custom                  │    │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL ORGANIZATION                           │
│  • Results folder created                                       │
│  • All files organized                                          │
│  • Documentation generated                                      │
│  • Ready for deployment                                         │
└─────────────────────────────────────────────────────────────────┘
```

**What happens in this step:**
1. **Evaluation**: Tests model performance
2. **Export**: Converts to different formats
3. **Organization**: Creates clean results structure
4. **Documentation**: Records what was accomplished

## Alternative Data Flow Paths

### **Interactive vs. Non-Interactive Mode**

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE CHOICE                        │
│  • Interactive mode (guided)                                   │
│  • Non-interactive mode (automated)                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTERACTIVE PATH                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
│  │ Step-by-step│ │ User        │ │ Validation               │   │
│  │ Prompts     │ │ Choices     │ │ & Confirmation           │   │
│  │             │ │             │ │                          │   │
│  • YOLO choice│ • Parameters  │ • Check inputs            │   │
│  • Model size │ • Settings    │ • Confirm choices         │   │
│  • Training   │ • Options     │ • Proceed                  │   │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NON-INTERACTIVE PATH                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
│  │ Command     │ │ Default     │ │ Automated                │   │
│  │ Line Args   │ │ Values      │ │ Execution                │   │
│  │             │ │             │ │                          │   │
│  • --model-type│ • Use        │ • Start training           │   │
│  • --epochs    │ • predefined │ • No prompts               │   │
│  • --batch-size│ • settings   │ • Full automation          │   │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERGENCE POINT                            │
│  Both paths lead to the same training execution                 │
│  • Same data flow                                              │
│  • Same training process                                        │
│  • Same results organization                                    │
└─────────────────────────────────────────────────────────────────┘
```

### **Error Handling and Recovery**

```
┌─────────────────────────────────────────────────────────────────┐
│                    ERROR DETECTION                              │
│  • Dataset issues detected                                      │
│  • Configuration problems                                       │
│  • Resource limitations                                         │
│  • Training failures                                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ERROR RECOVERY                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
│  │ Auto-fix     │ │ User        │ │ Fallback                │   │
│  │ Attempts     │ │ Notification │ │ Options                 │   │
│  │             │ │             │ │                          │   │
│  • Try to fix  │ • Show error │ • Alternative              │   │
│  • Retry       │ • Explain    │ • paths                    │   │
│  • Continue    │ • Suggest    │ • Manual                   │   │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RECOVERY RESULT                               │
│  • Issue resolved automatically                                 │
│  • User guided to fix manually                                 │
│  • Alternative approach suggested                               │
│  • Training continues or stops gracefully                       │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

### **Key Points to Remember:**

1. **Data flows in one direction**: Input → Processing → Output → Results
2. **Each step builds on the previous**: You can't skip steps
3. **Automation handles most work**: You only need to make key decisions
4. **Multiple paths available**: Interactive or automated based on your preference
5. **Error handling built-in**: System tries to fix problems automatically
6. **Results are organized**: Everything is saved in logical structure

### **What This Means for You:**

- **Simple workflow**: Just follow the prompts or use command line
- **Predictable results**: Same input always produces same output structure
- **Easy debugging**: Clear flow makes it easy to find problems
- **Flexible usage**: Choose your preferred interaction method
- **Professional results**: Industry-standard organization and naming

---

**Next**: We'll move to the [Core Components](02-core-components/01-main-training-script.md) section to examine each major component in detail.
