# Examples & Demos

## What These Examples Do

The examples and demos are like guided tours that show you exactly how to use the system in real-world scenarios. Think of them as interactive tutorials that walk you through different ways to train YOLO models, from simple automated training to complex custom configurations.

## Examples Overview

### **File Structure**
```
examples/
├── demo_interactive_training.py    # Interactive training demo (4.6KB)
└── export_dataset.py               # Roboflow dataset export example (5.0KB)
```

### **Purpose of Each Example**
- **`demo_interactive_training.py`**: Shows different training modes and options
- **`export_dataset.py`**: Demonstrates Roboflow integration and dataset export

## Interactive Training Demo

### **What It Does**
The interactive training demo is like a training simulator that lets you try different training approaches without actually starting training. It's perfect for learning how the system works and experimenting with different configurations.

### **How to Run**
```bash
cd examples
python demo_interactive_training.py
```

### **Demo Features**

#### **1. Command Line Options Display**
```bash
python train.py --model-type yolov8 --help
```
**What it shows:**
- All available command line options
- Parameter descriptions and usage
- Default values for each option
- Help text and examples

**Best for:** Understanding what options are available

#### **2. Automated Training Demo**
```bash
python train.py --model-type yolov8 --non-interactive --results-folder demo_run
```
**What it does:**
- Runs training with all default values
- No user prompts or interruptions
- Creates results folder named "demo_run"
- Perfect for automation and scripting

**Best for:** Production training, automation, repeated experiments

#### **3. Validation Only Demo**
```bash
python train.py --model-type yolov8 --validate-only --non-interactive --results-folder demo_validation
```
**What it does:**
- Tests existing model without training
- Evaluates performance on validation data
- No training time required
- Quick model assessment

**Best for:** Testing pre-trained models, performance evaluation

#### **4. Quick Training Demo**
```bash
python train.py --model-type yolov8 --epochs 5 --batch-size 4 --image-size 640 --results-folder quick_test
```
**What it does:**
- Custom training parameters
- Short training (5 epochs)
- Small batch size (4)
- Reduced image size (640x640)
- No interactive prompts

**Best for:** Quick experiments, testing configurations, development

#### **5. Full Interactive Experience**
```bash
python train.py
```
**What it does:**
- Guides you through every decision
- Explains each option
- Helps you choose optimal settings
- Creates custom results folder
- Educational experience

**Best for:** Learning, first-time users, custom configurations

### **Demo Workflow**

#### **Step 1: Choose Demo Type**
```
YOLO Interactive Training System Demo
============================================================
This demo shows you different ways to run YOLO training.
Choose which example you'd like to try:

Available examples:
1. Show all available command line options
2. Run training with defaults (no prompts)
3. Run validation only with defaults
4. Run quick training with custom parameters (no prompts)
5. Interactive training with YOLO version selection (full experience)
6. Exit demo
```

#### **Step 2: Execute Demo**
For each demo option, you can:
- **Run the command**: Execute it immediately
- **Skip the command**: See what it would do without running
- **Continue demo**: Try another example
- **Exit demo**: Finish the demonstration

#### **Step 3: Learn from Results**
Each demo shows you:
- **Command output**: What the system produces
- **Success/failure**: Whether the command worked
- **Error messages**: What went wrong (if anything)
- **Next steps**: How to proceed

### **Demo Scenarios**

#### **Scenario 1: Learning the System**
1. Start with option 1 (help display)
2. Try option 5 (full interactive experience)
3. Learn what each parameter does
4. Understand the workflow

#### **Scenario 2: Quick Testing**
1. Use option 4 (quick training)
2. Test with small dataset
3. Verify system works
4. Check results quickly

#### **Scenario 3: Production Setup**
1. Use option 2 (automated training)
2. Set up repeatable training
3. Configure for your needs
4. Automate the process

#### **Scenario 4: Model Evaluation**
1. Use option 3 (validation only)
2. Test existing models
3. Compare performance
4. Validate results

## Roboflow Dataset Export Example

### **What It Does**
The Roboflow export example shows you how to integrate with Roboflow (a popular dataset platform) to automatically download and prepare datasets for YOLO training. It's perfect if you're using Roboflow for dataset management.

### **How to Run**
```bash
cd examples
python export_dataset.py
```

### **Prerequisites**
```bash
# Install Roboflow package
pip install roboflow

# Set your API key
export ROBOFLOW_API_KEY='your_api_key_here'
```

### **Configuration Setup**
Before running, update the configuration in the script:

```python
config = {
    "api_key": os.getenv("ROBOFLOW_API_KEY"),
    "workspace": "your_workspace_name",      # Replace with your workspace
    "project_id": "your_project_id",         # Replace with your project ID
    "version": "your_version_number",        # Replace with your version
}
```

### **Export Process**

#### **Step 1: Initialize Roboflow**
```python
def export_dataset_for_yolo(api_key, workspace, project_id, version, yolo_version):
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Get project and version
    project = rf.workspace(workspace).project(project_id)
    dataset = project.version(version).download(yolo_version)
    
    return dataset.location
```

#### **Step 2: Export for Different YOLO Versions**
The script exports your dataset in multiple YOLO formats:

```python
yolo_versions = ["yolo11", "yolov8", "yolov5", "yolo"]

for yolo_version in yolo_versions:
    print(f"📦 Exporting for {yolo_version}...")
    
    dataset_path = export_dataset_for_yolo(
        api_key=config["api_key"],
        workspace=config["workspace"],
        project_id=config["project_id"],
        version=config["version"],
        yolo_version=yolo_version,
    )
```

#### **Step 3: Verify Dataset Structure**
After export, the script verifies the dataset is correct:

```python
def verify_dataset_structure(dataset_path: str) -> bool:
    required_dirs = [
        "train/images", "train/labels",
        "valid/images", "valid/labels",
        "test/images", "test/labels",
    ]
    
    required_files = ["data.yaml"]
    
    # Check all required directories and files
    # Validate data.yaml content
    # Ensure proper structure
```

### **Export Features**

#### **Multiple YOLO Format Support**
- **YOLO11**: Latest version format
- **YOLOv8**: Stable, well-tested format
- **YOLOv5**: Classic format
- **YOLO**: Generic format

#### **Automatic Structure Validation**
- **Folder Structure**: Ensures train/valid/test folders exist
- **File Presence**: Checks for images and labels
- **Configuration**: Validates data.yaml content
- **Format Compliance**: Ensures YOLO compatibility

#### **Error Handling**
- **API Key Validation**: Checks if Roboflow access is configured
- **Export Failures**: Handles download errors gracefully
- **Structure Issues**: Reports missing components
- **Configuration Problems**: Guides you to fix setup issues

### **Integration with Training System**

#### **Seamless Workflow**
1. **Export from Roboflow**: Get dataset in YOLO format
2. **Automatic Preparation**: System detects and prepares dataset
3. **Start Training**: Begin training immediately
4. **No Manual Work**: Everything happens automatically

#### **Example Workflow**
```bash
# 1. Export dataset from Roboflow
python examples/export_dataset.py

# 2. Dataset is automatically prepared
# 3. Start training with prepared dataset
python train.py --model-type yolov8 --non-interactive
```

## Practical Usage Examples

### **Example 1: First-Time User Learning**

#### **Goal**: Learn how the system works
#### **Steps**:
1. **Run the demo**: `python examples/demo_interactive_training.py`
2. **Choose option 5**: Full interactive experience
3. **Follow prompts**: Learn what each option does
4. **Understand workflow**: See how everything connects

#### **Expected Outcome**:
- Understanding of all training options
- Confidence in using the system
- Knowledge of best practices

### **Example 2: Quick Experiment**

#### **Goal**: Test a new configuration quickly
#### **Steps**:
1. **Use quick training**: Option 4 in demo
2. **Set parameters**: epochs=5, batch_size=4, image_size=640
3. **Run training**: No prompts, fast execution
4. **Check results**: Verify configuration works

#### **Expected Outcome**:
- Quick validation of settings
- Fast feedback on configuration
- Ready for full training

### **Example 3: Production Automation**

#### **Goal**: Set up automated training pipeline
#### **Steps**:
1. **Use automated mode**: Option 2 in demo
2. **Configure parameters**: Set all options via command line
3. **Create script**: Automate the training process
4. **Schedule runs**: Set up regular training

#### **Expected Outcome**:
- Repeatable training process
- Automated workflow
- Consistent results

### **Example 4: Dataset Integration**

#### **Goal**: Use Roboflow datasets with the system
#### **Steps**:
1. **Export dataset**: Run `python examples/export_dataset.py`
2. **Verify structure**: Ensure dataset is correct
3. **Start training**: Use exported dataset immediately
4. **Monitor progress**: Track training with new data

#### **Expected Outcome**:
- Integrated dataset workflow
- No manual preparation needed
- Seamless training experience

## Best Practices for Using Examples

### **For Beginners**
1. **Start with demos**: Use `demo_interactive_training.py` first
2. **Follow the prompts**: Let the system guide you
3. **Try different options**: Experiment with various configurations
4. **Learn from output**: Understand what each command produces

### **For Intermediate Users**
1. **Customize examples**: Modify scripts for your needs
2. **Combine approaches**: Mix automated and interactive modes
3. **Extend functionality**: Add your own examples
4. **Optimize workflows**: Create efficient training pipelines

### **For Advanced Users**
1. **Script automation**: Create custom training scripts
2. **Integration**: Connect with other systems
3. **Customization**: Extend examples for specific use cases
4. **Production deployment**: Use examples as templates

## Troubleshooting Example Issues

### **Common Demo Problems**

#### **"Command not found"**
- **Problem**: Python or script not in PATH
- **Solution**: Use full path or navigate to examples folder
- **Alternative**: Run from project root directory

#### **"Permission denied"**
- **Problem**: Script not executable
- **Solution**: Make script executable with `chmod +x script.py`
- **Alternative**: Run with `python script.py`

#### **"Module not found"**
- **Problem**: Required packages not installed
- **Solution**: Install dependencies with pip
- **Alternative**: Check requirements.txt and install

### **Roboflow Export Issues**

#### **"API key not set"**
- **Problem**: Roboflow API key not configured
- **Solution**: Set environment variable `ROBOFLOW_API_KEY`
- **Alternative**: Update script with API key directly

#### **"Project not found"**
- **Problem**: Incorrect workspace/project configuration
- **Solution**: Check Roboflow dashboard for correct values
- **Alternative**: Verify project exists and is accessible

#### **"Export failed"**
- **Problem**: Dataset export process failed
- **Solution**: Check Roboflow service status
- **Alternative**: Try different YOLO format

## Extending the Examples

### **Creating Custom Examples**

#### **Template Structure**
```python
#!/usr/bin/env python3
"""
Custom Example for [Your Use Case]

This script demonstrates [specific functionality].
"""

def main():
    """Main function for custom example."""
    print("Custom Example")
    print("=" * 50)
    
    # Your custom logic here
    
    print("Example completed!")

if __name__ == "__main__":
    main()
```

#### **Integration Points**
- **Configuration**: Use system configuration classes
- **Utilities**: Leverage existing utility modules
- **Training**: Integrate with training system
- **Validation**: Use built-in validation functions

### **Adding New Demo Options**

#### **Extending Demo Script**
```python
# Add new example to examples list
examples.append({
    "command": "python train.py --custom-option",
    "description": "Custom training configuration",
})

# Add new menu option
print("7. Custom training configuration")

# Handle new choice
elif choice == "7":
    run_command(examples[6]["command"], examples[6]["description"])
```

#### **Best Practices**
- **Clear descriptions**: Explain what each example does
- **Consistent format**: Follow existing example structure
- **Error handling**: Include proper error handling
- **Documentation**: Document new examples clearly

---

**Next**: We'll explore the [Testing Framework](02-testing-framework.md) to understand how the system is tested and validated.
