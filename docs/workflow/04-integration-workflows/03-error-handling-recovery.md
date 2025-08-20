# Error Handling & Recovery

## What This System Does

Error handling and recovery is the safety net that keeps your training pipeline running smoothly even when things go wrong. Think of it as the system's immune system - it detects problems, responds appropriately, and helps you recover quickly. This ensures that a single failure doesn't bring down your entire training process and provides clear guidance on how to fix issues.

## Error Handling Architecture Overview

### **The Resilience Strategy**

The system implements a multi-layered approach to error handling:

```
Detection → Classification → Response → Recovery → Prevention
    ↓           ↓           ↓         ↓         ↓
  Monitoring  Categorizing  Actions   Restore   Learn
```

### **Error Handling Layers**

1. **Input Validation Layer** - Prevent invalid data from entering the system
2. **Runtime Protection Layer** - Catch and handle errors during execution
3. **Recovery Layer** - Restore system to working state after failures
4. **Monitoring Layer** - Track errors and provide diagnostic information
5. **Prevention Layer** - Learn from errors to prevent future occurrences

## Input Validation Layer

### **Dataset Validation**

#### **Structure Validation**
```python
def validate_dataset_input(dataset_path: Path) -> bool:
    """Validate dataset input before processing."""
    
    # Check directory structure
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Check for required files
    required_files = ["images", "labels"]
    for req_file in required_files:
        if not (dataset_path / req_file).exists():
            logger.warning(f"Missing required directory: {req_file}")
    
    # Check file counts
    image_files = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png"))
    label_files = list(dataset_path.rglob("*.txt")) + list(dataset_path.rglob("*.json"))
    
    if len(image_files) == 0:
        raise ValueError("No image files found in dataset")
    
    if len(label_files) == 0:
        logger.warning("No label files found in dataset")
    
    return True
```

**Error Prevention:**
- **Early Detection**: Validates dataset structure before processing
- **Clear Messages**: Provides specific information about what's missing
- **Graceful Degradation**: Warns about non-critical issues while continuing

#### **Format Validation**
```python
def validate_annotation_formats(dataset_path: Path) -> Dict[str, List[str]]:
    """Validate annotation formats and identify issues."""
    
    validation_results = {
        "valid_files": [],
        "invalid_files": [],
        "warnings": []
    }
    
    for label_file in dataset_path.rglob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            # Validate YOLO format
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                
                if len(parts) != 5:
                    validation_results["invalid_files"].append(
                        f"{label_file}:{line_num} - Invalid format (expected 5 parts, got {len(parts)})"
                    )
                    continue
                
                # Validate numeric values
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Check value ranges
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        validation_results["warnings"].append(
                            f"{label_file}:{line_num} - Coordinates out of range [0,1]"
                        )
                    
                    if not (0 < width <= 1 and 0 < height <= 1):
                        validation_results["warnings"].append(
                            f"{label_file}:{line_num} - Dimensions out of range (0,1]"
                        )
                        
                except ValueError:
                    validation_results["invalid_files"].append(
                        f"{label_file}:{line_num} - Non-numeric values"
                    )
            
            if not validation_results["invalid_files"]:
                validation_results["valid_files"].append(str(label_file))
                
        except Exception as e:
            validation_results["invalid_files"].append(
                f"{label_file} - Read error: {e}"
            )
    
    return validation_results
```

**Error Prevention:**
- **Format Checking**: Validates YOLO annotation format
- **Range Validation**: Ensures coordinate values are within valid ranges
- **Detailed Reporting**: Provides specific line numbers and error descriptions

### **Configuration Validation**

#### **Parameter Validation**
```python
def validate_configuration(config: "YOLOConfig") -> bool:
    """Validate training configuration."""
    errors = []
    warnings = []
    
    # Check required fields
    if not config.model_type:
        errors.append("Model type is required")
    
    if not config.weights:
        errors.append("Model weights are required")
    
    # Check numeric parameters
    if config.epochs <= 0:
        errors.append("Epochs must be positive")
    
    if config.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if config.image_size <= 0:
        errors.append("Image size must be positive")
    
    # Check parameter ranges
    if config.batch_size > 128:
        warnings.append("Large batch size may cause memory issues")
    
    if config.image_size > 1024:
        warnings.append("Large image size may slow training significantly")
    
    # Check device availability
    if config.device == "cuda" and not torch.cuda.is_available():
        errors.append("CUDA requested but not available")
    
    # Report errors and warnings
    if warnings:
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
    
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        return False
    
    return True
```

**Error Prevention:**
- **Parameter Bounds**: Ensures parameters are within valid ranges
- **Device Checking**: Verifies hardware capabilities before training
- **Warning System**: Alerts about potential issues without stopping execution

## Runtime Protection Layer

### **Exception Handling Patterns**

#### **Graceful Degradation Pattern**
```python
def _detect_classes(self) -> List[str]:
    """Detect class names from annotations with fallback strategies."""
    class_names = set()
    
    # Try multiple detection methods with fallbacks
    detection_methods = [
        self._detect_classes_from_yolo_labels,
        self._detect_classes_from_coco_annotations,
        self._detect_classes_from_class_mapping,
        self._detect_classes_from_directory_structure
    ]
    
    for method in detection_methods:
        try:
            result = method()
            if result and result.get("classes"):
                class_names.update(result["classes"])
                logger.info(f"Detected {len(result['classes'])} classes using {method.__name__}")
                break
        except Exception as e:
            logger.debug(f"Method {method.__name__} failed: {e}")
            continue
    
    # Fallback: assume single class if nothing detected
    if not class_names:
        logger.warning("No classes detected, assuming single class 'object'")
        class_names = {"object"}
    
    return sorted(list(class_names))
```

**Error Handling Strategy:**
- **Multiple Methods**: Tries different detection approaches
- **Graceful Fallback**: Provides reasonable defaults when detection fails
- **Detailed Logging**: Records what worked and what didn't

#### **Retry Pattern**
```python
def download_weights_with_retry(model_type: str, size: str, max_retries: int = 3) -> bool:
    """Download weights with automatic retry on failure."""
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading weights (attempt {attempt + 1}/{max_retries})")
            
            if download_weights(model_type, size):
                logger.info("Weights downloaded successfully")
                return True
            else:
                raise RuntimeError("Download failed")
                
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("All download attempts failed")
                return False
    
    return False
```

**Error Handling Strategy:**
- **Automatic Retry**: Attempts operation multiple times
- **Exponential Backoff**: Increases delay between retries
- **Clear Reporting**: Shows progress and final outcome

#### **Resource Management Pattern**
```python
def safe_model_loading(config: "YOLOConfig") -> Optional[nn.Module]:
    """Safely load model with resource management."""
    
    try:
        # Check available memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 2:
                logger.warning(f"Low GPU memory ({gpu_memory:.1f}GB), may cause issues")
        
        # Load model
        model = load_yolo_model(config)
        
        # Verify model loaded correctly
        if model is None:
            raise RuntimeError("Model loading returned None")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 3, config.image_size, config.image_size)
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info("Model loaded and verified successfully")
        return model
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        
        # Clean up any partial resources
        if 'model' in locals():
            del model
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None
```

**Error Handling Strategy:**
- **Resource Checking**: Verifies system capabilities before loading
- **Model Verification**: Tests model functionality after loading
- **Cleanup**: Ensures resources are properly released on failure

### **Training Error Handling**

#### **Training Loop Protection**
```python
def safe_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: "YOLOConfig"
) -> Dict[str, Any]:
    """Training loop with comprehensive error handling."""
    
    training_state = {
        "epoch": 0,
        "batch": 0,
        "total_loss": 0.0,
        "successful_batches": 0,
        "failed_batches": 0,
        "errors": []
    }
    
    try:
        model.train()
        
        for batch_idx, (images, labels, paths, shapes) in enumerate(train_loader):
            try:
                # Move data to device
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Check for invalid loss values
                if not torch.isfinite(loss):
                    logger.warning(f"Invalid loss value detected: {loss}")
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Update metrics
                training_state["total_loss"] += loss.item()
                training_state["successful_batches"] += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"GPU out of memory at batch {batch_idx}: {e}")
                    training_state["errors"].append(f"OOM at batch {batch_idx}: {e}")
                    
                    # Clear GPU cache and continue
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    training_state["failed_batches"] += 1
                    continue
                    
                else:
                    logger.error(f"Runtime error at batch {batch_idx}: {e}")
                    training_state["errors"].append(f"Runtime at batch {batch_idx}: {e}")
                    training_state["failed_batches"] += 1
                    continue
                    
            except Exception as e:
                logger.error(f"Unexpected error at batch {batch_idx}: {e}")
                training_state["errors"].append(f"Unexpected at batch {batch_idx}: {e}")
                training_state["failed_batches"] += 1
                continue
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        training_state["errors"].append("Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training loop failed: {e}")
        training_state["errors"].append(f"Training loop failed: {e}")
    
    finally:
        # Calculate final metrics
        if training_state["successful_batches"] > 0:
            training_state["avg_loss"] = training_state["total_loss"] / training_state["successful_batches"]
        else:
            training_state["avg_loss"] = float('inf')
        
        # Log summary
        logger.info(f"Training completed: {training_state['successful_batches']} successful, "
                   f"{training_state['failed_batches']} failed")
    
    return training_state
```

**Error Handling Strategy:**
- **Batch-Level Protection**: Continues training even if individual batches fail
- **Memory Management**: Handles GPU out-of-memory errors gracefully
- **Progress Tracking**: Records successful and failed operations
- **Graceful Interruption**: Handles user interruption cleanly

## Recovery Layer

### **Checkpoint Recovery**

#### **Automatic Checkpoint Management**
```python
def recover_from_failure(
    config: "YOLOConfig",
    checkpoint_manager: CheckpointManager,
    failure_point: str
) -> Optional[Dict[str, Any]]:
    """Recover training state from the most recent checkpoint."""
    
    try:
        # Find the best recovery checkpoint
        recovery_checkpoint = None
        
        if failure_point == "training_start":
            # Use the latest checkpoint
            recovery_checkpoint = checkpoint_manager.get_latest_checkpoint()
        elif failure_point == "training_loop":
            # Use the best checkpoint based on metrics
            recovery_checkpoint = checkpoint_manager.get_best_checkpoint()
        elif failure_point == "validation":
            # Use the last successful training checkpoint
            recovery_checkpoint = checkpoint_manager.get_latest_checkpoint()
        
        if recovery_checkpoint and recovery_checkpoint.exists():
            logger.info(f"Recovering from checkpoint: {recovery_checkpoint}")
            
            # Load checkpoint data
            checkpoint_data = torch.load(recovery_checkpoint, map_location="cpu")
            
            # Extract recovery information
            recovery_info = {
                "checkpoint_path": recovery_checkpoint,
                "epoch": checkpoint_data.get("epoch", 0),
                "metrics": checkpoint_data.get("metrics", {}),
                "model_state": checkpoint_data.get("model", None),
                "optimizer_state": checkpoint_data.get("optimizer", None),
                "scheduler_state": checkpoint_data.get("scheduler", None),
                "training_config": checkpoint_data.get("config", None)
            }
            
            logger.info(f"Recovery successful: epoch {recovery_info['epoch']}")
            return recovery_info
            
        else:
            logger.warning("No suitable checkpoint found for recovery")
            return None
            
    except Exception as e:
        logger.error(f"Recovery failed: {e}")
        return None
```

**Recovery Strategy:**
- **Smart Checkpoint Selection**: Chooses appropriate checkpoint based on failure type
- **State Restoration**: Recovers model, optimizer, and training state
- **Progress Preservation**: Continues from where training left off

#### **Incremental Recovery**
```python
def incremental_recovery(
    config: "YOLOConfig",
    checkpoint_manager: CheckpointManager,
    target_epoch: int
) -> bool:
    """Recover training incrementally to reach target epoch."""
    
    try:
        # Find the closest checkpoint to target epoch
        available_checkpoints = checkpoint_manager.get_all_checkpoints()
        
        if not available_checkpoints:
            logger.warning("No checkpoints available for incremental recovery")
            return False
        
        # Find the best starting point
        best_checkpoint = None
        min_distance = float('inf')
        
        for checkpoint in available_checkpoints:
            checkpoint_epoch = checkpoint.get("epoch", 0)
            distance = abs(checkpoint_epoch - target_epoch)
            
            if distance < min_distance:
                min_distance = distance
                best_checkpoint = checkpoint
        
        if best_checkpoint:
            logger.info(f"Starting incremental recovery from epoch {best_checkpoint['epoch']}")
            
            # Load checkpoint
            checkpoint_path = Path(best_checkpoint["path"])
            if checkpoint_path.exists():
                # Resume training from this checkpoint
                return resume_training_from_checkpoint(checkpoint_path, config, target_epoch)
            else:
                logger.warning(f"Checkpoint file not found: {checkpoint_path}")
                return False
        else:
            logger.warning("No suitable checkpoint found for incremental recovery")
            return False
            
    except Exception as e:
        logger.error(f"Incremental recovery failed: {e}")
        return False
```

**Recovery Strategy:**
- **Smart Starting Point**: Finds the best checkpoint to resume from
- **Progress Tracking**: Monitors recovery progress
- **Flexible Targets**: Allows recovery to specific epochs

### **Data Recovery**

#### **Dataset Recovery**
```python
def recover_dataset_preparation(
    dataset_path: Path,
    failure_point: str
) -> Optional[Path]:
    """Recover dataset preparation from failure point."""
    
    try:
        # Check what was already prepared
        prepared_path = dataset_path / "prepared"
        
        if not prepared_path.exists():
            logger.info("No prepared dataset found, starting fresh preparation")
            return auto_prepare_dataset(dataset_path, "yolo")
        
        # Check preparation progress
        preparation_state_file = prepared_path / "preparation_state.json"
        
        if preparation_state_file.exists():
            with open(preparation_state_file, 'r') as f:
                state = json.load(f)
            
            last_completed_step = state.get("last_completed_step", "")
            logger.info(f"Resuming from step: {last_completed_step}")
            
            # Resume from the last completed step
            if last_completed_step == "structure_analysis":
                return _resume_from_structure_analysis(dataset_path, prepared_path)
            elif last_completed_step == "format_conversion":
                return _resume_from_format_conversion(dataset_path, prepared_path)
            elif last_completed_step == "reorganization":
                return _resume_from_reorganization(dataset_path, prepared_path)
            elif last_completed_step == "validation":
                return _resume_from_validation(dataset_path, prepared_path)
            else:
                logger.info("Preparation state unclear, starting fresh")
                return auto_prepare_dataset(dataset_path, "yolo")
        else:
            logger.info("No preparation state found, starting fresh preparation")
            return auto_prepare_dataset(dataset_path, "yolo")
            
    except Exception as e:
        logger.error(f"Dataset recovery failed: {e}")
        return None
```

**Recovery Strategy:**
- **Progress Tracking**: Records preparation steps as they complete
- **Resume Capability**: Continues from the last successful step
- **State Persistence**: Saves preparation state for recovery

#### **Partial Data Recovery**
```python
def recover_partial_dataset(
    dataset_path: Path,
    prepared_path: Path
) -> bool:
    """Recover partially prepared dataset."""
    
    try:
        # Check what directories exist
        required_dirs = ["train/images", "train/labels", "valid/images", "valid/labels"]
        existing_dirs = []
        missing_dirs = []
        
        for dir_path in required_dirs:
            full_path = prepared_path / dir_path
            if full_path.exists() and list(full_path.glob("*")):
                existing_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)
        
        logger.info(f"Existing directories: {existing_dirs}")
        logger.info(f"Missing directories: {missing_dirs}")
        
        if not existing_dirs:
            logger.info("No prepared data found, starting fresh")
            return False
        
        # Try to complete missing parts
        for missing_dir in missing_dirs:
            logger.info(f"Completing missing directory: {missing_dir}")
            
            if missing_dir.endswith("/images"):
                # Copy images from source
                source_split = missing_dir.split("/")[0]
                source_images = dataset_path / source_split / "images"
                
                if source_images.exists():
                    target_images = prepared_path / missing_dir
                    target_images.mkdir(parents=True, exist_ok=True)
                    
                    # Copy image files
                    for img_file in source_images.glob("*"):
                        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                            shutil.copy2(img_file, target_images)
                    
                    logger.info(f"Restored images for {missing_dir}")
            
            elif missing_dir.endswith("/labels"):
                # Convert labels for this split
                source_split = missing_dir.split("/")[0]
                source_labels = dataset_path / source_split / "labels"
                
                if source_labels.exists():
                    target_labels = prepared_path / missing_dir
                    target_labels.mkdir(parents=True, exist_ok=True)
                    
                    # Convert labels to YOLO format
                    convert_labels_to_yolo(source_labels, target_labels)
                    
                    logger.info(f"Restored labels for {missing_dir}")
        
        # Validate final structure
        if _validate_final_structure(prepared_path):
            logger.info("Dataset recovery completed successfully")
            return True
        else:
            logger.warning("Dataset recovery completed but validation failed")
            return False
            
    except Exception as e:
        logger.error(f"Partial dataset recovery failed: {e}")
        return False
```

**Recovery Strategy:**
- **Partial Restoration**: Rebuilds missing parts of the dataset
- **Smart Copying**: Reuses existing prepared data
- **Validation**: Ensures recovered dataset is valid

## Monitoring Layer

### **Error Tracking**

#### **Comprehensive Error Logging**
```python
class ErrorTracker:
    """Track and categorize errors throughout the system."""
    
    def __init__(self):
        self.errors = []
        self.error_counts = {}
        self.error_categories = {
            "dataset": [],
            "model": [],
            "training": [],
            "validation": [],
            "export": [],
            "system": []
        }
    
    def log_error(self, error: Exception, context: str, category: str = "system"):
        """Log an error with context and categorization."""
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "category": category,
            "traceback": traceback.format_exc()
        }
        
        # Add to error list
        self.errors.append(error_info)
        
        # Update error counts
        error_key = f"{category}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Add to category
        if category in self.error_categories:
            self.error_categories[category].append(error_info)
        
        # Log error
        logger.error(f"Error in {category}: {error}")
        logger.debug(f"Error context: {context}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts,
            "category_breakdown": {
                category: len(errors) for category, errors in self.error_categories.items()
            },
            "recent_errors": self.errors[-10:] if self.errors else []
        }
    
    def get_common_errors(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get most common errors."""
        sorted_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_errors[:limit]
    
    def clear_errors(self):
        """Clear all tracked errors."""
        self.errors.clear()
        self.error_counts.clear()
        for category in self.error_categories:
            self.error_categories[category].clear()
```

**Monitoring Strategy:**
- **Categorization**: Groups errors by system component
- **Context Tracking**: Records what was happening when errors occurred
- **Trend Analysis**: Identifies common error patterns

#### **Performance Monitoring**
```python
class PerformanceMonitor:
    """Monitor system performance and detect degradation."""
    
    def __init__(self):
        self.metrics = {
            "memory_usage": [],
            "gpu_utilization": [],
            "training_speed": [],
            "error_rates": []
        }
        self.thresholds = {
            "memory_warning": 0.8,  # 80% memory usage
            "gpu_warning": 0.9,     # 90% GPU utilization
            "speed_warning": 0.5    # 50% slower than baseline
        }
    
    def check_memory_health(self) -> Dict[str, Any]:
        """Check memory health and provide warnings."""
        health_status = {"status": "healthy", "warnings": []}
        
        if torch.cuda.is_available():
            # GPU memory
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            if gpu_memory > self.thresholds["memory_warning"]:
                health_status["status"] = "warning"
                health_status["warnings"].append(
                    f"High GPU memory usage: {gpu_memory:.1%}"
                )
            
            # System memory
            import psutil
            system_memory = psutil.virtual_memory()
            
            if system_memory.percent > 80:
                health_status["status"] = "warning"
                health_status["warnings"].append(
                    f"High system memory usage: {system_memory.percent:.1%}"
                )
        
        return health_status
    
    def detect_performance_degradation(self) -> List[str]:
        """Detect performance degradation patterns."""
        warnings = []
        
        # Check training speed
        if len(self.metrics["training_speed"]) > 10:
            recent_speed = np.mean(self.metrics["training_speed"][-5:])
            baseline_speed = np.mean(self.metrics["training_speed"][:5])
            
            if recent_speed < baseline_speed * self.thresholds["speed_warning"]:
                warnings.append(f"Training speed degraded: {recent_speed:.2f} vs {baseline_speed:.2f}")
        
        # Check error rates
        if len(self.metrics["error_rates"]) > 10:
            recent_error_rate = np.mean(self.metrics["error_rates"][-5:])
            
            if recent_error_rate > 0.1:  # 10% error rate
                warnings.append(f"High error rate detected: {recent_error_rate:.1%}")
        
        return warnings
```

**Monitoring Strategy:**
- **Threshold Monitoring**: Warns when metrics exceed safe limits
- **Trend Analysis**: Detects performance degradation over time
- **Proactive Alerts**: Warns before problems become critical

### **Diagnostic Information**

#### **System Health Check**
```python
def perform_system_health_check() -> Dict[str, Any]:
    """Perform comprehensive system health check."""
    
    health_report = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "checks": {},
        "recommendations": []
    }
    
    # Check Python environment
    try:
        import torch
        health_report["checks"]["pytorch"] = {
            "status": "healthy",
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            health_report["checks"]["pytorch"]["cuda_version"] = torch.version.cuda
            health_report["checks"]["pytorch"]["gpu_count"] = torch.cuda.device_count()
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            health_report["checks"]["pytorch"]["gpu_memory_gb"] = gpu_memory
            
            if gpu_memory < 4:
                health_report["recommendations"].append(
                    "GPU memory is limited. Consider reducing batch size or image size."
                )
    except ImportError:
        health_report["checks"]["pytorch"] = {"status": "error", "message": "PyTorch not installed"}
        health_report["overall_status"] = "degraded"
    
    # Check required packages
    required_packages = ["ultralytics", "opencv-python", "Pillow", "numpy"]
    for package in required_packages:
        try:
            module = __import__(package.replace("-", "_"))
            health_report["checks"][package] = {
                "status": "healthy",
                "version": getattr(module, "__version__", "unknown")
            }
        except ImportError:
            health_report["checks"][package] = {"status": "error", "message": "Package not installed"}
            health_report["overall_status"] = "degraded"
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        health_report["checks"]["disk_space"] = {
            "status": "healthy" if free_gb > 10 else "warning",
            "free_gb": free_gb,
            "total_gb": total / (1024**3)
        }
        
        if free_gb < 10:
            health_report["recommendations"].append(
                f"Low disk space: {free_gb:.1f}GB free. Consider cleaning up old files."
            )
    except Exception as e:
        health_report["checks"]["disk_space"] = {"status": "error", "message": str(e)}
    
    # Check dataset directory
    dataset_path = Path("dataset")
    if dataset_path.exists():
        dataset_files = list(dataset_path.rglob("*"))
        health_report["checks"]["dataset"] = {
            "status": "healthy",
            "file_count": len(dataset_files),
            "total_size_mb": sum(f.stat().st_size for f in dataset_files if f.is_file()) / (1024**2)
        }
    else:
        health_report["checks"]["dataset"] = {"status": "warning", "message": "Dataset directory not found"}
    
    return health_report
```

**Diagnostic Strategy:**
- **Comprehensive Checking**: Examines all system components
- **Clear Status**: Provides overall health status
- **Actionable Recommendations**: Suggests specific improvements

## Prevention Layer

### **Error Prevention Strategies**

#### **Proactive Validation**
```python
def validate_before_training(config: "YOLOConfig") -> Tuple[bool, List[str]]:
    """Validate everything before starting training."""
    
    validation_results = []
    all_valid = True
    
    # Validate configuration
    if not validate_configuration(config):
        validation_results.append("Configuration validation failed")
        all_valid = False
    
    # Validate dataset
    dataset_path = Path(config.dataset_config.get("data_yaml_path", "")).parent
    if not dataset_path.exists():
        validation_results.append("Dataset path does not exist")
        all_valid = False
    else:
        # Check dataset structure
        if not _validate_dataset_structure(dataset_path):
            validation_results.append("Dataset structure is invalid")
            all_valid = False
        
        # Check data quality
        quality_report = check_data_quality(dataset_path)
        if quality_report["image_issues"] or quality_report["label_issues"]:
            validation_results.append("Dataset has quality issues")
            all_valid = False
    
    # Validate model weights
    weights_path = Path(config.weights)
    if not weights_path.exists():
        validation_results.append("Model weights file not found")
        all_valid = False
    
    # Validate hardware requirements
    if config.device == "cuda" and not torch.cuda.is_available():
        validation_results.append("CUDA requested but not available")
        all_valid = False
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        if free_gb < 5:
            validation_results.append(f"Insufficient disk space: {free_gb:.1f}GB free")
            all_valid = False
    except Exception:
        validation_results.append("Could not check disk space")
    
    return all_valid, validation_results
```

**Prevention Strategy:**
- **Pre-flight Check**: Validates everything before training starts
- **Resource Verification**: Ensures sufficient disk space and memory
- **Early Detection**: Catches problems before they cause failures

#### **Best Practices Enforcement**
```python
def enforce_training_best_practices(config: "YOLOConfig") -> List[str]:
    """Enforce training best practices and provide recommendations."""
    
    recommendations = []
    
    # Check batch size
    if config.batch_size > 64:
        recommendations.append("Large batch size detected. Consider reducing to 32 or less for stability.")
    
    if config.batch_size < 4:
        recommendations.append("Very small batch size detected. Consider increasing to 8 or more for efficiency.")
    
    # Check learning rate
    lr = config.model_config.get("learning_rate", 0.01)
    if lr > 0.1:
        recommendations.append("High learning rate detected. Consider reducing to 0.01 or less for stability.")
    
    if lr < 0.0001:
        recommendations.append("Very low learning rate detected. Consider increasing to 0.001 or more for reasonable training time.")
    
    # Check image size
    if config.image_size > 1024:
        recommendations.append("Large image size detected. This will significantly slow training and increase memory usage.")
    
    # Check epochs
    if config.epochs > 1000:
        recommendations.append("Very high epoch count detected. Consider starting with 100-200 epochs and monitoring progress.")
    
    if config.epochs < 10:
        recommendations.append("Very low epoch count detected. Consider increasing to at least 50 epochs for meaningful training.")
    
    # Check patience
    if config.patience > config.epochs // 2:
        recommendations.append("Patience is very high relative to total epochs. Consider reducing to 10-20% of total epochs.")
    
    # Check device
    if config.device == "cpu" and config.batch_size > 16:
        recommendations.append("Large batch size with CPU training. Consider reducing batch size or using GPU if available.")
    
    return recommendations
```

**Prevention Strategy:**
- **Parameter Validation**: Checks for common configuration mistakes
- **Performance Guidance**: Suggests optimal parameter ranges
- **Resource Optimization**: Recommends settings for better performance

### **Learning from Errors**

#### **Error Pattern Analysis**
```python
def analyze_error_patterns(error_tracker: ErrorTracker) -> Dict[str, Any]:
    """Analyze error patterns to prevent future occurrences."""
    
    analysis = {
        "common_patterns": [],
        "prevention_strategies": [],
        "system_improvements": []
    }
    
    # Analyze error categories
    category_errors = error_tracker.error_categories
    
    # Dataset errors
    if category_errors["dataset"]:
        dataset_errors = [e["error_type"] for e in category_errors["dataset"]]
        if "FileNotFoundError" in dataset_errors:
            analysis["common_patterns"].append("Dataset files not found")
            analysis["prevention_strategies"].append("Implement better dataset validation before training")
        
        if "ValueError" in dataset_errors:
            analysis["common_patterns"].append("Invalid dataset format")
            analysis["prevention_strategies"].append("Add format validation and conversion utilities")
    
    # Model errors
    if category_errors["model"]:
        model_errors = [e["error_type"] for e in category_errors["model"]]
        if "RuntimeError" in model_errors:
            analysis["common_patterns"].append("Model runtime errors")
            analysis["prevention_strategies"].append("Add model validation and compatibility checks")
    
    # Training errors
    if category_errors["training"]:
        training_errors = [e["error_type"] for e in category_errors["training"]]
        if "CUDA out of memory" in str(training_errors):
            analysis["common_patterns"].append("GPU memory issues")
            analysis["prevention_strategies"].append("Implement automatic batch size adjustment")
            analysis["system_improvements"].append("Add memory monitoring and warnings")
    
    # Export errors
    if category_errors["export"]:
        export_errors = [e["error_type"] for e in category_errors["export"]]
        if "ONNX export failed" in str(export_errors):
            analysis["common_patterns"].append("Model export failures")
            analysis["prevention_strategies"].append("Add export compatibility checks")
    
    return analysis
```

**Learning Strategy:**
- **Pattern Recognition**: Identifies common error types
- **Prevention Planning**: Develops strategies to avoid future errors
- **System Improvement**: Suggests enhancements to prevent errors

## User-Friendly Error Messages

### **Error Message Design**

#### **Clear and Actionable Messages**
```python
def create_user_friendly_error_message(error: Exception, context: str) -> str:
    """Create user-friendly error message with actionable guidance."""
    
    error_type = type(error).__name__
    
    # Common error patterns and solutions
    error_guidance = {
        "FileNotFoundError": {
            "message": "File or directory not found",
            "common_causes": [
                "Incorrect file path",
                "Missing dataset directory",
                "File permissions issue"
            ],
            "solutions": [
                "Check that the file path is correct",
                "Ensure the dataset directory exists",
                "Verify file permissions"
            ]
        },
        "RuntimeError": {
            "message": "Runtime error occurred during execution",
            "common_causes": [
                "GPU out of memory",
                "Invalid model configuration",
                "Data format mismatch"
            ],
            "solutions": [
                "Reduce batch size or image size",
                "Check model configuration",
                "Verify data format compatibility"
            ]
        },
        "ValueError": {
            "message": "Invalid value or parameter",
            "common_causes": [
                "Invalid configuration parameters",
                "Data format issues",
                "Parameter out of range"
            ],
            "solutions": [
                "Review configuration parameters",
                "Check data format",
                "Ensure parameters are within valid ranges"
            ]
        },
        "ImportError": {
            "message": "Required package not available",
            "common_causes": [
                "Package not installed",
                "Version incompatibility",
                "Virtual environment not activated"
            ],
            "solutions": [
                "Install required package: pip install package_name",
                "Check package versions",
                "Activate virtual environment"
            ]
        }
    }
    
    # Get guidance for this error type
    guidance = error_guidance.get(error_type, {
        "message": "An unexpected error occurred",
        "common_causes": ["Unknown cause"],
        "solutions": ["Check the error details and system logs"]
    })
    
    # Build user-friendly message
    message = f"""
        Error: {guidance['message']}

        Context: {context}
        Error Details: {str(error)}

        Common Causes:
"""
    
    for cause in guidance["common_causes"]:
        message += f"   • {cause}\n"
    
            message += "\nSolutions:\n"
    for solution in guidance["solutions"]:
        message += f"   • {solution}\n"
    
    message += f"""
    For more help:
   • Check the documentation
   • Review system logs
   • Contact support with error details
"""
    
    return message
```

**Message Design Strategy:**
- **Clear Language**: Uses simple, non-technical terms
- **Actionable Solutions**: Provides specific steps to fix the problem
- **Context Information**: Explains what was happening when the error occurred

### **Troubleshooting Guides**

#### **Common Problem Solutions**
```python
def generate_troubleshooting_guide(error_type: str) -> str:
    """Generate troubleshooting guide for common errors."""
    
    guides = {
        "dataset_preparation": """
Dataset Preparation Issues

Common Problems:
1. Dataset not found
   - Ensure dataset folder exists in project root
   - Check folder permissions
   - Verify dataset structure

2. Invalid annotation format
   - Convert annotations to YOLO format
   - Use provided conversion utilities
   - Check annotation file syntax

3. Missing images or labels
   - Verify all images have corresponding labels
   - Check file extensions (.jpg, .png, .txt)
   - Ensure consistent naming convention

Solutions:
• Run dataset validation: python utils/validate_dataset.py
• Use auto-preparation: python utils/prepare_dataset.py
• Check dataset structure guide in documentation
""",
        
        "training_failures": """
Training Failures

Common Problems:
1. GPU out of memory
   - Reduce batch size
   - Reduce image size
   - Use gradient accumulation

2. Training not converging
   - Check learning rate
   - Verify data quality
   - Monitor loss curves

3. Slow training
   - Increase number of workers
   - Enable mixed precision
   - Use GPU if available

Solutions:
• Start with default parameters
• Monitor training progress
• Check system resources
• Review training logs
""",
        
        "model_export": """
Model Export Issues

Common Problems:
1. Export format not supported
   - Check supported formats
   - Install required dependencies
   - Verify model compatibility

2. Export fails
   - Check model format
   - Verify input/output shapes
   - Review export parameters

3. Exported model doesn't work
   - Test with sample input
   - Verify export settings
   - Check target platform compatibility

Solutions:
• Use supported export formats
• Test exported models
• Check export documentation
• Verify platform requirements
"""
    }
    
    return guides.get(error_type, "No troubleshooting guide available for this error type.")
```

**Troubleshooting Strategy:**
- **Problem Categorization**: Groups issues by system component
- **Step-by-Step Solutions**: Provides clear action steps
- **Prevention Tips**: Helps avoid future occurrences

## Best Practices for Error Handling

### **For Beginners**
1. **Read error messages carefully**: Error messages often contain the solution
2. **Check common causes first**: Start with the most likely problems
3. **Use validation tools**: Run system health checks before training
4. **Start with defaults**: Use default settings to avoid configuration errors

### **For Intermediate Users**
1. **Implement error handling**: Add try-catch blocks in custom code
2. **Monitor system resources**: Watch memory and disk usage
3. **Use checkpoints**: Enable automatic checkpointing for recovery
4. **Log errors**: Implement comprehensive error logging

### **For Advanced Users**
1. **Custom error handling**: Implement domain-specific error handling
2. **Automated recovery**: Build automatic recovery mechanisms
3. **Performance monitoring**: Implement comprehensive system monitoring
4. **Error prevention**: Design systems that prevent common errors

---

**Next**: We'll explore [Performance Optimization](04-performance-optimization.md) to understand how to maximize system efficiency and training speed.
