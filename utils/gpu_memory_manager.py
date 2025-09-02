"""
GPU Memory Management utilities for YOLO training system.
Provides comprehensive GPU memory cleanup, monitoring, and optimization.
"""

import logging
import gc
import time
import psutil
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager
from pathlib import Path
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """Comprehensive GPU memory management for YOLO training."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize GPU memory manager.
        
        Args:
            device: GPU device to manage (e.g., 'cuda:0'). Auto-detects if None.
        """
        self.device = device
        self.cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
        
        # YOLO model parameter counts (approximate millions of parameters)
        self.YOLO_PARAMETERS = {
            "yolov5": {
                "n": 1.9,   # YOLOv5n: ~1.9M parameters
                "s": 7.2,   # YOLOv5s: ~7.2M parameters  
                "m": 21.2,  # YOLOv5m: ~21.2M parameters
                "l": 46.5,  # YOLOv5l: ~46.5M parameters
                "x": 86.7   # YOLOv5x: ~86.7M parameters
            },
            "yolov8": {
                "n": 3.2,   # YOLOv8n: ~3.2M parameters
                "s": 11.2,  # YOLOv8s: ~11.2M parameters
                "m": 25.9,  # YOLOv8m: ~25.9M parameters
                "l": 43.7,  # YOLOv8l: ~43.7M parameters
                "x": 68.2   # YOLOv8x: ~68.2M parameters
            },
            "yolo11": {
                "n": 2.6,   # YOLO11n: ~2.6M parameters
                "s": 9.4,   # YOLO11s: ~9.4M parameters
                "m": 20.1,  # YOLO11m: ~20.1M parameters
                "l": 25.3,  # YOLO11l: ~25.3M parameters
                "x": 56.9   # YOLO11x: ~56.9M parameters
            }
        }
        
        if self.cuda_available:
            if device is None:
                self.device = f"cuda:{torch.cuda.current_device()}"
            self.device_id = int(self.device.split(':')[-1]) if ':' in self.device else 0
            
            # Get GPU info
            self.gpu_name = torch.cuda.get_device_name(self.device_id)
            self.total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
            
            logger.info(f"GPU Memory Manager initialized for {self.gpu_name}")
            logger.info(f"Total GPU Memory: {self.total_memory / 1e9:.2f} GB")
        else:
            logger.warning("CUDA not available. GPU memory management disabled.")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        if not self.cuda_available:
            return {"error": "CUDA not available"}
        
        # GPU memory stats
        allocated = torch.cuda.memory_allocated(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        max_allocated = torch.cuda.max_memory_allocated(self.device_id)
        max_reserved = torch.cuda.max_memory_reserved(self.device_id)
        
        # Calculate percentages
        allocated_pct = (allocated / self.total_memory) * 100
        reserved_pct = (reserved / self.total_memory) * 100
        
        # System memory stats
        system_memory = psutil.virtual_memory()
        
        return {
            "gpu": {
                "device": self.device,
                "name": self.gpu_name,
                "total_memory_gb": self.total_memory / 1e9,
                "allocated_gb": allocated / 1e9,
                "reserved_gb": reserved / 1e9,
                "free_gb": (self.total_memory - reserved) / 1e9,
                "allocated_pct": allocated_pct,
                "reserved_pct": reserved_pct,
                "max_allocated_gb": max_allocated / 1e9,
                "max_reserved_gb": max_reserved / 1e9,
            },
            "system": {
                "total_gb": system_memory.total / 1e9,
                "available_gb": system_memory.available / 1e9,
                "used_gb": system_memory.used / 1e9,
                "used_pct": system_memory.percent
            }
        }
    
    def clear_gpu_cache(self, aggressive: bool = True) -> Dict[str, Any]:
        """
        Clear GPU memory cache.
        
        Args:
            aggressive: If True, performs more thorough cleanup
            
        Returns:
            Dictionary with cleanup results
        """
        if not self.cuda_available:
            return {"error": "CUDA not available"}
        
        # Get memory before cleanup
        memory_before = self.get_memory_stats()
        
        logger.info("Starting GPU memory cleanup...")
        
        # Step 1: Python garbage collection
        if aggressive:
            logger.info("Running aggressive Python garbage collection...")
            for _ in range(3):  # Multiple passes for thorough cleanup
                collected = gc.collect()
                logger.debug(f"Collected {collected} objects")
        else:
            gc.collect()
        
        # Step 2: Clear PyTorch cache
        logger.info("Clearing PyTorch GPU cache...")
        torch.cuda.empty_cache()
        
        # Step 3: Reset peak memory stats (if aggressive)
        if aggressive:
            logger.info("Resetting GPU memory statistics...")
            torch.cuda.reset_peak_memory_stats(self.device_id)
            torch.cuda.reset_accumulated_memory_stats(self.device_id)
        
        # Step 4: Synchronize GPU operations
        torch.cuda.synchronize(self.device_id)
        
        # Give GPU a moment to fully clear
        time.sleep(0.5)
        
        # Get memory after cleanup
        memory_after = self.get_memory_stats()
        
        # Calculate freed memory
        freed_gb = memory_before["gpu"]["reserved_gb"] - memory_after["gpu"]["reserved_gb"]
        
        result = {
            "cleanup_type": "aggressive" if aggressive else "standard",
            "memory_before": memory_before,
            "memory_after": memory_after,
            "freed_gb": freed_gb,
            "freed_pct": (freed_gb / (memory_before["gpu"]["total_memory_gb"])) * 100,
            "success": True
        }
        
        logger.info(f"GPU memory cleanup completed. Freed: {freed_gb:.2f} GB")
        return result
    
    def profile_memory_usage(self) -> Dict[str, Any]:
        """
        Profile current memory usage by tensor types.
        
        Returns:
            Dictionary with detailed memory profiling
        """
        if not self.cuda_available:
            return {"error": "CUDA not available"}
        
        logger.info("Profiling GPU memory usage...")
        
        # Get memory breakdown by tensor
        tensor_memory = {}
        total_tensors = 0
        
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda and obj.device.index == self.device_id:
                tensor_type = type(obj).__name__
                tensor_size = obj.element_size() * obj.nelement()
                
                if tensor_type not in tensor_memory:
                    tensor_memory[tensor_type] = {
                        "count": 0,
                        "total_bytes": 0,
                        "avg_size_mb": 0
                    }
                
                tensor_memory[tensor_type]["count"] += 1
                tensor_memory[tensor_type]["total_bytes"] += tensor_size
                total_tensors += 1
        
        # Calculate averages and sort by memory usage
        for tensor_type, stats in tensor_memory.items():
            stats["total_gb"] = stats["total_bytes"] / 1e9
            stats["avg_size_mb"] = stats["total_bytes"] / (stats["count"] * 1e6)
        
        # Sort by total memory usage
        sorted_tensors = sorted(
            tensor_memory.items(), 
            key=lambda x: x[1]["total_bytes"], 
            reverse=True
        )
        
        # Get current memory stats
        current_stats = self.get_memory_stats()
        
        return {
            "total_tensors": total_tensors,
            "tensor_breakdown": dict(sorted_tensors[:10]),  # Top 10 memory users
            "current_memory": current_stats,
            "profiling_time": time.time()
        }
    
    def optimize_for_training(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize GPU settings for training.
        
        Args:
            batch_size: Current batch size (for optimization recommendations)
            
        Returns:
            Dictionary with optimization results and recommendations
        """
        if not self.cuda_available:
            return {"error": "CUDA not available"}
        
        logger.info("Optimizing GPU settings for training...")
        
        # Clear memory first
        cleanup_result = self.clear_gpu_cache(aggressive=True)
        
        # Set optimal CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory management
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.95, self.device_id)
        
        # Environment variables for memory optimization
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        
        # Get memory stats after optimization
        memory_stats = self.get_memory_stats()
        available_memory_gb = memory_stats["gpu"]["free_gb"]
        
        # Generate batch size recommendations
        recommendations = self._generate_batch_size_recommendations(
            available_memory_gb, batch_size
        )
        
        result = {
            "optimization_applied": True,
            "settings": {
                "cudnn_benchmark": True,
                "cudnn_deterministic": False,
                "tf32_enabled": True,
                "memory_fraction": 0.95,
            },
            "memory_stats": memory_stats,
            "recommendations": recommendations,
            "cleanup_result": cleanup_result
        }
        
        logger.info("GPU optimization completed successfully")
        return result
    
    def _generate_batch_size_recommendations(
        self, available_memory_gb: float, current_batch_size: Optional[int]
    ) -> Dict[str, Any]:
        """Generate batch size recommendations based on available memory."""
        
        # GPU-specific recommendations based on RTX 4070 (12GB)
        if "RTX 4070" in self.gpu_name:
            if available_memory_gb >= 10:
                recommended_batch = 16  # More conservative for large models
                max_batch = 32
            elif available_memory_gb >= 8:
                recommended_batch = 12
                max_batch = 16
            elif available_memory_gb >= 6:
                recommended_batch = 8
                max_batch = 12
            elif available_memory_gb >= 4:
                recommended_batch = 6
                max_batch = 8
            else:
                recommended_batch = 4
                max_batch = 6
        else:
            # Generic recommendations
            memory_per_batch_estimate = 0.5  # GB per batch (more conservative estimate)
            recommended_batch = max(2, int(available_memory_gb / memory_per_batch_estimate))
            max_batch = int(recommended_batch * 1.2)
        
        return {
            "available_memory_gb": available_memory_gb,
            "current_batch_size": current_batch_size,
            "recommended_batch_size": recommended_batch,
            "max_safe_batch_size": max_batch,
            "memory_per_batch_estimate_gb": 0.5,
            "optimization_notes": [
                f"GPU: {self.gpu_name}",
                f"Available memory: {available_memory_gb:.2f} GB",
                "Recommendations are for large models (YOLOv5l, YOLOv8l)",
                "For nano/small models, you can use 2-4x larger batch sizes",
                "High-resolution images (1280px+) require smaller batches",
                "Test batch sizes incrementally to find optimal settings"
            ]
        }
    
    @contextmanager
    def training_memory_context(self):
        """
        Context manager for training with automatic memory cleanup.
        Use this to wrap training loops for automatic memory management.
        """
        try:
            # Setup before training
            logger.info("Setting up training memory context...")
            self.optimize_for_training()
            
            yield self
            
        except Exception as e:
            logger.error(f"Error in training context: {e}")
            raise
        
        finally:
            # Cleanup after training
            logger.info("Cleaning up training memory context...")
            self.clear_gpu_cache(aggressive=True)
    
    def monitor_training_memory(
        self, 
        log_interval: int = 10,
        memory_threshold_pct: float = 90.0
    ) -> Dict[str, Any]:
        """
        Monitor memory during training and provide warnings.
        
        Args:
            log_interval: Log memory stats every N calls
            memory_threshold_pct: Warn when memory usage exceeds this percentage
            
        Returns:
            Current memory status and warnings
        """
        if not self.cuda_available:
            return {"error": "CUDA not available"}
        
        stats = self.get_memory_stats()
        
        warnings = []
        recommendations = []
        
        # Check memory usage
        if stats["gpu"]["reserved_pct"] > memory_threshold_pct:
            warnings.append(f"High GPU memory usage: {stats['gpu']['reserved_pct']:.1f}%")
            recommendations.append("Consider reducing batch size or clearing cache")
        
        if stats["gpu"]["reserved_pct"] > 95:
            warnings.append("Critical GPU memory usage - OOM risk!")
            recommendations.append("Immediate memory cleanup recommended")
        
        # Check system memory
        if stats["system"]["used_pct"] > 85:
            warnings.append(f"High system memory usage: {stats['system']['used_pct']:.1f}%")
            recommendations.append("Consider reducing data loader workers")
        
        return {
            "memory_stats": stats,
            "warnings": warnings,
            "recommendations": recommendations,
            "timestamp": time.time()
        }
    
    def estimate_training_memory_usage(
        self, 
        model_size: str = "l", 
        image_size: int = 640, 
        batch_size: int = 8,
        model_version: str = "yolov5"
    ) -> Dict[str, Any]:
        """
        Estimate memory usage for training configuration with enhanced safety analysis.
        
        Args:
            model_size: Model size (n, s, m, l, x)
            image_size: Input image size
            batch_size: Training batch size
            model_version: YOLO version (yolov5, yolov8, yolo11)
            
        Returns:
            Memory usage estimation with safety warnings and recommendations
        """
        if not self.cuda_available:
            return {"error": "CUDA not available"}
        
        # Use the updated model memory estimation with correction factors
        model_memory = self._estimate_model_memory(model_version, model_size)
        batch_memory = self._estimate_batch_memory(image_size, batch_size)
        
        # Additional training overhead (optimizer states, loss computation, etc.)
        training_overhead = (model_memory + batch_memory) * 0.3
        
        # Total estimated memory with safety margin
        total_estimated = model_memory + batch_memory + training_overhead
        total_with_margin = total_estimated * 1.2  # 20% safety buffer
        
        # Get current available memory
        stats = self.get_memory_stats()
        available_memory = stats["gpu"]["free_gb"]
        
        # Calculate memory usage ratio
        memory_ratio = total_with_margin / available_memory if available_memory > 0 else float('inf')
        
        # Determine safety level and generate warnings
        safety_level = "safe"
        warnings = []
        recommendations = []
        
        if memory_ratio > 0.95:
            safety_level = "critical"
            warnings.append("⚠️ CRITICAL: Predicted to exceed GPU memory - training will likely fail")
        elif memory_ratio > 0.85:
            safety_level = "high_risk"
            warnings.append("⚠️ HIGH RISK: Very close to memory limit - consider reducing parameters")
        elif memory_ratio > 0.75:
            safety_level = "moderate_risk"
            warnings.append("⚠️ MODERATE RISK: Close to memory limit - monitor for OOM errors")
        elif memory_ratio > 0.6:
            safety_level = "low_risk"
            warnings.append("ℹ️ LOW RISK: Should work but monitor memory usage")
        
        # Known problematic configuration warnings
        if model_size.lower() in ['l', 'x'] and batch_size >= 4:
            warnings.append("⚠️ LARGE MODEL + HIGH BATCH: This combination often fails - start with batch_size=1 or 2")
        
        if image_size >= 1280 and batch_size >= 8:
            warnings.append("⚠️ HIGH RESOLUTION + HIGH BATCH: Memory usage grows exponentially")
        
        if model_size.lower() in ['l', 'x'] and image_size >= 1280:
            warnings.append("⚠️ LARGE MODEL + HIGH RESOLUTION: Consider starting with 640px or 1024px")
        
        # Generate recommendations for high-risk configurations
        if safety_level in ['critical', 'high_risk']:
            if batch_size > 1:
                recommendations.append(f"Reduce batch_size from {batch_size} to {max(1, batch_size // 2)}")
            if image_size > 640:
                new_size = 640 if image_size > 1024 else 1024
                recommendations.append(f"Reduce image_size from {image_size} to {new_size}")
            if model_size.lower() in ['l', 'x']:
                smaller_size = 'm' if model_size.lower() == 'l' else 'l'
                recommendations.append(f"Try smaller model: {model_version}{smaller_size} instead of {model_version}{model_size}")
        
        # Calculate recommended batch size for safe operation
        safe_memory_limit = available_memory * 0.8  # 80% threshold for safety
        safe_batch_memory = safe_memory_limit - model_memory - training_overhead
        recommended_batch = max(1, int(safe_batch_memory / (batch_memory / batch_size)))
        
        will_fit = safety_level != "critical"
        should_proceed = safety_level in ["safe", "low_risk"]
        
        return {
            "model_version": model_version,
            "model_size": model_size,
            "image_size": image_size,
            "batch_size": batch_size,
            "estimated_usage": {
                "model_memory_gb": round(model_memory, 2),
                "batch_memory_gb": round(batch_memory, 2),
                "training_overhead_gb": round(training_overhead, 2),
                "total_estimated_gb": round(total_estimated, 2),
                "total_with_margin_gb": round(total_with_margin, 2),
                "memory_per_item_gb": round(batch_memory / batch_size, 3)
            },
            "gpu_status": {
                "available_memory_gb": available_memory,
                "memory_ratio": round(memory_ratio, 2),
                "utilization_pct": round((total_with_margin / available_memory) * 100, 1),
                "will_fit": will_fit,
                "should_proceed": should_proceed
            },
            "safety_analysis": {
                "safety_level": safety_level,
                "warnings": warnings,
                "recommendations": recommendations
            },
            "recommendations": {
                "recommended_batch_size": min(recommended_batch, batch_size),
                "alternative_configs": self._get_alternative_configs(
                    model_size, image_size, available_memory, model_version
                )
            }
        }
    
    def _get_alternative_configs(
        self, model_size: str, image_size: int, available_memory_gb: float, model_version: str = "yolov5"
    ) -> List[Dict[str, Any]]:
        """Get alternative configuration suggestions."""
        alternatives = []
        
        # Alternative 1: Reduce batch size
        for batch in [6, 4, 2, 1]:
            # Calculate directly without recursion
            model_memory = self._estimate_model_memory(model_version, model_size)
            batch_memory = self._estimate_batch_memory(image_size, batch)
            total_estimated = (model_memory + batch_memory) * 1.3  # Safety margin
            
            if total_estimated <= available_memory_gb:
                alternatives.append({
                    "change": f"Reduce batch size to {batch}",
                    "config": {"batch_size": batch, "image_size": image_size, "model_size": model_size},
                    "estimated_memory_gb": total_estimated
                })
                break
        
        # Alternative 2: Reduce image size
        for img_size in [1024, 896, 768, 640]:
            if img_size < image_size:
                # Calculate directly without recursion
                model_memory = self._estimate_model_memory(model_version, model_size)
                batch_memory = self._estimate_batch_memory(img_size, 8)
                total_estimated = (model_memory + batch_memory) * 1.3  # Safety margin
                
                if total_estimated <= available_memory_gb:
                    alternatives.append({
                        "change": f"Reduce image size to {img_size}px",
                        "config": {"batch_size": 8, "image_size": img_size, "model_size": model_size},
                        "estimated_memory_gb": total_estimated
                    })
                    break
        
        # Alternative 3: Use smaller model
        for smaller_model in ["m", "s", "n"]:
            if smaller_model < model_size:
                # Calculate directly without recursion
                model_memory = self._estimate_model_memory(model_version, smaller_model)
                batch_memory = self._estimate_batch_memory(image_size, 8)
                total_estimated = (model_memory + batch_memory) * 1.3  # Safety margin
                
                if total_estimated <= available_memory_gb:
                    alternatives.append({
                        "change": f"Use {smaller_model.upper()} model instead of {model_size.upper()}",
                        "config": {"batch_size": 8, "image_size": image_size, "model_size": smaller_model},
                        "estimated_memory_gb": total_estimated
                    })
                    break
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def _get_model_parameters(self, version: str, size: str) -> float:
        """
        Get parameter count for specified YOLO model.
        
        Args:
            version: YOLO version (yolov5, yolov8, yolo11)
            size: Model size (n, s, m, l, x)
            
        Returns:
            Parameter count in millions
        """
        version_key = version.lower()
        size_key = size.lower()
        
        if version_key not in self.YOLO_PARAMETERS:
            logger.warning(f"Unknown YOLO version: {version}. Using yolov8 defaults.")
            version_key = "yolov8"
        
        if size_key not in self.YOLO_PARAMETERS[version_key]:
            logger.warning(f"Unknown model size: {size}. Using 'm' defaults.")
            size_key = "m"
        
        params_millions = self.YOLO_PARAMETERS[version_key][size_key]
        return params_millions * 1_000_000  # Convert to actual parameter count
    
    def _estimate_model_memory(self, version: str, size: str) -> float:
        """Estimate memory required for model weights and architecture with correction factors."""
        params = self._get_model_parameters(version, size)
        
        # Base memory for model weights (4 bytes per parameter for float32)
        weights_memory = params * 4
        
        # Additional memory for model architecture, buffers, etc.
        architecture_overhead = weights_memory * 0.5
        
        # Convert to GB
        base_memory = (weights_memory + architecture_overhead) / (1024**3)
        
        # Apply size-specific correction factors based on real training data
        # These factors account for actual memory usage during training
        correction_factors = {
            'n': 4.0,   # nano models: 4x multiplier (conservative for small models)
            's': 4.5,   # small models: 4.5x multiplier  
            'm': 5.0,   # medium models: 5x multiplier
            'l': 6.5,   # large models: 6.5x multiplier (average of 5.8x and 7.5x observed)
            'x': 7.0    # extra large models: 7x multiplier (conservative estimate)
        }
        
        size_key = size.lower()
        correction_factor = correction_factors.get(size_key, 5.0)  # Default to 5x if size not found
        
        corrected_memory = base_memory * correction_factor
        
        return corrected_memory
    
    def _estimate_batch_memory(self, image_size: int, batch_size: int) -> float:
        """Estimate batch memory usage in GB with correction factors."""
        # Each image: channels(3) * height * width * bytes_per_pixel(4 for FP32)
        single_image_mb = (3 * image_size * image_size * 4) / (1024 * 1024)
        
        # Base calculation: input images + feature maps + gradients + activations
        base_multiplier = 12  # Conservative base multiplier
        batch_memory_mb = single_image_mb * batch_size * base_multiplier
        
        # Apply batch-size specific correction factors based on real training data
        if batch_size >= 8:
            batch_correction = 2.8  # Large batches have exponential memory growth
        elif batch_size >= 4:
            batch_correction = 2.3  # Medium batches need significant correction
        elif batch_size >= 2:
            batch_correction = 1.9  # Small batches still need correction
        else:
            batch_correction = 1.6  # Even batch size 1 needs some correction
        
        # Image size also affects memory usage significantly
        if image_size >= 1280:
            img_size_correction = 2.2  # High-res images need much more memory
        elif image_size >= 1024:
            img_size_correction = 1.8  # Medium-res needs significant correction
        elif image_size >= 640:
            img_size_correction = 1.4  # Standard resolution needs some correction
        else:
            img_size_correction = 1.2  # Lower resolutions minimal correction
        
        # Combine corrections
        total_correction = batch_correction * img_size_correction
        corrected_memory_mb = batch_memory_mb * total_correction
        
        return corrected_memory_mb / 1024  # Convert to GB
    
    def emergency_cleanup(self) -> Dict[str, Any]:
        """
        Emergency GPU memory cleanup for out-of-memory situations.
        
        Returns:
            Results of emergency cleanup
        """
        if not self.cuda_available:
            return {"error": "CUDA not available"}
        
        logger.warning("Performing emergency GPU memory cleanup...")
        
        # Multiple aggressive cleanup passes
        results = []
        for i in range(3):
            logger.info(f"Emergency cleanup pass {i+1}/3...")
            
            # Aggressive garbage collection
            for _ in range(5):
                gc.collect()
            
            # Clear all possible caches
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Reset all memory stats
            torch.cuda.reset_peak_memory_stats(self.device_id)
            torch.cuda.reset_accumulated_memory_stats(self.device_id)
            
            # Synchronize
            torch.cuda.synchronize(self.device_id)
            
            # Wait between passes
            time.sleep(1)
            
            # Check memory
            stats = self.get_memory_stats()
            results.append({
                "pass": i + 1,
                "memory_after_gb": stats["gpu"]["reserved_gb"],
                "memory_pct": stats["gpu"]["reserved_pct"]
            })
        
        final_stats = self.get_memory_stats()
        
        logger.info(f"Emergency cleanup completed. Final memory usage: {final_stats['gpu']['reserved_pct']:.1f}%")
        
        return {
            "emergency_cleanup": True,
            "cleanup_passes": results,
            "final_memory_stats": final_stats,
            "success": final_stats["gpu"]["reserved_pct"] < 80
        }


# Convenience functions for easy integration
def check_training_config(
    model_size: str = "l", 
    image_size: int = 640, 
    batch_size: int = 8,
    model_version: str = "yolov5"
) -> Dict[str, Any]:
    """
    Check if a training configuration will fit in GPU memory.
    
    Args:
        model_size: Model size (n, s, m, l, x)
        image_size: Input image size
        batch_size: Training batch size
        model_version: YOLO version (yolov5, yolov8, yolo11)
        
    Returns:
        Configuration analysis and recommendations
    """
    manager = GPUMemoryManager()
    return manager.estimate_training_memory_usage(model_size, image_size, batch_size, model_version)


def clear_gpu_memory(aggressive: bool = True) -> Dict[str, Any]:
    """
    Convenience function to clear GPU memory.
    
    Args:
        aggressive: Whether to perform aggressive cleanup
        
    Returns:
        Cleanup results
    """
    manager = GPUMemoryManager()
    return manager.clear_gpu_cache(aggressive=aggressive)


def get_gpu_memory_stats() -> Dict[str, Any]:
    """
    Convenience function to get GPU memory statistics.
    
    Returns:
        Memory statistics
    """
    manager = GPUMemoryManager()
    return manager.get_memory_stats()


def optimize_gpu_for_training(batch_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function to optimize GPU for training.
    
    Args:
        batch_size: Current batch size
        
    Returns:
        Optimization results
    """
    manager = GPUMemoryManager()
    return manager.optimize_for_training(batch_size=batch_size)


def emergency_gpu_cleanup() -> Dict[str, Any]:
    """
    Convenience function for emergency GPU cleanup.
    
    Returns:
        Emergency cleanup results
    """
    manager = GPUMemoryManager()
    return manager.emergency_cleanup()


# Training integration decorators
def with_gpu_cleanup(aggressive: bool = True):
    """
    Decorator to automatically clean GPU memory after function execution.
    
    Args:
        aggressive: Whether to perform aggressive cleanup
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                clear_gpu_memory(aggressive=aggressive)
        return wrapper
    return decorator


def with_memory_monitoring(threshold_pct: float = 90.0):
    """
    Decorator to monitor memory usage during function execution.
    
    Args:
        threshold_pct: Memory threshold for warnings
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = GPUMemoryManager()
            
            # Log initial memory state
            initial_stats = manager.get_memory_stats()
            logger.info(f"Starting function with GPU memory: {initial_stats['gpu']['reserved_pct']:.1f}%")
            
            try:
                result = func(*args, **kwargs)
                
                # Monitor final memory state
                final_stats = manager.get_memory_stats()
                if final_stats["gpu"]["reserved_pct"] > threshold_pct:
                    logger.warning(f"High memory usage after function: {final_stats['gpu']['reserved_pct']:.1f}%")
                
                return result
                
            except Exception as e:
                # Check if it's a memory error
                if "out of memory" in str(e).lower():
                    logger.error("Out of memory error detected. Performing emergency cleanup...")
                    manager.emergency_cleanup()
                raise
                
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo/test the GPU memory manager
    print("GPU Memory Manager Demo")
    print("=" * 50)
    
    manager = GPUMemoryManager()
    
    if manager.cuda_available:
        # Show initial stats
        print("\nInitial Memory Stats:")
        stats = manager.get_memory_stats()
        print(f"GPU: {stats['gpu']['name']}")
        print(f"Total Memory: {stats['gpu']['total_memory_gb']:.2f} GB")
        print(f"Used Memory: {stats['gpu']['reserved_gb']:.2f} GB ({stats['gpu']['reserved_pct']:.1f}%)")
        
        # Clear memory
        print("\nClearing GPU memory...")
        cleanup = manager.clear_gpu_cache(aggressive=True)
        print(f"Freed: {cleanup['freed_gb']:.2f} GB")
        
        # Show optimization recommendations
        print("\nOptimization Recommendations:")
        opt_result = manager.optimize_for_training()
        recs = opt_result["recommendations"]
        print(f"Recommended batch size: {recs['recommended_batch_size']}")
        print(f"Max safe batch size: {recs['max_safe_batch_size']}")
        
    else:
        print("CUDA not available - GPU memory management disabled")
