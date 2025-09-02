#!/usr/bin/env python3
"""
GPU Memory Management CLI Tool for YOLO Training System.
Provides command-line interface for GPU memory monitoring and cleanup.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.gpu_memory_manager import GPUMemoryManager, clear_gpu_memory, get_gpu_memory_stats, check_training_config


def format_memory_output(stats: Dict[str, Any], detailed: bool = False) -> str:
    """Format memory statistics for display."""
    if "error" in stats:
        return f"❌ {stats['error']}"
    
    gpu = stats["gpu"]
    system = stats["system"]
    
    output = []
    output.append("🖥️  GPU Memory Status")
    output.append("=" * 50)
    output.append(f"GPU: {gpu['name']}")
    output.append(f"Total Memory: {gpu['total_memory_gb']:.2f} GB")
    output.append(f"Used Memory: {gpu['reserved_gb']:.2f} GB ({gpu['reserved_pct']:.1f}%)")
    output.append(f"Free Memory: {gpu['free_gb']:.2f} GB")
    
    if detailed:
        output.append(f"Allocated: {gpu['allocated_gb']:.2f} GB ({gpu['allocated_pct']:.1f}%)")
        output.append(f"Peak Allocated: {gpu['max_allocated_gb']:.2f} GB")
        output.append(f"Peak Reserved: {gpu['max_reserved_gb']:.2f} GB")
    
    output.append("")
    output.append("💻 System Memory Status")
    output.append("-" * 30)
    output.append(f"Total RAM: {system['total_gb']:.2f} GB")
    output.append(f"Used RAM: {system['used_gb']:.2f} GB ({system['used_pct']:.1f}%)")
    output.append(f"Available RAM: {system['available_gb']:.2f} GB")
    
    return "\n".join(output)


def cmd_status(args):
    """Show current GPU memory status."""
    print("Checking GPU memory status...")
    stats = get_gpu_memory_stats()
    print(format_memory_output(stats, detailed=args.detailed))
    
    if args.json:
        print("\nJSON Output:")
        print(json.dumps(stats, indent=2))


def cmd_clear(args):
    """Clear GPU memory cache."""
    print("Clearing GPU memory cache...")
    
    result = clear_gpu_memory(aggressive=args.aggressive)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    freed_gb = result.get("freed_gb", 0)
    cleanup_type = result.get("cleanup_type", "unknown")
    
    print(f"✅ GPU memory cleanup completed ({cleanup_type})")
    print(f"💾 Freed: {freed_gb:.2f} GB")
    
    if args.show_after:
        print("\nMemory status after cleanup:")
        stats = result["memory_after"]
        print(format_memory_output(stats))
    
    if args.json:
        print("\nJSON Output:")
        print(json.dumps(result, indent=2))


def cmd_optimize(args):
    """Optimize GPU settings for training."""
    print("Optimizing GPU for training...")
    
    manager = GPUMemoryManager()
    
    if not manager.cuda_available:
        print("❌ CUDA not available")
        return
    
    result = manager.optimize_for_training(batch_size=args.batch_size)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print("✅ GPU optimization completed")
    
    # Show recommendations
    recs = result["recommendations"]
    print("\n📊 Batch Size Recommendations:")
    print("-" * 30)
    print(f"Available Memory: {recs['available_memory_gb']:.2f} GB")
    if args.batch_size:
        print(f"Current Batch Size: {recs['current_batch_size']}")
    print(f"Recommended: {recs['recommended_batch_size']}")
    print(f"Max Safe: {recs['max_safe_batch_size']}")
    
    # Show optimization notes
    print("\n📝 Notes:")
    for note in recs["optimization_notes"]:
        print(f"  • {note}")
    
    if args.json:
        print("\nJSON Output:")
        print(json.dumps(result, indent=2))


def cmd_check(args):
    """Check training configuration memory requirements."""
    # Get parameters from command line or prompt
    model_size = args.model
    image_size = args.image_size
    batch_size = args.batch_size
    model_version = getattr(args, 'version', None)
    
    if not model_version:
        model_version = input("YOLO version (yolov5/yolov8/yolo11) [default: yolov5]: ").strip().lower() or "yolov5"
    if not model_size:
        model_size = input("Model size (n/s/m/l/x): ").strip().lower()
    if not image_size:
        image_size = int(input("Image size (e.g., 640, 1280): "))
    if not batch_size:
        batch_size = int(input("Batch size (e.g., 8, 16): "))
    
    print(f"\n🔍 Training Configuration Analysis")
    print("=" * 50)
    print(f"Model: {model_version.upper()}{model_size.upper()}")
    print(f"Image Size: {image_size}px")
    print(f"Batch Size: {batch_size}")
    
    try:
        result = check_training_config(model_size, image_size, batch_size, model_version)
        
        print(f"\nModel Memory: {result['estimated_usage']['model_memory_gb']:.2f} GB")
        print(f"Batch Memory: {result['estimated_usage']['batch_memory_gb']:.2f} GB")
        print(f"Total Estimated: {result['estimated_usage']['total_estimated_gb']:.2f} GB")
        print(f"Available Memory: {result['gpu_status']['available_memory_gb']:.2f} GB")
        
        if result['gpu_status']['will_fit']:
            print(f"✅ Will Fit: {result['gpu_status']['utilization_pct']:.1f}% utilization")
        else:
            print(f"❌ Won't Fit: {result['gpu_status']['utilization_pct']:.1f}% over capacity")
            print(f"\n💡 Recommended batch size: {result['recommendations']['recommended_batch_size']}")
            
            if result['recommendations']['alternative_configs']:
                print(f"\n🔧 Alternative configurations:")
                for i, alt in enumerate(result['recommendations']['alternative_configs'], 1):
                    print(f"{i}. {alt['change']} (Est: {alt['estimated_memory_gb']:.2f} GB)")
        
        if args.json:
            print("\nJSON Output:")
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")


def cmd_profile(args):
    """Profile GPU memory usage by tensor types."""
    print("Profiling GPU memory usage...")
    
    manager = GPUMemoryManager()
    
    if not manager.cuda_available:
        print("❌ CUDA not available")
        return
    
    profile = manager.profile_memory_usage()
    
    if "error" in profile:
        print(f"❌ Error: {profile['error']}")
        return
    
    print(f"✅ Found {profile['total_tensors']} GPU tensors")
    
    if profile["tensor_breakdown"]:
        print("\n🔍 Top Memory Users:")
        print("-" * 50)
        for tensor_type, stats in list(profile["tensor_breakdown"].items())[:10]:
            print(f"{tensor_type:<20} {stats['total_gb']:.3f} GB ({stats['count']} tensors)")
    else:
        print("No GPU tensors found")
    
    # Show current memory status
    current = profile["current_memory"]
    print(f"\n📊 Current Usage: {current['gpu']['reserved_pct']:.1f}% of GPU memory")
    
    if args.json:
        print("\nJSON Output:")
        print(json.dumps(profile, indent=2))


def cmd_emergency(args):
    """Perform emergency GPU memory cleanup."""
    print("⚠️  Performing EMERGENCY GPU memory cleanup...")
    print("This may take a moment...")
    
    manager = GPUMemoryManager()
    
    if not manager.cuda_available:
        print("❌ CUDA not available")
        return
    
    result = manager.emergency_cleanup()
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    success = result.get("success", False)
    final_usage = result["final_memory_stats"]["gpu"]["reserved_pct"]
    
    if success:
        print("✅ Emergency cleanup successful!")
    else:
        print("⚠️  Emergency cleanup completed but memory usage still high")
    
    print(f"💾 Final memory usage: {final_usage:.1f}%")
    
    if args.verbose:
        print("\nCleanup passes:")
        for i, pass_result in enumerate(result["cleanup_passes"], 1):
            memory_pct = pass_result["memory_pct"]
            print(f"  Pass {i}: {memory_pct:.1f}% memory usage")
    
    if args.json:
        print("\nJSON Output:")
        print(json.dumps(result, indent=2))


def cmd_monitor(args):
    """Monitor GPU memory in real-time."""
    import time
    
    print("Starting GPU memory monitoring...")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    manager = GPUMemoryManager()
    
    if not manager.cuda_available:
        print("❌ CUDA not available")
        return
    
    try:
        while True:
            monitor_result = manager.monitor_training_memory(
                memory_threshold_pct=args.threshold
            )
            
            if "error" in monitor_result:
                print(f"❌ Error: {monitor_result['error']}")
                break
            
            stats = monitor_result["memory_stats"]
            warnings = monitor_result["warnings"]
            
            # Clear screen and show current status
            if not args.no_clear:
                print("\033[2J\033[H", end="")  # Clear screen
            
            print(f"🕐 {time.strftime('%H:%M:%S')}")
            print(format_memory_output(stats))
            
            if warnings:
                print("\n⚠️  Warnings:")
                for warning in warnings:
                    print(f"  • {warning}")
                
                recommendations = monitor_result["recommendations"]
                if recommendations:
                    print("\n💡 Recommendations:")
                    for rec in recommendations:
                        print(f"  • {rec}")
            
            print("-" * 60)
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="GPU Memory Management for YOLO Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                          # Show current GPU memory status
  %(prog)s clear --aggressive              # Aggressive memory cleanup  
  %(prog)s optimize --batch-size 16        # Optimize for batch size 16
  %(prog)s profile                         # Profile memory usage by tensors
  %(prog)s emergency                       # Emergency memory cleanup
  %(prog)s monitor --threshold 85          # Monitor with 85%% warning threshold
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show GPU memory status")
    status_parser.add_argument("--detailed", action="store_true", 
                              help="Show detailed memory statistics")
    status_parser.add_argument("--json", action="store_true", 
                              help="Output in JSON format")
    status_parser.set_defaults(func=cmd_status)
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear GPU memory cache")
    clear_parser.add_argument("--aggressive", action="store_true", 
                             help="Perform aggressive cleanup")
    clear_parser.add_argument("--show-after", action="store_true", 
                             help="Show memory status after cleanup")
    clear_parser.add_argument("--json", action="store_true", 
                             help="Output in JSON format")
    clear_parser.set_defaults(func=cmd_clear)
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize GPU for training")
    optimize_parser.add_argument("--batch-size", type=int, 
                                help="Current batch size for recommendations")
    optimize_parser.add_argument("--json", action="store_true", 
                                help="Output in JSON format")
    optimize_parser.set_defaults(func=cmd_optimize)
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check training configuration memory requirements")
    check_parser.add_argument("--model", type=str, help="Model size (n/s/m/l/x)")
    check_parser.add_argument("--image-size", type=int, help="Image size (e.g., 640, 1280)")
    check_parser.add_argument("--batch-size", type=int, help="Batch size (e.g., 8, 16)")
    check_parser.add_argument("--version", type=str, choices=["yolov5", "yolov8", "yolo11"], 
                             default="yolov5", help="YOLO version (default: yolov5)")
    check_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    check_parser.set_defaults(func=cmd_check)
    
    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Profile GPU memory usage")
    profile_parser.add_argument("--json", action="store_true", 
                               help="Output in JSON format")
    profile_parser.set_defaults(func=cmd_profile)
    
    # Emergency command
    emergency_parser = subparsers.add_parser("emergency", help="Emergency memory cleanup")
    emergency_parser.add_argument("--verbose", action="store_true", 
                                 help="Show detailed cleanup information")
    emergency_parser.add_argument("--json", action="store_true", 
                                 help="Output in JSON format")
    emergency_parser.set_defaults(func=cmd_emergency)
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor GPU memory in real-time")
    monitor_parser.add_argument("--interval", type=float, default=2.0, 
                               help="Update interval in seconds (default: 2.0)")
    monitor_parser.add_argument("--threshold", type=float, default=90.0, 
                               help="Warning threshold percentage (default: 90.0)")
    monitor_parser.add_argument("--no-clear", action="store_true", 
                               help="Don't clear screen between updates")
    monitor_parser.set_defaults(func=cmd_monitor)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run the selected command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
