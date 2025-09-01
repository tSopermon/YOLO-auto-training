#!/usr/bin/env python3
"""
Comprehensive TensorBoard management utility for YOLO training.
Handles launching, status checking, and management of TensorBoard instances.
"""

import subprocess
import socket
import webbrowser
import logging
import argparse
import sys
from pathlib import Path
from .tensorboard_launcher import launch_tensorboard

logger = logging.getLogger(__name__)


def check_tensorboard_running():
    """Check if TensorBoard is already running and return port if found."""
    try:
        # Check for running TensorBoard processes
        result = subprocess.run(["pgrep", "-f", "tensorboard"], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Try to find the port
            common_ports = [6006, 6007, 6008, 6009]
            for port in common_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()
                    if result == 0:
                        return port
                except:
                    continue
            return 6006  # Default assumption
        return None
    except:
        return None


def stop_tensorboard():
    """Stop all running TensorBoard processes."""
    try:
        print("🛑 Stopping TensorBoard...")
        subprocess.run(["pkill", "-f", "tensorboard"])
        print("✅ TensorBoard stopped")
        return True
    except Exception as e:
        print(f"❌ Error stopping TensorBoard: {e}")
        return False


def status_and_open():
    """Check TensorBoard status and open in browser if running."""
    port = check_tensorboard_running()
    if port:
        print(f"✅ TensorBoard is running at: http://localhost:{port}")
        webbrowser.open(f"http://localhost:{port}")
        return port
    else:
        print("❌ TensorBoard is not running")
        return None


def list_experiments():
    """List all available experiments and their TensorBoard data status."""
    logs_dir = Path("logs")
    
    print("Available experiments:")
    print("=" * 50)
    
    if not logs_dir.exists():
        print("No logs directory found. Run some training first!")
        return []
        
    experiments = [d for d in logs_dir.iterdir() if d.is_dir()]
    if not experiments:
        print("No experiments found in logs directory.")
        return []
        
    experiment_list = []
    for exp in sorted(experiments):
        # Check if it has TensorBoard data
        tensorboard_files = list(exp.rglob("events.out.tfevents*"))
        status = "✅ Has TensorBoard data" if tensorboard_files else "❌ No TensorBoard data"
        print(f"  {exp.name:<20} {status}")
        experiment_list.append((exp.name, bool(tensorboard_files)))
        
    return experiment_list


def launch_experiment_tensorboard(experiment_name, port=6006):
    """Launch TensorBoard for a specific experiment."""
    logs_dir = Path("logs")
    experiment_path = logs_dir / experiment_name
    
    if not experiment_path.exists():
        print(f"❌ Experiment '{experiment_name}' not found in logs directory.")
        experiments = [d.name for d in logs_dir.iterdir() if d.is_dir()]
        if experiments:
            print("Available experiments:")
            for exp in experiments:
                print(f"  {exp}")
        return None
    
    print(f"🚀 Launching TensorBoard for experiment: {experiment_name}")
    print(f"📁 Looking in: {experiment_path}")
    
    # Launch TensorBoard (it will automatically find the correct subdirectory)
    process = launch_tensorboard(experiment_path, port=port, wait_for_logs=False)
    
    if process:
        print(f"🌐 TensorBoard is running at: http://localhost:{port}")
        return process
    else:
        print("❌ Failed to start TensorBoard")
        return None


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="TensorBoard Manager for YOLO training results")
    parser.add_argument(
        "command",
        nargs='?',
        choices=['status', 'stop', 'list', 'launch'],
        help="Command to execute: status, stop, list, or launch"
    )
    parser.add_argument(
        "experiment", 
        nargs='?',
        help="Experiment folder name for launch command (e.g., '22', 'experiment-1')"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6006, 
        help="Port to run TensorBoard on (default: 6006)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        # Default behavior: list experiments
        experiments = list_experiments()
        if experiments:
            print(f"\nUsage examples:")
            print(f"  python -m utils.tensorboard_manager status    # Check if running")
            print(f"  python -m utils.tensorboard_manager launch 22 # Launch experiment 22")
            print(f"  python -m utils.tensorboard_manager stop      # Stop TensorBoard")
        return
    
    if args.command == "status":
        status_and_open()
    elif args.command == "stop":
        stop_tensorboard()
    elif args.command == "list":
        list_experiments()
    elif args.command == "launch":
        if not args.experiment:
            print("❌ Please specify an experiment name for launch command")
            print("Usage: python -m utils.tensorboard_manager launch <experiment_name>")
            return
        process = launch_experiment_tensorboard(args.experiment, args.port)
        if process:
            print("Press Ctrl+C to stop TensorBoard")
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping TensorBoard...")
                process.terminate()
                print("✅ TensorBoard stopped")


if __name__ == "__main__":
    main()
