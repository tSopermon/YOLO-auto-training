#!/usr/bin/env python3
"""
System Capabilities Demo - YOLO Auto-Training System
=====================================================

This script demonstrates all available functionalities in the YOLO auto-training system.
Run with different options to see what the system can do.

Usage:
    python examples/system_capabilities_demo.py --help
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def get_python_executable():
    """Get the correct Python executable path."""
    venv_python = Path(__file__).parent.parent / '.venv' / 'bin' / 'python'
    if venv_python.exists():
        return str(venv_python)
    return sys.executable

def run_command(cmd, description):
    """Run a command and show the output."""
    print(f"\n🔹 {description}")
    print("=" * 60)
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:", result.stderr)
        print(f"Return code: {result.returncode}")
    except Exception as e:
        print(f"Error running command: {e}")

def main():
    parser = argparse.ArgumentParser(description="YOLO Auto-Training System Capabilities Demo")
    parser.add_argument('--demo', choices=[
        'config', 'dataset', 'weights', 'training', 'export', 'tensorboard', 'tests', 'all'
    ], help='Which demo to run')
    parser.add_argument('--list-only', action='store_true', help='Just list capabilities without running commands')
    
    args = parser.parse_args()
    
    python_exe = get_python_executable()
    
    print("🚀 YOLO Auto-Training System - Capabilities Demo")
    print("=" * 80)
    print(f"Python executable: {python_exe}")
    print(f"Working directory: {Path(__file__).parent.parent}")
    
    demos = {
        'config': {
            'description': 'Configuration System',
            'commands': [
                f"{python_exe} -c \"from config.config import Config; c=Config(); print('✅ Config loaded successfully')\"",
                f"ls -la config/",
            ]
        },
        'dataset': {
            'description': 'Dataset Management',
            'commands': [
                f"{python_exe} utils/prepare_dataset.py --help",
                f"{python_exe} examples/create_test_dataset.py --help",
                f"ls -la dataset/",
            ]
        },
        'weights': {
            'description': 'Pretrained Weights Management',
            'commands': [
                f"{python_exe} utils/download_pretrained_weights.py --list",
                f"{python_exe} utils/download_pretrained_weights.py --downloaded",
                f"ls -la pretrained_weights/",
            ]
        },
        'training': {
            'description': 'Training System',
            'commands': [
                f"{python_exe} train.py --help",
                f"{python_exe} examples/demo_interactive_training.py --help",
                f"ls -la logs/",
            ]
        },
        'export': {
            'description': 'Model Export System',
            'commands': [
                f"{python_exe} utils/export_existing_models.py --help",
                f"ls -la exported_models/",
            ]
        },
        'tensorboard': {
            'description': 'TensorBoard Integration',
            'commands': [
                f"{python_exe} utils/tensorboard_launcher.py --help",
                f"{python_exe} -c \"from utils.tensorboard_manager import TensorBoardManager; print('✅ TensorBoard manager available')\"",
            ]
        },
        'tests': {
            'description': 'Testing Framework',
            'commands': [
                f"{python_exe} -m pytest tests/ --version",
                f"ls -la tests/",
                f"{python_exe} tests/run_tests.py --help",
            ]
        }
    }
    
    if args.list_only:
        print("\n📋 Available System Capabilities:")
        print("=" * 50)
        for name, demo in demos.items():
            print(f"  🔸 {name}: {demo['description']}")
        print(f"\nUsage: python {os.path.basename(__file__)} --demo <capability>")
        return
    
    if args.demo == 'all':
        selected_demos = demos.items()
    elif args.demo:
        if args.demo in demos:
            selected_demos = [(args.demo, demos[args.demo])]
        else:
            print(f"❌ Unknown demo: {args.demo}")
            return
    else:
        print("\n📋 Available Demos:")
        for name, demo in demos.items():
            print(f"  🔸 {name}: {demo['description']}")
        print(f"\nUsage: python {os.path.basename(__file__)} --demo <name>")
        print(f"       python {os.path.basename(__file__)} --list-only")
        return
    
    for demo_name, demo_info in selected_demos:
        print(f"\n🎯 Running Demo: {demo_info['description']}")
        print("=" * 80)
        
        for cmd in demo_info['commands']:
            run_command(cmd, f"Executing: {cmd.split()[-1] if '--help' in cmd else 'command'}")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
