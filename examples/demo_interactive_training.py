#!/usr/bin/env python3
"""
Demo script showing how to use the interactive YOLO training system.
This script demonstrates the different ways to run training.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and show its description."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print(f"{'='*60}")

    response = input("Run this command? (y/N): ").strip().lower()
    if response in ["y", "yes"]:
        try:
            # Run the command
            result = subprocess.run(command.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print("SUCCESS: Command completed successfully!")
                if result.stdout:
                    print("Output:", result.stdout)
            else:
                print("FAILED: Command failed!")
                if result.stderr:
                    print("Error:", result.stderr)
        except Exception as e:
            print(f"ERROR: Error running command: {e}")
    else:
        print("SKIPPED")


def main():
    """Demo the interactive training system."""
    print("YOLO Interactive Training System Demo")
    print("=" * 60)
    print("This demo shows you different ways to run YOLO training.")
    print("Choose which example you'd like to try:")

    examples = [
        {
            "command": "python train.py --model-type yolov8 --help",
            "description": "Show all available command line options",
        },
        {
            "command": "python train.py --model-type yolov8 --non-interactive --results-folder demo_run",
            "description": "Run training with defaults (no prompts)",
        },
        {
            "command": "python train.py --model-type yolov8 --validate-only --non-interactive --results-folder demo_validation",
            "description": "Run validation only with defaults",
        },
        {
            "command": "python train.py --model-type yolov8 --epochs 5 --batch-size 4 --image-size 640 --results-folder quick_test",
            "description": "Run quick training with custom parameters (no prompts)",
        },
    ]

    print("\nAvailable examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")

    print("\n5. Interactive training with YOLO version selection (full experience)")
    print("6. Exit demo")

    while True:
        choice = input("\nChoose an option (1-6): ").strip()

        if choice == "1":
            run_command(examples[0]["command"], examples[0]["description"])
        elif choice == "2":
            run_command(examples[1]["command"], examples[1]["description"])
        elif choice == "3":
            run_command(examples[2]["command"], examples[2]["description"])
        elif choice == "4":
            run_command(examples[3]["command"], examples[3]["description"])
        elif choice == "5":
            print("\nInteractive Training Experience")
            print("=" * 60)
            print("This will start the full interactive training experience.")
            print("You'll be prompted for:")
            print("• YOLO version (YOLO11, YOLOv8, YOLOv5)")
            print("• Model size (n/s/m/l/x)")
            print("• Number of epochs")
            print("• Batch size")
            print("• Image size")
            print("• Learning rate")
            print("• Training device")
            print("• Advanced options")
            print("• Results folder name")
            print(
                "\nNOTE: This will start actual training if you complete all prompts!"
            )

            response = input("Start interactive training? (y/N): ").strip().lower()
            if response in ["y", "yes"]:
                print("\nStarting interactive training...")
                print("Type 'python train.py' to begin with full interactive setup!")
                print(
                    "Or 'python train.py --model-type yolov8' to skip YOLO version selection."
                )
                print("Or press Ctrl+C to cancel.")
            else:
                print("Interactive training cancelled.")
        elif choice == "6":
            print("\nDemo completed! Happy training!")
            break
        else:
            print("Invalid choice. Please enter 1-6.")

        # Ask if user wants to continue
        if choice != "6":
            continue_demo = input("\nContinue with demo? (y/N): ").strip().lower()
            if continue_demo not in ["y", "yes"]:
                print("\nDemo completed! Happy training!")
                break


if __name__ == "__main__":
    main()
