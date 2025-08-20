#!/usr/bin/env python3
"""
Test runner for YOLO training repository.
Run all tests or specific test categories.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_pattern=None, verbose=False, coverage=False, parallel=False):
    """Run pytest with specified options."""
    cmd = ["python", "-m", "pytest"]

    # Add test directory
    cmd.append("tests/")

    # Add test pattern if specified
    if test_pattern:
        cmd.append(f"-k={test_pattern}")

    # Add verbose flag
    if verbose:
        cmd.append("-v")

    # Add coverage
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])

    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])

    # Add other useful options
    cmd.extend(
        [
            "--tb=short",  # Short traceback format
            "--strict-markers",  # Strict marker checking
            "--durations=10",  # Show 10 slowest tests
        ]
    )

    print(f"Running tests with command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 60)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print(f"❌ Tests failed with exit code: {e.returncode}")
        return False


def run_specific_tests():
    """Run specific test categories."""
    test_categories = {
        "config": "Test configuration system",
        "data": "Test data loading utilities",
        "checkpoint": "Test checkpoint management",
        "model": "Test model loading",
        "training": "Test training utilities",
        "monitor": "Test training monitoring",
        "integration": "Test integration scenarios",
    }

    print("Available test categories:")
    for category, description in test_categories.items():
        print(f"  {category:12} - {description}")

    print("\nRunning all test categories...")

    all_passed = True
    for category in test_categories:
        print(f"\n🧪 Running {category} tests...")
        print("-" * 40)

        if run_tests(test_pattern=category, verbose=True):
            print(f"✅ {category} tests passed")
        else:
            print(f"❌ {category} tests failed")
            all_passed = False

    return all_passed


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run YOLO training tests")
    parser.add_argument(
        "--pattern",
        "-k",
        type=str,
        help="Test pattern to run (e.g., 'config' for configuration tests)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Run with coverage reporting"
    )
    parser.add_argument(
        "--parallel", "-p", action="store_true", help="Run tests in parallel"
    )
    parser.add_argument(
        "--all", "-a", action="store_true", help="Run all test categories separately"
    )

    args = parser.parse_args()

    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Error: tests directory not found. Run from repository root.")
        sys.exit(1)

    print("🚀 YOLO Training Test Runner")
    print("=" * 60)

    if args.all:
        # Run all test categories separately
        success = run_specific_tests()
        sys.exit(0 if success else 1)
    else:
        # Run tests with specified options
        success = run_tests(
            test_pattern=args.pattern,
            verbose=args.verbose,
            coverage=args.coverage,
            parallel=args.parallel,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
