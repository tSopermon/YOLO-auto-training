"""
TensorBoard launcher utility for YOLO training.
Handles TensorBoard server startup and browser automation.
"""

import socket
import subprocess
import webbrowser
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def launch_tensorboard(log_dir: str | Path, port: int = None, wait_for_logs: bool = True) -> subprocess.Popen:
    """
    Launch TensorBoard server process.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        port: Port to run TensorBoard on. If None, finds a free port.
        wait_for_logs: Whether to wait for TensorBoard logs to be created
        
    Returns:
        subprocess.Popen: TensorBoard server process
    """
    if port is None:
        port = find_free_port()
    
    log_path = Path(log_dir)
    
    # For Ultralytics, wait for the nested directory structure to be created
    if wait_for_logs:
        import time as time_module
        max_wait = 30  # Maximum wait time in seconds
        wait_interval = 1  # Check every 1 second

        logger.info(f"Waiting for Ultralytics to create TensorBoard logs in {log_path}...")
        
        for elapsed in range(0, max_wait, wait_interval):
            # Look for nested directory structure (e.g., logs/22/22/)
            for subdir in log_path.rglob("*"):
                if subdir.is_dir() and subdir.name == log_path.name:
                    # Found nested directory with same name, check for TensorBoard files
                    tensorboard_files = list(subdir.glob("events.out.tfevents*"))
                    if tensorboard_files:
                        logger.info(f"Found TensorBoard logs in: {subdir}")
                        log_dir = subdir
                        break
            else:
                time_module.sleep(wait_interval)
                continue
            break
        else:
            logger.warning(f"No TensorBoard logs found after {max_wait}s, using base directory: {log_dir}")
    
    # Ultralytics saves TensorBoard logs in the experiment subdirectory
    # Check if there are subdirectories with TensorBoard events
    tensorboard_dirs = []
    
    # Look for TensorBoard event files in subdirectories
    if log_path.exists():
        for subdir in log_path.rglob("*"):
            if subdir.is_dir():
                # Check if this directory contains TensorBoard event files
                if any(f.name.startswith("events.out.tfevents") for f in subdir.iterdir() if f.is_file()):
                    tensorboard_dirs.append(subdir)
        
        # If we found specific TensorBoard directories, use the most recent one
        if tensorboard_dirs:
            tensorboard_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            log_dir = tensorboard_dirs[0]
            logger.info(f"Using TensorBoard directory: {log_dir}")
        else:
            # Fallback: check if this is an Ultralytics experiment structure
            experiment_dirs = [d for d in log_path.iterdir() if d.is_dir() and d.name.isdigit()]
            if experiment_dirs:
                # Use the most recent experiment directory
                experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                nested_dir = experiment_dirs[0] / experiment_dirs[0].name
                if nested_dir.exists():
                    log_dir = nested_dir
                    logger.info(f"Using Ultralytics nested directory: {log_dir}")
                else:
                    log_dir = experiment_dirs[0]
                    logger.info(f"Using Ultralytics experiment directory: {log_dir}")
        
    try:
        # Get the current Python executable to ensure we use the same environment
        import sys
        python_executable = sys.executable
        
        # Start TensorBoard process
        process = subprocess.Popen(
            [
                python_executable, '-m', 'tensorboard.main',  # Use current Python environment
                '--logdir', str(log_dir),
                '--port', str(port),
                '--bind_all'  # Allow external connections
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give TensorBoard time to start
        time.sleep(3)
        
        # Open browser
        url = f'http://localhost:{port}'
        webbrowser.open(url)
        
        logger.info(f"TensorBoard launched at {url}")
        logger.info(f"Monitoring directory: {log_dir}")
        return process
        
    except Exception as e:
        logger.warning(f"Failed to launch TensorBoard: {e}")
        return None
