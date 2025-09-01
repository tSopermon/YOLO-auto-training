"""
Simplified training monitor for YOLO models.
Provides basic monitoring and TensorBoard UI launch.
"""

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Simplified training monitor for YOLO models."""

    def __init__(
        self,
        config,
        log_dir: Union[str, Path],
        enable_tensorboard: bool = True,
        enable_wandb: bool = False,
    ):
        """
        Initialize training monitor.

        Args:
            config: Training configuration
            log_dir: Directory to save logs and visualizations
            enable_tensorboard: Whether to enable TensorBoard logging
            enable_wandb: Whether to enable Weights & Biases logging
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Launch TensorBoard if enabled (Ultralytics has built-in support, but we can still launch the UI)
        self.tensorboard_process = None
        if enable_tensorboard:
            self._setup_tensorboard()

        logger.info(f"Training monitor initialized: {self.log_dir}")

    def _setup_tensorboard(self) -> None:
        """Setup and launch TensorBoard UI (Ultralytics handles logging automatically)."""
        try:
            from .tensorboard_launcher import launch_tensorboard
            import threading
            import time

            # Launch TensorBoard in a separate thread with delay to wait for Ultralytics logs
            def delayed_tensorboard_launch():
                logger.info("Waiting for training to start and TensorBoard logs to be created...")
                time.sleep(10)  # Wait 10 seconds for training to start
                self.tensorboard_process = launch_tensorboard(self.log_dir, wait_for_logs=True)
                if self.tensorboard_process:
                    logger.info(f"TensorBoard logging enabled: {self.log_dir}")
                    logger.info("TensorBoard web interface launched in browser")
                    logger.info("Note: Ultralytics will automatically log training metrics to TensorBoard")
                    logger.info("TensorBoard will remain running after training completes for result analysis")

            # Start the delayed launch in a separate thread (not daemon so it persists)
            tensorboard_thread = threading.Thread(target=delayed_tensorboard_launch, daemon=False)
            tensorboard_thread.start()

        except ImportError:
            logger.warning(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
        except Exception as e:
            logger.warning(f"Failed to setup TensorBoard: {e}")

    def close(self, keep_tensorboard: bool = True) -> None:
        """Close the monitor and cleanup resources."""
        if self.tensorboard_process and not keep_tensorboard:
            try:
                self.tensorboard_process.terminate()
                logger.info("TensorBoard server stopped")
            except Exception as e:
                logger.warning(f"Error stopping TensorBoard: {e}")
        elif self.tensorboard_process and keep_tensorboard:
            logger.info("TensorBoard server is still running - you can view training results in your browser")
            logger.info("To stop TensorBoard later, use: pkill -f tensorboard")

        logger.info("Training monitor closed successfully")
