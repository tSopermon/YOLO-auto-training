"""
Training callbacks for integrating monitoring with Ultralytics training.
"""

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER, callbacks

from .training_monitor import TrainingMonitor


def register_training_callbacks():
    """Register custom training callbacks with Ultralytics."""
    
    def on_pretrain_routine_end(trainer: BaseTrainer):
        """Called before training starts."""
        monitor: TrainingMonitor = trainer.args.get('custom_args', {}).get('monitor')
        if monitor:
            monitor.log_model_info(trainer.model)
            monitor.log_config()
            
    def on_train_epoch_end(trainer: BaseTrainer):
        """Called at end of each training epoch."""
        monitor: TrainingMonitor = trainer.args.get('custom_args', {}).get('monitor')
        if monitor:
            metrics = trainer.metrics
            monitor.log_epoch_metrics(
                epoch=trainer.epoch,
                train_metrics={'loss': trainer.loss.item()},
                val_metrics={'loss': trainer.validator.loss.item()} if hasattr(trainer.validator, 'loss') else {},
                learning_rate=trainer.optimizer.param_groups[0]['lr']
            )
            
    def on_train_end(trainer: BaseTrainer):
        """Called after training ends."""
        monitor: TrainingMonitor = trainer.args.get('custom_args', {}).get('monitor')
        if monitor:
            monitor.create_training_plots(save_plots=True)
            monitor.save_training_summary()
            monitor.close()
    
    # Register callbacks
    callbacks.default_callbacks.update({
        'on_pretrain_routine_end': on_pretrain_routine_end,
        'on_train_epoch_end': on_train_epoch_end,
        'on_train_end': on_train_end
    })
