import torch
import wandb

from src.logger import logger
from src.registry import LOGGER


@LOGGER.register_module(force=True)
class WandbLogger:
    def __init__(self, log_dir, project_name, run_name=None, config=None):
        self.log_dir = log_dir
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.run = None

    def initialize(self):
        self.run = wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=self.config,
            dir=self.log_dir,
        )
        logger.info(f'Wandb run initialized: {self.run_name or self.run.name}')

    def log(self, metrics, step=None):
        if self.run is None:
            raise RuntimeError(
                'Wandb run has not been initialized. Call `initialize()` first.'
            )
        wandb.log(metrics, step=step)

    def save_model(self, model, model_path):
        if self.run is None:
            raise RuntimeError(
                'Wandb run has not been initialized. Call `initialize()` first.'
            )

        torch.save(model.state_dict(), model_path)
        logger.info(f'Model saved locally at {model_path}')

        wandb.save(model_path)
        logger.info(f'Model uploaded to wandb: {model_path}')

    def finish(self):
        if self.run is not None:
            self.run.finish()
            logger.info('Wandb run finished.')
        else:
            logger.info('Wandb run was not initialized.')
