import os
import shutil
from argparse import Namespace

from mmengine import Config

from src.logger import logger
from src.registry import CONFIG
from src.utils import assemble_project_path, init_before_training


@CONFIG.register_module(force=True)
class ConfigBuilder:
    def __init__(self, config_path: str, args: Namespace):
        self.config_path = config_path
        self.args = args
        self.config = Config.fromfile(filename=config_path)
        self._cfg_options = {}
        self._parse_args()

    def _parse_args(self):
        for item, value in self.args.__dict__.items():
            if item != 'config' and value is not None:
                self._cfg_options[item] = value
        self.config.merge_from_dict(self._cfg_options)

    def _set_exp_path(self):
        self.config.exp_path = assemble_project_path(
            os.path.join(self.config.workdir, self.config.tag)
        )
        if self.config.if_remove is None:
            self.config.if_remove = input(f"| Arguments PRESS 'y' to REMOVE: {self.config.exp_path}? ") == 'y'

        if self.config.if_remove:
            shutil.rmtree(self.config.exp_path, ignore_errors=True)
            logger.info(f'| Arguments Remove work_dir: {self.config.exp_path}')
        else:
            logger.info(f'| Arguments Keep work_dir: {self.config.exp_path}')

        os.makedirs(self.config.exp_path, exist_ok=True)

    def _set_directories(self):
        self.config.checkpoint_path = os.path.join(self.config.exp_path, self.config.checkpoint_path)
        os.makedirs(self.config.checkpoint_path, exist_ok=True)
        logger.info(f'| Arguments Checkpoint path: {self.config.checkpoint_path}')

        self.config.wandb_path = os.path.join(self.config.exp_path, self.config.wandb_path)
        os.makedirs(self.config.wandb_path, exist_ok=True)
        logger.info(f'| Arguments Wandb path: {self.config.wandb_path}')

    def _init_seed(self):
        init_before_training(self.config.seed)
        logger.info(f'| Arguments Seed: {self.config.seed}')

    def build(self) -> Config:
        self._set_exp_path()
        self._set_directories()
        self._init_seed()
        return self.config
