import torch

from src.pyrescue.hooks.forward_hooks import NaNDectectorHook
from src.pyrescue.logger.logger_setup import setup_logging
from src.pyrescue.state_manager import StateManager
from src.pyrescue.utils.config_loader import load_config


class TrainingMonitor:
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config["logging"])

        self.nan_detector_hook = NaNDectectorHook(logger=self.logger)
        self.state_manager = StateManager(
            logger=self.logger, config=self.config["state_manager"]
        )

    def _register_model_forward_hooks(self, model: torch.nn.Module):
        # This will later be replaced with a more adaptive function to register both forward and backward hooks.
        model.register_forward_hook(self.nan_detector_hook)

    def start(self, model: torch.nn.Module, optimiser: torch.optim.Optimizer):
        # TODO: Add logic to handle adding optimiser hooks
        self._register_model_forward_hooks(model)
        self.logger.info("Monitoring started.")
