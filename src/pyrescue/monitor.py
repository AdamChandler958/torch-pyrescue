import torch

from src.pyrescue.hooks.forward_hooks import NaNDectectorHook
from src.pyrescue.logger.logger_setup import setup_logging
from src.pyrescue.state_manager import StateManager
from src.pyrescue.utils.config_loader import load_config


class TrainingMonitor:
    def __init__(
        self,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        config_path: str = None,
    ):
        self.model = model
        self.optimiser = optimiser
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config["logging"])

        self.state_manager = StateManager(
            model=self.model,
            optimiser=self.optimiser,
            logger=self.logger,
            config=self.config["state_manager"],
        )

        self.nan_detector_hook = NaNDectectorHook(
            logger=self.logger, state_manager=self.state_manager
        )

    def _register_model_forward_hooks(self):
        # This will later be replaced with a more adaptive function to register both forward and backward hooks.
        self.model.register_forward_hook(self.nan_detector_hook)

    def step(self):
        self.state_manager.save_state()

    def start(self):
        # TODO: Add logic to handle adding optimiser hooks
        self._register_model_forward_hooks()
        self.logger.info("Monitoring started.")
