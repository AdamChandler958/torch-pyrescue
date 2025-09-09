import logging

import torch

from src.pyrescue.state_manager import StateManager


class NaNDectectorHook:
    def __init__(self, logger: logging.Logger, state_manager: StateManager):
        self.logger = logger
        self.state_manager = state_manager
        self.detected = False

    def __call__(
        self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ):
        if torch.isnan(output).any():
            self.logger.error(
                "Nan value detected in the output of %s", module.__class__.__name__
            )
            self.detected = True

            self.logger.info("Reverting save state")
            self.state_manager.load_state()
            self.state_manager.apply_lr_decrease()

            self.detected = False
