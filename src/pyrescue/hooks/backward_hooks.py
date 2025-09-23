import logging

import torch

from src.pyrescue.state_manager import StateManager


class VanishingGradientHook:
    def __init__(self, logger: logging.Logger, state_manager: StateManager):
        self.logger = logger
        self.state_manager = state_manager
        self.gradient_norms = []
        self.is_vanishing = False

    def hook(self, module: torch.nn.Module, input_grads: tuple, ouput_grads: tuple):
        for grad in input_grads:
            if grad is not None:
                norm = torch.norm(grad)
                self.gradient_norms(norm.item())

        self.check_condition()

    def check_condition(self, threshold=1e-8, window_size=50):
        if len(self.gradient_norms) >= window_size:
            avg_norm = sum(self.gradient_norms[-window_size:]) / window_size
            if avg_norm < threshold:
                self.logger.warning("Vanishing gradients detected")
                self.is_vanishing = True
