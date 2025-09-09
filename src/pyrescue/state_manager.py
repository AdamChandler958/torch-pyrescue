import logging
import os
import uuid

import torch
import torch.nn as nn


class StateManager:
    def __init__(
        self,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        logger: logging.Logger,
        config: dict,
    ):
        self.model = model
        self.optimiser = optimiser
        self.checkpoint_directory = config["checkpoint_directory"]
        self.logger = logger
        self.current_save_state = {"model_dict": None}

    def save_state(self):
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        try:
            torch.save(
                self.model.state_dict(),
                f"{self.checkpoint_directory}/{self.model.__class__.__name__}-checkpoint-{uuid.uuid4().hex[:8]}.pth",
            )
            self.current_save_state["model_dict"] = self.model.state_dict()
        except Exception as e:
            self.logger.error(
                f"Failed to save {self.model.__class__.__name__} model weights with error: {e}"
            )
            raise

    def load_state(self):
        try:
            self.model.load_state_dict(self.current_save_state["model_dict"])
        except Exception as e:
            self.logger.error(
                f"Failed to load model weights for {self.model.__class__.__name__} with error: {e}"
            )
            raise

    def apply_lr_decrease(self, reduction_proportion: float = 0.5):
        current_lr = self.optimiser.param_groups[0]["lr"]
        self.optimiser.param_groups[0]["lr"] = current_lr * reduction_proportion
