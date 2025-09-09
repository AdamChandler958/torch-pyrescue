import logging
import os
import uuid

import torch
import torch.nn as nn


class StateManager:
    def __init__(self, model: nn.Module, logger: logging.Logger, config: dict):
        self.model = model
        self.checkpoint_directory = config["checkpoint_directory"]
        self.logger = logger
        self.current_save_state = {"model_dict": None}

    def save_state(self):
        self.logger.info(
            "Saving current weights for model %s", self.model.__class__.__name__
        )
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        try:
            torch.save(
                self.model.state_dict(),
                f"{self.checkpoint_directory}/{self.model.__class__.__name__}-checkpoint-{uuid.uuid4().hex[:8]}.pth",
            )
            self.current_save_state = self.model.state_dict()
            self.logger.info(
                f"Successfully saved {self.model.__class__.__name__} model weights"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to save {self.model.__class__.__name__} model weights with error: {e}"
            )
            raise

    def load_state(self):
        self.logger.info(f"Loading weight dict for {self.model.__class__.__name__}")
        try:
            self.model.state_dict().update(self.current_save_state["model_dict"])
            self.logger.info(
                f"Successfully updated weights for {self.model.__class__.__name__}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load model weights for {self.model.__class__.__name__} with error: {e}"
            )
