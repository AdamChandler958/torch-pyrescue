import os
import uuid

import torch
import torch.nn as nn

from src.pyrescue.logger.logger import logger


class StateManager:
    def __init__(self, checkpoint_directory: str = "model_checkpoints"):
        self.checkpoint_directory = checkpoint_directory

    def save_state(self, model: nn.Module):
        logger.info("Saving current weights for model %s", model.__class__.__name__)
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        try:
            torch.save(
                model.state_dict(),
                f"{self.checkpoint_directory}/{model.__class__.__name__}-checkpoint-{uuid.uuid4().hex[:8]}.pth",
            )
            logger.info(f"Successfully saved {model.__class__.__name__} model weights")
        except Exception as e:
            logger.error(
                f"Failed to save {model.__class__.__name__} model weights with error: {e}"
            )
            raise

    def load_state(self, model, optimiser):
        pass
