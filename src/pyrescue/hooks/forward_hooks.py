import logging

import torch

from src.pyrescue.hooks.hook import Hook

logger = logging.getLogger(__name__)


class NaNDectectorHook(Hook):
    def __init__(self, name: str):
        super().__init__(name, is_forward_hook=True)

    def hook(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        if torch.isnan(output).any():
            logger.warning(
                "Nan value detected in the output of %s", module.__class__.__name__
            )

            # Trigger the observer and then reset.
            self.flag_status = True
            self.flag_status = False
