import logging

import torch


class NaNDectectorHook:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def __call__(
        self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
    ):
        if torch.isnan(output).any():
            self.logger.error(
                "Nan value detected in the output of %s", module.__class__.__name__
            )
