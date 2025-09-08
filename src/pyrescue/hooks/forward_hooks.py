import torch

from src.pyrescue.logger import logger


def nan_detector_hook(
    module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
):
    if torch.isnan(output).any():
        logger.error(
            "Nan value detected in the output of %s", module.__class__.__name__
        )
