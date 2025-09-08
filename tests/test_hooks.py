import pytest
import torch
from pytest_mock import MockerFixture

from src.pyrescue.hooks.forward_hooks import nan_detector_hook


@pytest.fixture
def simple_model():
    return torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Linear(5, 2))


def test_nan_detector_hook(simple_model: torch.nn.Module, mocker: MockerFixture):
    hook_handle = simple_model.register_forward_hook(nan_detector_hook)
    mock_log_error = mocker.patch("src.pyrescue.hooks.forward_hooks.logger.error")

    dummy_input = torch.randn(1, 10)
    _ = simple_model(dummy_input)

    mock_log_error.assert_not_called()

    problematic_input = torch.tensor([[float("nan")] * 10], dtype=torch.float32)

    try:
        _ = simple_model(problematic_input)
    except Exception:
        pass

    mock_log_error.assert_called()

    hook_handle.remove()
