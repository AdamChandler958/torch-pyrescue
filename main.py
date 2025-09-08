import torch

from src.pyrescue.state_manager import StateManager

model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Linear(5, 2))

sm = StateManager()

sm.save_state(model)
