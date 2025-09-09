import torch

from src.pyrescue.monitor import TrainingMonitor

model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Linear(5, 2))

optimiser = torch.optim.Adam(model.parameters())

monitor = TrainingMonitor(model, optimiser)

monitor.start()

for epoch in range(10):
    try:
        monitor.step()
        optimiser.zero_grad()
        if epoch == 3:
            input = torch.tensor([[float("nan")] * 10], dtype=torch.float32)
        else:
            input = torch.randn((1, 10))

        ouputs = model(input)

        loss = torch.sum(ouputs)
        loss.backward()
        optimiser.step()

        monitor.logger.info(f"Epoch {epoch + 1} complete. Loss: {loss.item()}")
    except Exception as e:
        monitor.logger.error(
            f"Training interrupted due to an issue: {e}. Reverting to an earlier state and continuing."
        )
        continue
