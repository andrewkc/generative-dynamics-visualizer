from __future__ import annotations
import torch
import torch.nn.functional as F
from tqdm import tqdm


class FlowTrainer:

    def __init__(
        self,
        model,
        optimizer,
        device="cpu",
    ):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_epoch(
        self,
        dataset,
        steps_per_epoch,
        batch_size,
    ):

        self.model.train()

        running_loss = 0.0

        progress = tqdm(range(steps_per_epoch))

        for _ in progress:

            x1 = dataset.sample(batch_size).to(self.device)

            x0 = torch.randn_like(x1)

            t = torch.rand(
                batch_size,
                device=self.device,
            )

            t_expanded = t.unsqueeze(1)

            xt = (1.0 - t_expanded) * x0 + t_expanded * x1

            target_velocity = x1 - x0

            prediction = self.model(
                xt,
                t,
            )

            loss = F.mse_loss(
                prediction,
                target_velocity,
            )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            progress.set_postfix(loss=loss.item())

        return running_loss / steps_per_epoch
