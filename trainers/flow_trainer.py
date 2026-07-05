from __future__ import annotations
import torch
from tqdm import tqdm

from flow_matching import (
    LinearInterpolation,
    ConstantVectorField,
    flow_matching_loss,
)

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
        self.interpolation = LinearInterpolation()
        self.vector_field = ConstantVectorField()
        
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

            xt = self.interpolation.sample(
                x0,
                x1,
                t,
            )

            target_velocity = self.vector_field.target(
                x0,
                x1,
            )

            prediction = self.model(
                xt,
                t,
            )

            loss = flow_matching_loss(
                prediction,
                target_velocity,
            )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            progress.set_postfix(loss=loss.item())

        return running_loss / steps_per_epoch
