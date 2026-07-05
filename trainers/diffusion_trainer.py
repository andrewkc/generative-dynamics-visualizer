from __future__ import annotations
import torch
from tqdm import tqdm

from diffusion.losses import (
    epsilon_loss,
    velocity_loss,
)

class DiffusionTrainer:

    def __init__(
        self,
        model,
        sde,
        optimizer,
        device="cpu",
        prediction_type="epsilon",
    ):

        self.model = model.to(device)
        self.sde = sde
        self.optimizer = optimizer
        self.device = device

        if prediction_type not in ["epsilon", "velocity"]:
            raise ValueError("prediction_type must be 'epsilon' or 'velocity'")

        self.prediction_type = prediction_type

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

            x0 = dataset.sample(batch_size).to(self.device)

            t = torch.rand(
                batch_size,
                device=self.device,
            )

            xt, noise = self.sde.perturb(x0, t)

            prediction = self.model(
                xt,
                t,
            )

            if self.prediction_type == "epsilon":

                target = noise

            else:

                mean, std = self.sde.marginal_prob(
                    x0,
                    t,
                )

                target = mean * noise - std * x0

            #loss = F.mse_loss(
            #    prediction,
            #    target,
            #)
            
            if self.prediction_type == "epsilon":
                loss = epsilon_loss(
                    prediction,
                    target,
                )

            else:
                loss = velocity_loss(
                    prediction,
                    target,
                )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            progress.set_postfix(loss=loss.item())

        return running_loss / steps_per_epoch
