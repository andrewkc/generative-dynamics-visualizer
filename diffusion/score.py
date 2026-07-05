from __future__ import annotations
import torch

class ScoreModel:

    def __init__(
        self,
        model,
        sde,
    ):

        self.model = model
        self.sde = sde

    @torch.no_grad()
    def __call__(
        self,
        x,
        t,
    ):

        prediction = self.model(
            x,
            t,
        )

        std = self.sde.marginal_std(
            t,
        )

        while std.ndim < x.ndim:
            std = std.unsqueeze(-1)

        if self.model.prediction_type == "epsilon":

            epsilon = prediction

        elif self.model.prediction_type == "velocity":

            # Placeholder
            epsilon = prediction

        else:
            raise ValueError("Unknown prediction type.")

        return -epsilon / std
