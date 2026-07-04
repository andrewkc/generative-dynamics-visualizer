# datasets/eight_gaussians.py

from __future__ import annotations

import math

import torch

from .base import BaseDistribution


class EightGaussians(BaseDistribution):

    def __init__(
        self,
        radius: float = 2.0,
        std: float = 0.08,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):

        super().__init__(device, dtype)

        self.radius = radius
        self.std = std

        angles = torch.linspace(
            0,
            2 * math.pi,
            steps=9,
        )[:-1]

        self.centers = torch.stack(
            [
                radius * torch.cos(angles),
                radius * torch.sin(angles),
            ],
            dim=1,
        ).to(device=self.device, dtype=self.dtype)

    @property
    def name(self):

        return "eight_gaussians"

    def sample(self, n_samples: int):

        indices = torch.randint(
            0,
            8,
            (n_samples,),
            device=self.device,
        )

        centers = self.centers[indices]

        noise = torch.randn(
            n_samples,
            2,
            device=self.device,
            dtype=self.dtype,
        ) * self.std

        return centers + noise