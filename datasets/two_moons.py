# datasets/two_moons.py

from __future__ import annotations

import torch
from sklearn.datasets import make_moons

from .base import BaseDistribution


class TwoMoons(BaseDistribution):

    def __init__(
        self,
        noise: float = 0.05,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):

        super().__init__(device, dtype)

        self.noise = noise

    @property
    def name(self):

        return "two_moons"

    def sample(self, n_samples: int):

        x, _ = make_moons(
            n_samples=n_samples,
            noise=self.noise,
        )

        x = torch.tensor(
            x,
            dtype=self.dtype,
            device=self.device,
        )

        return x