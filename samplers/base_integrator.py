from __future__ import annotations
from abc import ABC, abstractmethod
import torch

class BaseIntegrator(ABC):

    @abstractmethod
    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        drift_fn,
        diffusion_fn=None,
    ) -> torch.Tensor:
        pass