from __future__ import annotations
from abc import ABC, abstractmethod
import torch

class BaseSampler(ABC):
    """
    Base class for all samplers.
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
    ):

        self.model = model.to(device)
        self.device = device

        self.model.eval()

    @torch.no_grad()
    @abstractmethod
    def sample(
        self,
        n_samples: int,
        steps: int = 100,
    ):
        """
        Generate samples.
        """
        pass