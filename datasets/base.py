# datasets/base.py

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseDistribution(ABC):
    """
    Base class for every synthetic 2D distribution used in the project.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:

        self.device = torch.device(device)
        self.dtype = dtype
        self.dimension = 2

    @property
    @abstractmethod
    def name(self) -> str:
        """Distribution name."""
        pass

    @abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples.

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        Tensor of shape (n_samples, 2)
        """
        pass

    def __call__(self, n_samples: int) -> torch.Tensor:
        return self.sample(n_samples)