from __future__ import annotations
from abc import ABC, abstractmethod
import torch

class SDE(ABC):
    """
    Base class for continuous-time Stochastic Differential Equations.
    All diffusion processes (VP, VE, sub-VP) inherit from this class.
    """

    def __init__(self, T: float = 1.0):

        self.T = T

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    # ---------------------------------------------------------
    # Forward SDE

    @abstractmethod
    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Drift coefficient f(x,t)
        """

    @abstractmethod
    def diffusion(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Diffusion coefficient g(t)
        """

    # ---------------------------------------------------------
    # Closed-form perturbation kernel

    @abstractmethod
    def marginal_prob(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns

        mean(x_t | x0)

        std(x_t | x0)
        """

    # ---------------------------------------------------------
    # Prior distribution
    
    @abstractmethod
    def prior_sampling(
        self,
        shape: tuple[int, ...],
        device=None,
    ) -> torch.Tensor:
        """
        Sample from p_T(x)
        """

    @abstractmethod
    def prior_logp(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log probability of the prior.
        """

    @abstractmethod
    def marginal_std(
        self,
        t,
    ):
        pass
    
    # ---------------------------------------------------------
    # Perturbation

    def perturb(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Sample

            x_t ~ p(x_t | x0)
        """

        mean, std = self.marginal_prob(x0, t)

        noise = torch.randn_like(x0)

        xt = mean + std * noise

        return xt, noise

    # ---------------------------------------------------------

    def __repr__(self):

        return f"{self.__class__.__name__}(T={self.T})"

    def reverse_drift(
        self,
        x,
        t,
        score,
    ):

        """
        Reverse-time SDE drift

        f - g² score
        """

        drift = self.drift(
            x,
            t,
        )

        g = self.diffusion(t)

        while g.ndim < x.ndim:
            g = g.unsqueeze(-1)

        return drift - g**2 * score

    # ---------------------------------------------------------

    def probability_flow_drift(
        self,
        x,
        t,
        score,
    ):

        """
        Probability Flow ODE drift

        f - 1/2 g² score
        """

        drift = self.drift(
            x,
            t,
        )

        g = self.diffusion(t)

        while g.ndim < x.ndim:
            g = g.unsqueeze(-1)

        return drift - 0.5 * g**2 * score