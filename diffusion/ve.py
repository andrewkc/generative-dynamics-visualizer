from __future__ import annotations
import math
import torch
from .sde import SDE

class VESDE(SDE):
    """
    Variance Exploding SDE
    dx = sqrt(d sigma(t)^2 / dt) dW
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
    ):

        super().__init__()

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @property
    def name(self):

        return "ve"

    # ---------------------------------------------------------

    def sigma(self, t):

        return self.sigma_min * (
            self.sigma_max / self.sigma_min
        ) ** t

    # ---------------------------------------------------------

    def drift(self, x, t):

        return torch.zeros_like(x)

    # ---------------------------------------------------------

    def diffusion(self, t):

        sigma = self.sigma(t)

        return sigma * torch.sqrt(
            torch.tensor(
                2.0 * math.log(self.sigma_max / self.sigma_min),
                device=t.device,
                dtype=t.dtype,
            )
        )

    # ---------------------------------------------------------

    def marginal_prob(self, x0, t):

        sigma = self.sigma(t)

        while sigma.ndim < x0.ndim:
            sigma = sigma.unsqueeze(-1)

        mean = x0

        std = sigma

        return mean, std

    # ---------------------------------------------------------

    def prior_sampling(
        self,
        shape,
        device=None,
    ):

        return torch.randn(shape, device=device) * self.sigma_max

    # ---------------------------------------------------------

    def prior_logp(self, z):

        D = z.shape[1]

        return (
            -0.5 * D * math.log(2 * math.pi * self.sigma_max**2)
            - torch.sum(z**2, dim=1)
            / (2 * self.sigma_max**2)
        )
        
    def marginal_std(self, t):

        return self.sigma(t)