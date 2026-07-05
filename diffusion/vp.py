from __future__ import annotations
import torch
import math
from .sde import SDE

class VPSDE(SDE):

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
    ):

        super().__init__()

        self.beta_min = beta_min
        self.beta_max = beta_max

    @property
    def name(self):

        return "vp"

    def beta(self, t):

        return self.beta_min + t * (
            self.beta_max - self.beta_min
        )

    def drift(self, x, t):

        beta = self.beta(t)

        while beta.ndim < x.ndim:
            beta = beta.unsqueeze(-1)

        return -0.5 * beta * x

    def diffusion(self, t):

        return torch.sqrt(self.beta(t))

    def marginal_prob(self, x0, t):

        log_mean_coeff = (
            -0.25
            * t**2
            * (self.beta_max - self.beta_min)
            -0.5
            * t
            * self.beta_min
        )

        while log_mean_coeff.ndim < x0.ndim:
            log_mean_coeff = log_mean_coeff.unsqueeze(-1)

        mean = torch.exp(log_mean_coeff) * x0

        std = torch.sqrt(
            1.0 - torch.exp(2.0 * log_mean_coeff)
        )

        return mean, std

    def prior_sampling(
        self,
        shape,
        device=None,
    ):

        return torch.randn(shape, device=device)

    def prior_logp(self, z):

        N = z[0].numel()

        #return (
        #    -N / 2 * torch.log(torch.tensor(2 * torch.pi))
        #    - torch.sum(z**2, dim=1) / 2
        #)
        
        return (
            -0.5 * z.shape[1] * math.log(2.0 * math.pi)
            -0.5 * torch.sum(z ** 2, dim=1)
        )

    def marginal_std(self, t):

        log_mean_coeff = (

            -0.25
            * t**2
            * (self.beta_max-self.beta_min)

            -0.5
            * t
            * self.beta_min
        )

        std = torch.sqrt(
            1.0 -
            torch.exp(
                2.0 * log_mean_coeff
            )
        )

        return std