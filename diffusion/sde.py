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
    
    
    @torch.no_grad()
    def sample_forward_trajectory(
        self,
        x0: torch.Tensor,
        steps: int = 100,
        return_times: bool = False,
    ):
        """
        Simulate the forward SDE using Euler-Maruyama.

        Returns
        -------
        trajectory :
            list[Tensor(N,2)]

        times :
            Tensor(steps+1)
            (optional)
        """

        x = x0.clone()

        trajectory = [x.clone()]

        dt = self.T / steps

        times = torch.linspace(
            0.0,
            self.T,
            steps + 1,
            device=x.device,
        )

        for i in range(steps):

            t = torch.full(
                (x.shape[0],),
                float(times[i]),
                device=x.device,
            )

            drift = self.drift(
                x,
                t,
            )

            diffusion = self.diffusion(
                t,
            )

            while diffusion.ndim < x.ndim:
                diffusion = diffusion.unsqueeze(-1)

            noise = torch.randn_like(x)

            x = (
                x
                + drift * dt
                + diffusion * torch.sqrt(
                    torch.tensor(
                        dt,
                        device=x.device,
                    )
                )
                * noise
            )

            trajectory.append(
                x.clone()
            )

        if return_times:
            return trajectory, times

        return trajectory
    
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