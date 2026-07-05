from __future__ import annotations
import torch
from diffusion import ScoreModel

from .base_sampler import BaseSampler
from .euler_maruyama import EulerMaruyamaIntegrator

class ReverseSDESampler(BaseSampler):
    """
    Reverse-time SDE sampler.

    Integrates

        dx = [f(x,t) - g(t)^2 score(x,t)] dt + g(t) dW

    using any SDE integrator
    """

    def __init__(
        self,
        model,
        sde,
        integrator=None,
        device="cpu",
    ):

        super().__init__(model, device)

        self.sde = sde

        self.score_model = ScoreModel(
            model=model,
            sde=sde,
        )

        if integrator is None:
            integrator = EulerMaruyamaIntegrator()

        self.integrator = integrator

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        steps: int = 100,
        return_trajectory: bool = False,
    ):

        # Initial state
        x = self.sde.prior_sampling(
            (n_samples, 2),
            device=self.device,
        )

        trajectory = [x.clone()]

        dt = -self.sde.T / steps

        time_grid = torch.linspace(
            self.sde.T,
            1e-3,
            steps,
            device=self.device,
        )

        # Drift callback
        def drift_fn(x_current, t_current):

            score = self.score_model(
                x_current,
                t_current,
            )

            return self.sde.reverse_drift(
                x_current,
                t_current,
                score,
            )

        # Diffusion callback
        def diffusion_fn(t_current):

            return self.sde.diffusion(
                t_current,
            )

        # Reverse integration
        for t_scalar in time_grid:

            t = x.new_full(
                (n_samples,),
                float(t_scalar),
            )

            x = self.integrator.step(
                x=x,
                t=t,
                dt=dt,
                drift_fn=drift_fn,
                diffusion_fn=diffusion_fn,
            )

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory

        return x
