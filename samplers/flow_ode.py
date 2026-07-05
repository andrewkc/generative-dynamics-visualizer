from __future__ import annotations
import torch
from .base_sampler import BaseSampler
from .euler import EulerIntegrator

class FlowODESampler(BaseSampler):
    """
    Flow Matching sampler.

    Solves

        dx/dt = v_theta(x,t)

    using any ODE integrator.
    """

    def __init__(
        self,
        model,
        integrator=None,
        device="cpu",
    ):

        super().__init__(model, device)

        if integrator is None:
            integrator = EulerIntegrator()

        self.integrator = integrator

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        steps: int = 100,
        return_trajectory: bool = False,
    ):

        # Base distribution
        x = torch.randn(
            n_samples,
            2,
            device=self.device,
        )

        trajectory = [x.clone()]

        dt = 1.0 / steps

        time_grid = torch.linspace(
            0.0,
            1.0,
            steps,
            device=self.device,
        )

        # Velocity field callback
        def drift_fn(x_current, t_current):

            return self.model(
                x_current,
                t_current,
            )

        # ODE integration
        for t_scalar in time_grid:

            t = torch.full(
                (n_samples,),
                float(t_scalar),
                device=self.device,
            )

            x = self.integrator.step(
                x=x,
                t=t,
                dt=dt,
                drift_fn=drift_fn,
            )

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory

        return x
