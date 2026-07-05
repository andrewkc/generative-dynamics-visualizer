from __future__ import annotations
import torch
from .base_integrator import BaseIntegrator

class HeunIntegrator(BaseIntegrator):

    def step(
        self,
        x,
        t,
        dt,
        drift_fn,
        diffusion_fn=None,
    ):

        k1 = drift_fn(
            x,
            t,
        )

        predictor = x + dt * k1

        t_next = t + dt

        k2 = drift_fn(
            predictor,
            t_next,
        )

        return x + 0.5 * dt * (k1 + k2)
