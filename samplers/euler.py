from __future__ import annotations
import torch
from .base_integrator import BaseIntegrator

class EulerIntegrator(BaseIntegrator):

    def step(
        self,
        x,
        t,
        dt,
        drift_fn,
        diffusion_fn=None,
    ):

        drift = drift_fn(
            x,
            t,
        )

        return x + dt * drift
