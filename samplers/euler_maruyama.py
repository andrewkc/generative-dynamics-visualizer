from __future__ import annotations
import math
import torch
from .base_integrator import BaseIntegrator

class EulerMaruyamaIntegrator(BaseIntegrator):

    def step(
        self,
        x,
        t,
        dt,
        drift_fn,
        diffusion_fn,
    ):

        drift = drift_fn(
            x,
            t,
        )

        diffusion = diffusion_fn(
            t,
        )

        while diffusion.ndim < x.ndim:
            diffusion = diffusion.unsqueeze(-1)

        noise = torch.randn_like(x)

        #return x + dt * drift + math.sqrt(dt) * diffusion * noise
        return x + dt * drift + math.sqrt(abs(dt)) * diffusion * noise
