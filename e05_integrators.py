import torch

from samplers import (
    EulerIntegrator,
    EulerMaruyamaIntegrator,
    HeunIntegrator,
)

x = torch.randn(8, 2)

dt = 0.01

t = torch.zeros(8)

def drift_fn(x, t):
    return -x

def diffusion_fn(t):
    return torch.ones_like(t)

#
# Euler
#

euler = EulerIntegrator()

x1 = euler.step(
    x=x,
    t=t,
    dt=dt,
    drift_fn=drift_fn,
)

print("Euler:", x1.shape)

#
# Euler-Maruyama
#

emar = EulerMaruyamaIntegrator()

x2 = emar.step(
    x=x,
    t=t,
    dt=dt,
    drift_fn=drift_fn,
    diffusion_fn=diffusion_fn,
)

print("Euler-Maruyama:", x2.shape)

#
# Heun
#

heun = HeunIntegrator()

x3 = heun.step(
    x=x,
    t=t,
    dt=dt,
    drift_fn=drift_fn,
)

print("Heun:", x3.shape)