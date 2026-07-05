import torch

from diffusion import get_sde

sde = get_sde("vp")

x0 = torch.randn(1024, 2)

t = torch.rand(1024)

xt, noise = sde.perturb(x0, t)

print(xt.shape)
print(noise.shape)