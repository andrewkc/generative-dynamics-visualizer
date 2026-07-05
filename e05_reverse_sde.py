from diffusion import get_sde
from models import DiffusionModel
from samplers.reverse_sde import ReverseSDESampler

model = DiffusionModel()

sde = get_sde("vp")

sampler = ReverseSDESampler(
    model=model,
    sde=sde,
)

samples = sampler.sample(
    n_samples=5000,
    steps=200,
)

print(samples.shape)