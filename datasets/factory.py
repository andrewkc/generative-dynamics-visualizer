# datasets/factory.py

from .two_moons import TwoMoons
from .eight_gaussians import EightGaussians


def get_dataset(name: str, **kwargs):

    name = name.lower()

    if name == "two_moons":
        return TwoMoons(**kwargs)

    if name == "eight_gaussians":
        return EightGaussians(**kwargs)

    raise ValueError(f"Unknown dataset: {name}")