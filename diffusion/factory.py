from .vp import VPSDE
from .ve import VESDE
from .subvp import SubVPSDE

_SDES = {
    "vp": VPSDE,
    "ve": VESDE,
    "subvp": SubVPSDE,
}


def get_sde(name: str, **kwargs):
    name = name.lower()
    if name not in _SDES:
        raise ValueError(f"Unknown SDE '{name}'")

    return _SDES[name](**kwargs)
