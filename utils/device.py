from __future__ import annotations
import torch

def get_device() -> torch.device:
    """
    Return the best available device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")