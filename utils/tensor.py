from __future__ import annotations
import torch

def match_dims(
    tensor: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Expand tensor dimensions to match reference.
    """

    while tensor.ndim < reference.ndim:
        tensor = tensor.unsqueeze(-1)

    return tensor