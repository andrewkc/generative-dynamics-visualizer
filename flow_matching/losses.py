from __future__ import annotations
import torch
import torch.nn.functional as F

def flow_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:

    return F.mse_loss(
        prediction,
        target,
    )