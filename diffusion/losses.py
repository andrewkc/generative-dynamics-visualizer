from __future__ import annotations

import torch
import torch.nn.functional as F


def epsilon_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Standard DDPM / Score SDE epsilon prediction loss.
    """

    return F.mse_loss(
        prediction,
        target,
    )


def velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Velocity prediction loss.
    """

    return F.mse_loss(
        prediction,
        target,
    )