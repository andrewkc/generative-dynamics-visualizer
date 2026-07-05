from __future__ import annotations
import torch


class LinearInterpolation:
    """
    Linear interpolation used in Flow Matching

    x(t) = (1-t)x0 + tx1
    """

    def sample(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:

        while t.ndim < x0.ndim:
            t = t.unsqueeze(-1)

        return (1.0 - t) * x0 + t * x1
