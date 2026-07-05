from __future__ import annotations
import torch

class ConstantVectorField:
    """
    Ground-truth vector field for linear interpolation
    dx/dt = x1 - x0
    """

    def target(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:

        return x1 - x0