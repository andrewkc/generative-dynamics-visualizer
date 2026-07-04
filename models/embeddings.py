from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal time embedding used in diffusion models.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even.")

        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:

        if t.ndim == 0:
            t = t.unsqueeze(0)

        t = t.float()

        half_dim = self.embedding_dim // 2

        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half_dim,
                device=t.device,
                dtype=t.dtype,
            )
            / (half_dim - 1)
        )

        angles = t[:, None] * frequencies[None, :]

        embedding = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=1,
        )

        return embedding