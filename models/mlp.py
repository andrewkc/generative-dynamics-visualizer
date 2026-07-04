from __future__ import annotations

import torch
import torch.nn as nn

from .embeddings import SinusoidalTimeEmbedding


class MLP(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 256,
        time_embedding_dim: int = 64,
    ):

        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(
            embedding_dim=time_embedding_dim
        )

        input_dim = 2 + time_embedding_dim

        self.network = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, 2)

        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:

        if t.ndim == 0:
            t = t.repeat(x.shape[0])

        if t.ndim == 1:
            pass
        else:
            t = t.squeeze()

        t_embedding = self.time_embedding(t)

        x = torch.cat(
            [
                x,
                t_embedding,
            ],
            dim=1,
        )

        return self.network(x)