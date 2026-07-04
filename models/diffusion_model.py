from __future__ import annotations

import torch.nn as nn

from .mlp import MLP


class DiffusionModel(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 256,
        time_embedding_dim: int = 64,
    ):

        super().__init__()

        self.backbone = MLP(
            hidden_dim=hidden_dim,
            time_embedding_dim=time_embedding_dim,
        )

    def forward(self, x, t):

        return self.backbone(x, t)