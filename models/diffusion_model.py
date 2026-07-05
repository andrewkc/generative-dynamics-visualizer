from __future__ import annotations
import torch.nn as nn
from .mlp import MLP


class DiffusionModel(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 256,
        time_embedding_dim: int = 64,
        prediction_type: str = "epsilon",
    ):

        super().__init__()
        
        if prediction_type not in [
            "epsilon",
            "velocity",
        ]:
            raise ValueError(
                "prediction_type must be 'epsilon' or 'velocity'."
            )

        self.prediction_type = prediction_type

        self.backbone = MLP(
            hidden_dim=hidden_dim,
            time_embedding_dim=time_embedding_dim,
        )

    def forward(self, x, t):

        return self.backbone(x, t)