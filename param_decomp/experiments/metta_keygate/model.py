from __future__ import annotations

import torch
from torch import Tensor, nn


class KeygateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.encoder(inputs))

    def hidden(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

