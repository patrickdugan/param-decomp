from __future__ import annotations

import torch
from torch import Tensor, nn


class RecursiveGateTRM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, recursive_steps: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.transition = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.recursive_steps = recursive_steps

    def hidden(self, inputs: Tensor) -> Tensor:
        state = torch.tanh(self.input_proj(inputs))
        for _ in range(self.recursive_steps):
            state = self.norm(state + torch.tanh(self.transition(state)))
        return state

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.hidden(inputs))

