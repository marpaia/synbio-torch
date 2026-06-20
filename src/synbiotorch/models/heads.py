"""Task heads attached on top of a backbone's pooled representation."""

from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(pooled)).squeeze(-1)


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(pooled))
