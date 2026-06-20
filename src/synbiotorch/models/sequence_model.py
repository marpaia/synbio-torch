"""A sequence transformer: pretrained backbone + mean pooling + task head."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn


class SequenceModel(nn.Module):
    """Wraps a HuggingFace encoder backbone with a pooling step and a task head.

    Pooling is attention-masked mean pooling over the final hidden states, which
    works for backbones (like DNABERT-2) that expose no dedicated pooler output.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, *, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _pool(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
        summed = (hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        pooled = self._pool(hidden_state, attention_mask)
        return self.head(pooled)
