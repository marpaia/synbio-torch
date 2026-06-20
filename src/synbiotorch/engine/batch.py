"""Batch adapters decouple the training loop from a batch's concrete shape.

The Trainer never inspects ``input_ids`` or any modality-specific key. Instead it
asks an adapter to move a batch to the device, run the model on it, and pull out
the labels. Sequence and MLM batches are ``dict[str, Tensor]`` handled by
``TensorBatchAdapter``; graph batches (PyG ``Batch``) get their own adapter.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class BatchAdapter(Protocol):
    def to_device(self, batch: Any, device: torch.device) -> Any: ...

    def forward(self, model: torch.nn.Module, batch: Any) -> torch.Tensor: ...

    def labels(self, batch: Any) -> torch.Tensor: ...


class TensorBatchAdapter:
    """For ``dict[str, Tensor]`` batches: model inputs are every key but ``labels``."""

    label_key = "labels"

    def to_device(self, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in batch.items()}

    def forward(self, model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {k: v for k, v in batch.items() if k != self.label_key}
        return model(**inputs)

    def labels(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch[self.label_key]


class GraphBatchAdapter:
    """For PyG ``Batch`` objects: feed node/edge tensors to the graph model."""

    def to_device(self, batch: Any, device: torch.device) -> Any:
        return batch.to(device)

    def forward(self, model: torch.nn.Module, batch: Any) -> torch.Tensor:
        return model(batch.x, batch.edge_index, batch.edge_type, batch.batch)

    def labels(self, batch: Any) -> torch.Tensor:
        return batch.y
