"""Torch Dataset and padding collator over encoded Designs."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from synbiotorch.encoders.base import ModelInput, SupportsEncode
from synbiotorch.types import Design


class EncodedDataset(Dataset):
    """Lazily encodes Designs on ``__getitem__``.

    Works with any encoder: tensor encoders return ``ModelInput``, the graph
    encoder returns a PyG ``Data`` — the dataset just passes the result through
    to the (modality-appropriate) collator / loader.
    """

    def __init__(self, objects: Sequence[Design], encoder: SupportsEncode) -> None:
        self._objects = list(objects)
        self._encoder = encoder

    def __len__(self) -> int:
        return len(self._objects)

    def __getitem__(self, index: int) -> object:
        return self._encoder.encode(self._objects[index])


def pad_token_batch(batch: list[ModelInput], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad input_ids/attention_mask to the batch's longest sequence."""
    max_len = max(len(item.input_ids) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        length = len(item.input_ids)
        input_ids[i, :length] = torch.tensor(item.input_ids, dtype=torch.long)
        attention[i, :length] = torch.tensor(item.attention_mask, dtype=torch.long)
    return input_ids, attention


class Collator:
    """Pads a batch of ModelInputs and attaches supervised labels."""

    def __init__(self, pad_token_id: int, *, with_labels: bool = True, label_dtype: str = "float") -> None:
        self.pad_token_id = pad_token_id
        self.with_labels = with_labels
        self.label_dtype = label_dtype

    def __call__(self, batch: list[ModelInput]) -> dict[str, torch.Tensor]:
        input_ids, attention = pad_token_batch(batch, self.pad_token_id)
        out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention}
        if self.with_labels:
            labels = [item.label for item in batch]
            if any(label is None for label in labels):
                raise ValueError("a labeled batch contains an item with label=None")
            dtype = torch.long if self.label_dtype == "long" else torch.float
            out["labels"] = torch.tensor(labels, dtype=dtype)
        return out
