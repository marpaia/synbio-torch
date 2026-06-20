"""Masked-language-model pretraining objective."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from synbiotorch.datasets.mlm_collator import IGNORE_INDEX


class MlmTask:
    """Token-level cross-entropy over masked positions.

    The model emits ``[batch, seq, vocab]`` logits; the loss is averaged over the
    masked positions (``labels == -100`` ignored). ``val_loss`` is the masked
    cross-entropy, so perplexity is ``exp(val_loss)``; ``epoch_metrics`` also
    reports masked-token accuracy.
    """

    label_dtype = "long"

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=IGNORE_INDEX)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def epoch_metrics(self, preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        scored = labels != IGNORE_INDEX
        total = int(scored.sum())
        if total == 0:
            return {"masked_accuracy": 0.0}
        correct = int((preds[scored] == labels[scored]).sum())
        return {"masked_accuracy": correct / total}

    @property
    def primary_metric(self) -> tuple[str, str]:
        return ("loss", "min")
