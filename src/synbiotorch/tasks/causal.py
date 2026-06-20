"""Causal-language-model pretraining objective.

Next-token cross-entropy. The collator has already shifted targets into place, so
the loss is a plain token cross-entropy ignoring ``-100`` — identical in shape to
the MLM path. ``val_loss`` is the next-token cross-entropy (perplexity is
``exp(val_loss)``); ``epoch_metrics`` reports next-token accuracy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from synbiotorch.datasets.mlm_collator import IGNORE_INDEX


class CausalLMTask:
    label_dtype = "long"

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=IGNORE_INDEX)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def epoch_metrics(self, preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        scored = labels != IGNORE_INDEX
        total = int(scored.sum())
        if total == 0:
            return {"next_token_accuracy": 0.0}
        correct = int((preds[scored] == labels[scored]).sum())
        return {"next_token_accuracy": correct / total}

    @property
    def primary_metric(self) -> tuple[str, str]:
        return ("loss", "min")
