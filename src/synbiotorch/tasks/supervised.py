"""Supervised regression / classification objective."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SupervisedTask:
    """Regression (MSE) or classification (cross-entropy) with optional target transform.

    ``target_transform='log1p'`` trains in log space (common for expression and
    fitness targets, as in the original SeqTrainer DNABERT path) and reports
    metrics back in the original space via ``expm1``.
    """

    def __init__(
        self,
        objective: str = "regression",
        num_classes: int | None = None,
        target_transform: str = "none",
    ) -> None:
        self.objective = objective
        self.num_classes = num_classes
        self.target_transform = target_transform
        self.label_dtype = "long" if objective == "classification" else "float"

    def _forward_transform(self, labels: torch.Tensor) -> torch.Tensor:
        if self.objective == "regression" and self.target_transform == "log1p":
            return torch.log1p(labels)
        return labels

    def _inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.objective == "regression" and self.target_transform == "log1p":
            return np.expm1(values)
        return values

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.objective == "classification":
            return F.cross_entropy(logits, labels)
        return F.mse_loss(logits, self._forward_transform(labels))

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        if self.objective == "classification":
            return logits.argmax(dim=-1)
        return logits

    def epoch_metrics(self, preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        if self.objective == "classification":
            accuracy = float((preds == labels).mean()) if len(labels) else 0.0
            return {"accuracy": accuracy}
        preds = self._inverse_transform(preds)
        errors = preds - labels
        mae = float(np.abs(errors).mean()) if len(labels) else 0.0
        mse = float((errors**2).mean()) if len(labels) else 0.0
        var = float(((labels - labels.mean()) ** 2).mean()) if len(labels) else 0.0
        r2 = 1.0 - mse / var if var > 0 else 0.0
        return {"mae": mae, "mse": mse, "r2": r2}

    @property
    def primary_metric(self) -> tuple[str, str]:
        return ("accuracy", "max") if self.objective == "classification" else ("mae", "min")
