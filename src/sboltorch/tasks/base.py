"""The Task protocol — the training objective plug point.

A Task owns the loss, the label dtype, and the metrics. The backbone-freezing
decision lives in model construction, so 'frozen' and 'supervised' share one
SupervisedTask implementation and differ only in how the model is built.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Task(Protocol):
    label_dtype: str

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...

    def predict(self, logits: torch.Tensor) -> torch.Tensor: ...

    def epoch_metrics(self, preds: np.ndarray, labels: np.ndarray) -> dict[str, float]: ...

    @property
    def primary_metric(self) -> tuple[str, str]:
        """Return ``(metric_name, mode)`` where mode is 'min' or 'max'."""
        ...


def build_task(task_config: "object") -> Task:
    """Construct the task named by ``task_config.kind``."""
    from ..config import TaskConfig

    assert isinstance(task_config, TaskConfig)
    if task_config.kind in ("supervised", "frozen"):
        from .supervised import SupervisedTask

        return SupervisedTask(
            objective=task_config.objective,
            num_classes=task_config.num_classes,
            target_transform=task_config.target_transform,
        )
    if task_config.kind == "mlm":
        from .mlm import MlmTask

        return MlmTask()
    if task_config.kind == "causal":
        from .causal import CausalLMTask

        return CausalLMTask()
    raise NotImplementedError(f"task kind '{task_config.kind}' is not implemented yet")
