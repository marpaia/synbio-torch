"""Model construction: assemble a backbone + head (or an MLM model) from config."""

from __future__ import annotations

import torch.nn as nn

from sboltorch.config import ModelConfig, TaskConfig
from sboltorch.models.backbone import build_from_scratch_encoder, load_backbone
from sboltorch.models.causal import CausalLMModel, build_causal_model
from sboltorch.models.heads import ClassificationHead, RegressionHead
from sboltorch.models.mlm import MaskedLMModel, build_mlm_model
from sboltorch.models.sequence_model import SequenceModel

__all__ = [
    "build_model",
    "SequenceModel",
    "MaskedLMModel",
    "CausalLMModel",
    "RegressionHead",
    "ClassificationHead",
    "load_backbone",
]


def build_model(
    model_config: ModelConfig,
    task_config: TaskConfig,
    *,
    vocab_size: int | None = None,
    pad_token_id: int | None = None,
) -> nn.Module:
    """Build the model for the given task.

    - ``mlm`` → a MaskedLMModel (from-scratch needs ``vocab_size``/``pad_token_id``).
    - ``causal`` → a CausalLMModel decoder (from-scratch needs a decoder ``arch``,
      e.g. ``model_type: gpt2``).
    - ``frozen`` → a SequenceModel with a frozen backbone and a trainable head.
    - ``supervised`` → a SequenceModel fine-tuned end to end.
    """
    if task_config.kind == "mlm":
        if vocab_size is None or pad_token_id is None:
            raise ValueError("vocab_size and pad_token_id are required to build an MLM model")
        return build_mlm_model(model_config, vocab_size=vocab_size, pad_token_id=pad_token_id)

    if task_config.kind == "causal":
        if vocab_size is None or pad_token_id is None:
            raise ValueError("vocab_size and pad_token_id are required to build a causal LM model")
        return build_causal_model(model_config, vocab_size=vocab_size, pad_token_id=pad_token_id)

    if model_config.from_scratch:
        if vocab_size is None or pad_token_id is None:
            raise ValueError("vocab_size and pad_token_id are required for a from-scratch model")
        backbone, hidden_size = build_from_scratch_encoder(
            model_config, vocab_size=vocab_size, pad_token_id=pad_token_id
        )
    else:
        backbone, hidden_size = load_backbone(model_config.backbone, model_config)
    head: nn.Module
    if task_config.objective == "classification":
        assert task_config.num_classes is not None  # enforced by TaskConfig validation
        head = ClassificationHead(hidden_size, task_config.num_classes, model_config.dropout)
    else:
        head = RegressionHead(hidden_size, model_config.dropout)
    return SequenceModel(backbone, head, freeze_backbone=task_config.kind == "frozen")
