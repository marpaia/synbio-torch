"""Construct transformer backbones — pretrained or from scratch.

The from-scratch path is shared by the encoder, MLM, and causal builders: one
`AutoConfig` shaped from `ArchConfig`, plus the attention implementation (SDPA by
default, so FlashAttention kernels are used on CUDA without an extra dependency).
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig

from sboltorch.config import ModelConfig
from sboltorch.exceptions import ConfigError


def from_scratch_config(model_config: ModelConfig, *, vocab_size: int, pad_token_id: int) -> PretrainedConfig:
    """Build the HuggingFace config for a from-scratch model from ``ArchConfig``."""
    arch = model_config.arch
    extra: dict[str, Any] = {}
    if arch.rope_theta is not None:
        extra["rope_theta"] = arch.rope_theta
    return AutoConfig.for_model(
        arch.model_type,
        vocab_size=vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=arch.num_hidden_layers,
        num_attention_heads=arch.num_attention_heads,
        intermediate_size=arch.intermediate_size,
        max_position_embeddings=arch.max_position_embeddings,
        pad_token_id=pad_token_id,
        **extra,
    )


def attn_kwargs(model_config: ModelConfig) -> dict[str, str]:
    """Keyword args selecting the attention implementation for model construction."""
    return {"attn_implementation": model_config.arch.attn_implementation}


def load_backbone(model_name: str, model_config: ModelConfig | None = None) -> tuple[nn.Module, int]:
    """Return ``(backbone_module, hidden_size)`` for a HuggingFace encoder model.

    ``model_name`` may be a hub id or a local directory (e.g. a backbone written
    out by an MLM pretraining run).
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    extra = attn_kwargs(model_config) if model_config is not None else {}
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, **extra)
    hidden_size = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
    if hidden_size is None:
        raise ConfigError(f"could not determine hidden size for backbone {model_name}")
    return model, int(hidden_size)


def build_from_scratch_encoder(
    model_config: ModelConfig, *, vocab_size: int, pad_token_id: int
) -> tuple[nn.Module, int]:
    """Instantiate an untrained encoder sized to a given vocab (e.g. our k-mer vocab)."""
    config = from_scratch_config(model_config, vocab_size=vocab_size, pad_token_id=pad_token_id)
    return AutoModel.from_config(config, **attn_kwargs(model_config)), model_config.hidden_size
