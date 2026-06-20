"""Causal-LM (decoder) wrapper supporting from-scratch and continued pretraining.

Mirrors the MLM wrapper but over ``AutoModelForCausalLM``, so the generative path
shares the library's model-construction and backbone-reuse conventions:

- from-scratch: instantiate a decoder architecture (e.g. ``model_type: gpt2``)
  from ``arch`` + the tokenizer vocab.
- continued: load pretrained weights by hub id or local path.

After pretraining, ``save_pretrained`` writes the model so generation and later
runs can point ``model.backbone`` at the directory.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from sboltorch.config import ModelConfig
from sboltorch.models.backbone import attn_kwargs, from_scratch_config


class CausalLMModel(nn.Module):
    def __init__(self, lm: PreTrainedModel) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.lm(input_ids=input_ids, attention_mask=attention_mask).logits

    def save_pretrained(self, directory: str | Path) -> None:
        self.lm.save_pretrained(str(directory))


def build_causal_model(model_config: ModelConfig, *, vocab_size: int, pad_token_id: int) -> CausalLMModel:
    if model_config.from_scratch:
        config = from_scratch_config(model_config, vocab_size=vocab_size, pad_token_id=pad_token_id)
        lm = AutoModelForCausalLM.from_config(config, **attn_kwargs(model_config))
    else:
        lm = AutoModelForCausalLM.from_pretrained(
            model_config.backbone, trust_remote_code=True, **attn_kwargs(model_config)
        )
    return CausalLMModel(lm)
