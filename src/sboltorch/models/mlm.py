"""Masked-language-model wrapper supporting from-scratch and continued pretraining.

Both paths share the same module and forward signature; they differ only in how
the underlying ``AutoModelForMaskedLM`` is constructed:

- from-scratch: instantiate an architecture from ``arch`` + the tokenizer vocab
  (pretrain a DNA LM on the SBOL corpus with our own k-mer/char tokenizer).
- continued: load pretrained weights by hub id or local path.

After pretraining, ``save_pretrained`` writes the model so a later supervised run
can point ``model.backbone`` at the directory and load it as a plain encoder.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, PreTrainedModel

from sboltorch.config import ModelConfig
from sboltorch.models.backbone import attn_kwargs, from_scratch_config


class MaskedLMModel(nn.Module):
    def __init__(self, lm: PreTrainedModel) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.lm(input_ids=input_ids, attention_mask=attention_mask).logits

    def save_pretrained(self, directory: str | Path) -> None:
        self.lm.save_pretrained(str(directory))


def build_mlm_model(model_config: ModelConfig, *, vocab_size: int, pad_token_id: int) -> MaskedLMModel:
    if model_config.from_scratch:
        config = from_scratch_config(model_config, vocab_size=vocab_size, pad_token_id=pad_token_id)
        lm = AutoModelForMaskedLM.from_config(config, **attn_kwargs(model_config))
    else:
        lm = AutoModelForMaskedLM.from_pretrained(
            model_config.backbone, trust_remote_code=True, **attn_kwargs(model_config)
        )
    return MaskedLMModel(lm)
