"""Collator for causal-language-model pretraining.

Produces next-token targets aligned position-for-position with the inputs: the
target at position ``t`` is the token at ``t+1``. Shifting here (rather than in
the task) keeps the training loop's contract identical to MLM — ``predict`` is a
plain argmax and metrics are position-independent. The final position and any
padding targets are set to ``-100`` so the loss ignores them.
"""

from __future__ import annotations

import torch

from sboltorch.datasets.dataset import pad_token_batch
from sboltorch.datasets.mlm_collator import IGNORE_INDEX
from sboltorch.encoders.base import ModelInput


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[ModelInput]) -> dict[str, torch.Tensor]:
        input_ids, attention = pad_token_batch(batch, self.pad_token_id)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[:, :-1] = input_ids[:, 1:]
        # Never score predicting a pad token (the trailing remainder of a padded
        # batch); packed blocks have no pad, so only the last position is dropped.
        labels[labels == self.pad_token_id] = IGNORE_INDEX
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}
