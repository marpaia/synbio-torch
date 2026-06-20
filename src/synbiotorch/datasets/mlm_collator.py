"""Masking collator for masked-language-model pretraining.

Applies BERT-style dynamic masking to each batch: ~``mlm_probability`` of the
non-special content tokens are selected; of those, 80% become ``<mask>``, 10%
are replaced with a random token, and 10% are left unchanged. Unselected
positions get a label of -100 so they are ignored by the loss.
"""

from __future__ import annotations

import torch

from synbiotorch.datasets.dataset import pad_token_batch
from synbiotorch.encoders.base import ModelInput
from synbiotorch.tokenize.base import Tokenizer

IGNORE_INDEX = -100


class MlmCollator:
    def __init__(self, tokenizer: Tokenizer, *, mlm_probability: float = 0.15, seed: int | None = None) -> None:
        if tokenizer.mask_token_id is None:
            raise ValueError("tokenizer has no mask token; cannot do MLM")
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = int(tokenizer.mask_token_id)
        self.vocab_size = tokenizer.vocab_size
        self.mlm_probability = mlm_probability
        self._special_ids = torch.tensor(sorted(tokenizer.special_token_ids), dtype=torch.long)
        # A fixed seed makes masking reproducible (tests); None lets it vary per
        # batch/epoch, which is what real pretraining wants.
        self._seed = seed

    def __call__(self, batch: list[ModelInput]) -> dict[str, torch.Tensor]:
        input_ids, attention = pad_token_batch(batch, self.pad_token_id)
        gen = torch.Generator().manual_seed(self._seed) if self._seed is not None else None
        labels = input_ids.clone()

        maskable = ~torch.isin(input_ids, self._special_ids) & attention.bool()
        prob = torch.where(maskable, self.mlm_probability, 0.0)
        selected = torch.bernoulli(prob, generator=gen).bool()
        # Guarantee at least one target per row that has content, so the loss is
        # never computed over an all-ignored batch (which yields NaN).
        for row in range(selected.shape[0]):
            if maskable[row].any() and not selected[row].any():
                candidates = maskable[row].nonzero(as_tuple=True)[0]
                selected[row, candidates[0]] = True

        labels[~selected] = IGNORE_INDEX

        replace = torch.bernoulli(torch.full(input_ids.shape, 0.8), generator=gen).bool() & selected
        input_ids[replace] = self.mask_token_id
        randomize = torch.bernoulli(torch.full(input_ids.shape, 0.5), generator=gen).bool() & selected & ~replace
        random_tokens = torch.randint(self.vocab_size, input_ids.shape, generator=gen, dtype=torch.long)
        input_ids[randomize] = random_tokens[randomize]
        # The remaining ~10% of selected positions keep their original token.

        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}
