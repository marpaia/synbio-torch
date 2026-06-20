"""Adapter for any pretrained HuggingFace tokenizer.

Wraps a HuggingFace ``AutoTokenizer`` behind the library's Tokenizer protocol
so a pretrained backbone's own tokenizer (DNABERT-2, Nucleotide Transformer, or
any encoder that accepts a raw sequence string) is interchangeable with the
k-mer and character tokenizers. The underlying tokenizer is loaded lazily so
constructing the object does not trigger a model download.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from transformers import AutoTokenizer

from synbiotorch.tokenize.base import Encoded


class HFTokenizer:
    def __init__(self, model_name: str = "zhihan1996/DNABERT-2-117M", max_length: int = 512) -> None:
        self.model_name = model_name
        self._max_length = max_length

    @cached_property
    def _hf(self) -> Any:
        return AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @property
    def vocab_size(self) -> int:
        return int(self._hf.vocab_size)

    @property
    def pad_token_id(self) -> int:
        return int(self._hf.pad_token_id)

    @property
    def mask_token_id(self) -> int | None:
        return self._hf.mask_token_id

    @property
    def special_token_ids(self) -> frozenset[int]:
        return frozenset(self._hf.all_special_ids)

    @property
    def max_length(self) -> int:
        return self._max_length

    def tokenize_content(self, sequence: str) -> list[int]:
        return list(self._hf(sequence, add_special_tokens=False, truncation=False)["input_ids"])

    def encode(self, sequence: str) -> Encoded:
        out = self._hf(
            sequence,
            truncation=True,
            max_length=self._max_length,
            add_special_tokens=True,
        )
        return Encoded(input_ids=list(out["input_ids"]), attention_mask=list(out["attention_mask"]))

    def decode(self, ids: list[int]) -> str:
        return str(self._hf.decode(ids, skip_special_tokens=True))
