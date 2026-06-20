"""The Tokenizer protocol — one interface, swappable implementations.

SeqTrainer forked tokenization across every approach (BPE in one notebook,
k-mer counts in another, one-hot in a third). Here every tokenizer produces the
same ``Encoded`` shape, so encoders and models never care which one is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Encoded:
    """Token ids and attention mask for a single sequence."""

    input_ids: list[int]
    attention_mask: list[int]


@runtime_checkable
class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def pad_token_id(self) -> int: ...

    @property
    def mask_token_id(self) -> int | None: ...

    @property
    def special_token_ids(self) -> frozenset[int]:
        """Ids that must never be masked or treated as content (pad/cls/sep/mask/...)."""
        ...

    @property
    def max_length(self) -> int: ...

    def tokenize_content(self, sequence: str) -> list[int]:
        """Token ids for the sequence with no special wrapping or truncation."""
        ...

    def encode(self, sequence: str) -> Encoded: ...

    def decode(self, ids: list[int]) -> str:
        """Reconstruct a sequence string from token ids, dropping special tokens."""
        ...


def build_tokenizer(config: "object") -> Tokenizer:  # noqa: ANN001 - avoid config import cycle
    """Construct the tokenizer named by ``config.kind``."""
    from ..config import TokenizerConfig

    assert isinstance(config, TokenizerConfig)
    if config.kind == "kmer":
        from .kmer import KmerTokenizer

        return KmerTokenizer(k=config.k, stride=config.stride, max_length=config.max_length)
    if config.kind == "char":
        from .char import CharTokenizer

        return CharTokenizer(max_length=config.max_length)
    if config.kind == "hf":
        from .hf import HFTokenizer

        return HFTokenizer(model_name=config.model_name, max_length=config.max_length)
    raise ValueError(f"unknown tokenizer kind: {config.kind}")  # pragma: no cover
