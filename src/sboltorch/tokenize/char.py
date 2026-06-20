"""Character-level tokenizer over the IUPAC nucleotide alphabet."""

from __future__ import annotations

from sboltorch.tokenize.base import Encoded

# Full IUPAC nucleotide alphabet so ambiguous bases get real, distinct tokens.
_ALPHABET = "ACGTUNRYSWKMBDHV"
_SPECIAL = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]


class CharTokenizer:
    def __init__(self, max_length: int = 512) -> None:
        self._max_length = max_length
        self._vocab = {tok: i for i, tok in enumerate(_SPECIAL + list(_ALPHABET))}
        self._id_to_tok = {i: tok for tok, i in self._vocab.items()}
        self._unk = self._vocab["<unk>"]
        self._special_ids = frozenset(self._vocab[tok] for tok in _SPECIAL)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_token_id(self) -> int:
        return self._vocab["<pad>"]

    @property
    def mask_token_id(self) -> int | None:
        return self._vocab["<mask>"]

    @property
    def special_token_ids(self) -> frozenset[int]:
        return self._special_ids

    @property
    def max_length(self) -> int:
        return self._max_length

    def tokenize_content(self, sequence: str) -> list[int]:
        return [self._vocab.get(base, self._unk) for base in sequence.upper()]

    def encode(self, sequence: str) -> Encoded:
        content = self.tokenize_content(sequence)[: self._max_length - 2]
        ids = [self._vocab["<cls>"], *content, self._vocab["<sep>"]]
        return Encoded(input_ids=ids, attention_mask=[1] * len(ids))

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self._id_to_tok[i] for i in ids if i not in self._special_ids and self._id_to_tok.get(i, "<unk>") != "<unk>"
        )
