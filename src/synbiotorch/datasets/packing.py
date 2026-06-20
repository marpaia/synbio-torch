"""Token packing for language-model pretraining.

Concatenates tokenized documents into fixed-length blocks with no padding, so
every position carries signal. The unit of training becomes a block, not a
document: a document may span block boundaries and a block may hold pieces of
several documents. The trailing remainder shorter than one block is dropped, as
is standard for packed LM corpora.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from torch.utils.data import IterableDataset

from synbiotorch.datasets.streaming import iter_split_records
from synbiotorch.encoders.base import ModelInput
from synbiotorch.tokenize.base import Tokenizer
from synbiotorch.types import Design


class PackedDataset(IterableDataset):
    """Streams a corpus, tokenizes content, and yields fixed ``block_size`` blocks.

    Inherits the streaming dataset's worker partitioning and hash-split filtering
    so it composes with a multi-worker DataLoader exactly as the unpacked path
    does.
    """

    def __init__(
        self,
        source: Iterable[Design],
        tokenizer: Tokenizer,
        *,
        block_size: int,
        which: str | None = None,
        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ) -> None:
        self._source = source
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._which = which
        self._ratios = ratios
        self._seed = seed

    def __iter__(self) -> Iterator[ModelInput]:
        block_size = self._block_size
        buffer: list[int] = []
        for obj in iter_split_records(self._source, self._which, self._ratios, self._seed):
            if obj.sequence is None or not obj.sequence.elements:
                continue
            buffer.extend(self._tokenizer.tokenize_content(obj.sequence.elements))
            while len(buffer) >= block_size:
                block = buffer[:block_size]
                del buffer[:block_size]
                yield ModelInput(input_ids=block, attention_mask=[1] * block_size, label=None)
