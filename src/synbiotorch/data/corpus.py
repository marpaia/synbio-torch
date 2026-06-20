"""The Corpus protocol — the single interface training code reads data through."""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from synbiotorch.config import CorpusConfig
from synbiotorch.types import Design


@runtime_checkable
class Corpus(Protocol):
    """A stream of normalized Design records from any source."""

    def __iter__(self) -> Iterator[Design]: ...

    def fingerprint(self) -> str:
        """A stable content hash identifying this corpus for caching/reproducibility."""
        ...


def build_corpus(config: CorpusConfig) -> Corpus:
    """Construct the corpus implementation named by ``config.source``."""
    if config.source == "sbol_db":
        from synbiotorch.sources.sbol_db import SbolDbClient

        return SbolDbClient.from_config(config)
    if config.source == "fasta":
        from synbiotorch.sources.fasta import FastaCorpus

        return FastaCorpus.from_config(config)
    if config.source == "sbol":
        from synbiotorch.sources.sbol import SbolFileCorpus

        return SbolFileCorpus.from_config(config)
    if config.source == "genbank":
        from synbiotorch.sources.genbank import GenbankCorpus

        return GenbankCorpus.from_config(config)
    if config.source == "table":
        from synbiotorch.sources.table import TableCorpus

        return TableCorpus.from_config(config)
    if config.source == "synthetic":
        from synbiotorch.sources.synthetic import SyntheticCorpus

        return SyntheticCorpus(config.n, seed=config.synthetic_seed, with_labels=config.label_key is not None)
    raise ValueError(f"unknown corpus source: {config.source}")  # pragma: no cover
