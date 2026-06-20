"""Data layer: the Corpus protocol and reproducible Parquet materialization.

Corpus *sources* live in :mod:`synbiotorch.sources`; this package holds the
source-neutral pipeline pieces that consume them.
"""

from __future__ import annotations

from synbiotorch.data.corpus import Corpus, build_corpus
from synbiotorch.data.materialize import MaterializedCorpus, materialize

__all__ = [
    "Corpus",
    "build_corpus",
    "MaterializedCorpus",
    "materialize",
]
