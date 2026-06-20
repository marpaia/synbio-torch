"""Corpus sources: FASTA, SBOL, GenBank, synthetic, and the sbol-db REST client.

Every source normalizes to ``Design`` records behind the ``Corpus`` protocol, so
training code never branches on provenance.
"""

from __future__ import annotations

from synbiotorch.sources.fasta import FastaCorpus
from synbiotorch.sources.genbank import GenbankCorpus
from synbiotorch.sources.sbol import SbolFileCorpus, design_from_record, records_to_designs
from synbiotorch.sources.sbol_db import SbolDbClient
from synbiotorch.sources.synthetic import SyntheticCorpus, generate_components, write_sbol_turtle
from synbiotorch.sources.table import TableCorpus

__all__ = [
    "FastaCorpus",
    "SbolFileCorpus",
    "GenbankCorpus",
    "TableCorpus",
    "SbolDbClient",
    "SyntheticCorpus",
    "generate_components",
    "write_sbol_turtle",
    "records_to_designs",
    "design_from_record",
]
