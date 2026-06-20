"""The tabular (CSV/TSV) corpus source for labeled sequence datasets."""

from __future__ import annotations

import pytest

from synbiotorch.config import CorpusConfig
from synbiotorch.data.corpus import build_corpus
from synbiotorch.exceptions import ConfigError
from synbiotorch.sources.table import TableCorpus
from synbiotorch.types import Alphabet

CSV = """id,sequence,activity
p1,ACGTACGTAC,12.5
p2,TTTTGGGGCC,3.0
p3,,9.9
p4,ACACGTGTAC,7.25
"""

TSV = "sequence\tlabel\nACGT\t1\nGGCC\t0\n"


def test_csv_reads_sequence_label_and_id(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(CSV)
    corpus = TableCorpus(path, sequence_column="sequence", label_column="activity", id_column="id")
    designs = list(corpus)
    # The empty-sequence row (p3) is skipped.
    assert [d.display_id for d in designs] == ["p1", "p2", "p4"]
    assert designs[0].sequence.elements == "ACGTACGTAC"
    assert designs[0].label == 12.5
    assert designs[0].sequence.alphabet == Alphabet.DNA


def test_tsv_delimiter_and_synthetic_id(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text(TSV)
    designs = list(TableCorpus(path, sequence_column="sequence", label_column="label"))
    assert len(designs) == 2
    assert designs[0].iri == "data:0"
    assert designs[0].label == 1


def test_protein_alphabet_override(tmp_path):
    path = tmp_path / "prot.csv"
    path.write_text("sequence\nMKWVTFISLL\n")
    designs = list(TableCorpus(path, sequence_column="sequence", alphabet="protein"))
    assert designs[0].sequence.alphabet == Alphabet.PROTEIN


def test_build_corpus_table_requires_sequence_column():
    with pytest.raises(ConfigError):
        CorpusConfig(source="table", path="x.csv")


def test_build_corpus_dispatches_table(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(CSV)
    config = CorpusConfig(source="table", path=str(path), sequence_column="sequence", label_column="activity")
    corpus = build_corpus(config)
    assert isinstance(corpus, TableCorpus)
    assert len(list(corpus)) == 3
