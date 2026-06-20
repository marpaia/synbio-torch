from __future__ import annotations

from synbiotorch.sources.sbol import design_from_record
from synbiotorch.types import Alphabet, local_name


def test_local_name_handles_iri_and_curie():
    assert local_name("http://sbols.org/v3#elements") == "elements"
    assert local_name("https://identifiers.org/SO:0000167") == "0000167"
    assert local_name("sbol:Component") == "Component"


def test_alphabet_inference():
    assert Alphabet.from_encoding("https://identifiers.org/edam/protein") == Alphabet.PROTEIN
    assert Alphabet.from_encoding("http://example.org/rna") == Alphabet.RNA
    assert Alphabet.from_encoding(None) == Alphabet.DNA


def test_from_record_extracts_sequence_by_local_name(object_records):
    obj = design_from_record(object_records[0])
    assert obj.iri == "https://example.org/seqA"
    assert obj.sequence is not None
    assert obj.sequence.elements == "ACGTACGTACGT"
    assert obj.roles == ("SO:0000167",)


def test_from_record_without_sequence_is_none():
    record = {"iri": "x", "sbol_class": "http://sbols.org/v3#Component", "data": {}}
    obj = design_from_record(record)
    assert obj.sequence is None
