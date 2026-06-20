"""Parse real-world SBOL files (SynBioDex SBOLTestSuite) through the native SBOL binding.

These complement the synthetic fixtures with genuine SBOL2/SBOL3 documents,
covering RDF/XML, Turtle, and N-Triples serializations and both sequence-bearing
and abstract (sequence-free) designs. See fixtures/sbol/PROVENANCE.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synbiotorch.sources.sbol import SbolFileCorpus

FIXTURES = Path(__file__).parent / "fixtures" / "sbol"
ALL_FILES = sorted(p for p in FIXTURES.rglob("*") if p.suffix in {".ttl", ".nt", ".xml"})

# These vendored fixtures may be absent (e.g. not committed due to licensing);
# skip rather than fail when they are not present.
pytestmark = pytest.mark.skipif(not ALL_FILES, reason="SBOL fixtures not present (see fixtures/sbol/PROVENANCE.md)")


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: str(p.relative_to(FIXTURES)))
def test_every_fixture_parses_without_error(path):
    # Robustness: real files (including abstract designs) parse cleanly.
    objects = list(SbolFileCorpus(path, namespace="https://example.org/ns"))
    assert isinstance(objects, list)


def test_sbol2_plasmid_sequence():
    objs = list(SbolFileCorpus(FIXTURES / "sbol2" / "pICH44179.xml", namespace="https://example.org/ns"))
    seqs = [o for o in objs if o.sequence and o.sequence.elements]
    assert len(seqs) == 1
    assert len(seqs[0].sequence.elements) == 2307
    assert set(seqs[0].sequence.elements.upper()) <= set("ACGTN")


def test_sbol3_device_has_multiple_sequences():
    objs = list(SbolFileCorpus(FIXTURES / "sbol3" / "BBa_F2620_PoPSReceiver.ttl", namespace="https://example.org/ns"))
    seqs = [o for o in objs if o.sequence and o.sequence.elements]
    assert len(seqs) == 10


def test_abstract_design_has_no_sequences():
    # toggle_switch is composition-only (components/interactions, no elements);
    # the parser returns no sequences rather than failing.
    objs = list(SbolFileCorpus(FIXTURES / "sbol3" / "toggle_switch.ttl", namespace="https://example.org/ns"))
    assert all(o.sequence is None or not o.sequence.elements for o in objs)


def test_sbol3_components_carry_features_and_graph():
    # SBOL3 documents yield one record per Component, with features and a
    # composition graph the structure-aware and graph encoders consume.
    objs = list(SbolFileCorpus(FIXTURES / "sbol3" / "BBa_F2620_PoPSReceiver.ttl", namespace="https://example.org/ns"))
    comps = [o for o in objs if o.record_class.endswith("Component")]
    assert comps and len(comps) == len(objs)
    featured = [o for o in comps if o.features]
    assert featured, "expected at least one Component with annotated features"
    feat = featured[0].features[0]
    assert feat.roles or feat.instance_of  # a feature names a role and/or an instance
    assert featured[0].neighbors is not None and featured[0].neighbors.nodes


def test_abstract_design_components_have_subcomponent_graph():
    # An abstract design has no sequence but still yields a useful composition
    # graph: SubComponents wired by instanceOf into the graph slice.
    objs = list(SbolFileCorpus(FIXTURES / "sbol3" / "toggle_switch.ttl", namespace="https://example.org/ns"))
    edges = {e.predicate for o in objs if o.neighbors for e in o.neighbors.edges}
    assert any(p.endswith("hasFeature") for p in edges)
    assert any(p.endswith("instanceOf") for p in edges)
