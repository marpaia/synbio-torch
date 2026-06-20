"""Synthetic SBOL fixture generator."""

from __future__ import annotations

from synbiotorch.sources.synthetic import SyntheticCorpus, generate_components, write_sbol_turtle
from synbiotorch.types import local_name


def test_generation_is_deterministic():
    a = generate_components(10, seed=42)
    b = generate_components(10, seed=42)
    assert [o.iri for o in a] == [o.iri for o in b]
    assert [o.sequence.elements for o in a] == [o.sequence.elements for o in b]
    assert [o.label for o in a] == [o.label for o in b]


def test_components_have_sequence_features_and_graph():
    comp = generate_components(1, seed=1)[0]
    # Four features: promoter, RBS, CDS, terminator.
    assert len(comp.features) == 4
    roles = {r for f in comp.features for r in f.roles}
    assert len(roles) == 4
    # Feature ranges tile the parent sequence without gaps.
    ends = [f.locations[0].end for f in comp.features]
    assert ends[-1] == len(comp.sequence.elements)
    assert comp.features[0].locations[0].start == 1
    # Composition graph references the component, its features, parts, sequence.
    assert comp.neighbors is not None
    assert comp.neighbors.root_iri == comp.iri
    predicates = {e.predicate for e in comp.neighbors.edges}
    assert any(p.endswith("hasFeature") for p in predicates)
    assert any(p.endswith("instanceOf") for p in predicates)
    assert any(p.endswith("hasSequence") for p in predicates)


def test_parts_are_reused_across_components():
    comps = generate_components(40, seed=3)
    part_iris = {f.instance_of for c in comps for f in c.features}
    # The catalog is small, so 40 components must reuse parts.
    assert len(part_iris) < 40


def test_labels_track_promoter_identity():
    comps = generate_components(40, seed=5)
    # Same promoter part -> same label (strength is a function of the promoter).
    by_promoter: dict[str, set[float]] = {}
    for c in comps:
        promoter = next(f.instance_of for f in c.features if any("SO:0000167" in r for r in f.roles))
        by_promoter.setdefault(promoter, set()).add(round(c.label, 6))
    for labels in by_promoter.values():
        assert len(labels) == 1


def test_synthetic_corpus_is_iterable_and_fingerprinted():
    corpus = SyntheticCorpus(8, seed=2)
    assert len(list(corpus)) == 8
    assert corpus.fingerprint() == SyntheticCorpus(8, seed=2).fingerprint()
    assert corpus.fingerprint() != SyntheticCorpus(8, seed=3).fingerprint()


def test_sbol_turtle_roundtrips_through_local_corpus(tmp_path):
    from synbiotorch.sources.sbol import SbolFileCorpus

    comps = generate_components(5, seed=7)
    path = write_sbol_turtle(comps, tmp_path / "synthetic.ttl")
    # One record per Component, with sequence, features, and composition graph.
    parsed = list(SbolFileCorpus(path))
    assert len(parsed) == len(comps)

    elements = {o.sequence.elements for o in parsed if o.sequence}
    expected = {c.sequence.elements for c in comps}
    assert expected <= elements

    by_iri = {o.iri: o for o in parsed}
    for original in comps:
        got = by_iri[original.iri]
        # Each component round-trips its four annotated features with roles and ranges.
        assert len(got.features) == len(original.features) == 4
        assert {r for f in got.features for r in f.roles} == {r for f in original.features for r in f.roles}
        assert all(f.locations and f.locations[0].start is not None for f in got.features)
        assert {f.instance_of for f in got.features} == {f.instance_of for f in original.features}
        # The composition graph carries the hasFeature / instanceOf / hasSequence edges.
        predicates = {local_name(e.predicate) for e in got.neighbors.edges}
        assert {"hasFeature", "instanceOf", "hasSequence"} <= predicates
