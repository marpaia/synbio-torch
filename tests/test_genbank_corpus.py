"""The GenBank corpus source (native sbol-rs import via the binding)."""

from __future__ import annotations

from pathlib import Path

import pytest

from synbiotorch.config import CorpusConfig
from synbiotorch.data.corpus import build_corpus
from synbiotorch.exceptions import ConfigError
from synbiotorch.sources.genbank import GenbankCorpus

DEMO = Path(__file__).parent.parent / "examples" / "data" / "demo_tu.gb"


def test_genbank_corpus_yields_design_with_features_and_graph():
    corpus = GenbankCorpus(DEMO, namespace="https://example.org/demo")
    designs = list(corpus)
    assert len(designs) == 1
    design = designs[0]
    assert design.record_class.endswith("Component")
    assert design.sequence is not None and len(design.sequence.elements) == 120
    assert design.features
    # The composition graph carries hasFeature edges the graph encoder consumes.
    assert design.neighbors is not None
    assert any(e.predicate.endswith("hasFeature") for e in design.neighbors.edges)


def test_genbank_requires_namespace():
    with pytest.raises(ConfigError):
        CorpusConfig(source="genbank", path=str(DEMO))


def test_build_corpus_dispatches_genbank():
    config = CorpusConfig(source="genbank", path=str(DEMO), namespace="https://example.org/demo")
    corpus = build_corpus(config)
    assert isinstance(corpus, GenbankCorpus)
    assert len(list(corpus)) == 1
