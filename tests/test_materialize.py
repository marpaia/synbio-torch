from __future__ import annotations

import hashlib

from synbiotorch.data.materialize import materialize
from synbiotorch.types import Alphabet, Design, Sequence


class _FakeCorpus:
    """A minimal in-memory Corpus implementation for testing materialization."""

    def __init__(self, objects):
        self._objects = objects

    def __iter__(self):
        return iter(self._objects)

    def fingerprint(self):
        h = hashlib.sha256()
        for o in self._objects:
            h.update(o.iri.encode())
        return h.hexdigest()[:16]


def _objects():
    return [
        Design(
            iri=f"https://example.org/s{i}",
            record_class="http://sbols.org/v3#Sequence",
            sequence=Sequence(elements="ACGT" * (i + 1), alphabet=Alphabet.DNA),
            label=float(i),
        )
        for i in range(3)
    ]


def test_materialize_roundtrip(tmp_path):
    corpus = _FakeCorpus(_objects())
    mat = materialize(corpus, tmp_path)
    assert len(mat) == 3
    read_back = mat.read_all()
    assert read_back[1].sequence.elements == "ACGTACGT"
    assert read_back[2].label == 2.0
    assert mat.labels() == [0.0, 1.0, 2.0]


def test_materialize_is_cached(tmp_path):
    corpus = _FakeCorpus(_objects())
    first = materialize(corpus, tmp_path)
    second = materialize(corpus, tmp_path)
    # Same fingerprint dir reused, not rewritten under a new name.
    assert first.path == second.path
    assert first.fingerprint == second.fingerprint
