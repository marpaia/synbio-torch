"""SBOL custom annotations surface as raw fields.

The Kosuri composability corpus rides a numeric label and two split assignments
on each construct as SBOL annotation triples. Labels are read through
``label_key``; the split values must reach ``Design.raw`` so the ``column`` split
reads an SBOL annotation the same way it reads a table column.
"""

from __future__ import annotations

from pathlib import Path

from synbiotorch.datasets.splits import split_from_assignments
from synbiotorch.sources.sbol import SbolFileCorpus

SBOL3 = "http://sbols.org/v3#"
NS = "https://synbiotorch.test/kosuri/"
SO_PROMOTER = "https://identifiers.org/SO:0000167"
SO_RBS = "https://identifiers.org/SO:0000139"


def _construct(cid: str, pseq: str, rseq: str, log_prot: float, split: str) -> str:
    comp = f"{NS}construct/{cid}"
    seq = f"{comp}/sequence"

    def t(s: str, p: str, o: str) -> str:
        return f"<{s}> <{p}> {o} ."

    def ref(iri: str) -> str:
        return f"<{iri}>"

    rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    xsd_int = "http://www.w3.org/2001/XMLSchema#integer"
    lines = [
        t(comp, rdf_type, ref(f"{SBOL3}Component")),
        t(comp, f"{SBOL3}hasSequence", ref(seq)),
        t(seq, rdf_type, ref(f"{SBOL3}Sequence")),
        t(seq, f"{SBOL3}elements", f'"{pseq + rseq}"'),
    ]
    for role_name, role_iri, part_id, start, end in (
        ("promoter", SO_PROMOTER, "P1", 1, len(pseq)),
        ("rbs", SO_RBS, "R1", len(pseq) + 1, len(pseq) + len(rseq)),
    ):
        feat = f"{comp}/{role_name}"
        loc = f"{feat}/loc0"
        lines += [
            t(comp, f"{SBOL3}hasFeature", ref(feat)),
            t(feat, rdf_type, ref(f"{SBOL3}SubComponent")),
            t(feat, f"{SBOL3}instanceOf", ref(f"{NS}part/{role_name}/{part_id}")),
            t(feat, f"{SBOL3}role", ref(role_iri)),
            t(feat, f"{SBOL3}hasLocation", ref(loc)),
            t(loc, rdf_type, ref(f"{SBOL3}Range")),
            t(loc, f"{SBOL3}start", f'"{start}"^^<{xsd_int}>'),
            t(loc, f"{SBOL3}end", f'"{end}"^^<{xsd_int}>'),
        ]
    lines += [
        t(comp, f"{NS}prot", f'"{log_prot:.6f}"'),
        t(comp, f"{NS}split_random", f'"{split}"'),
    ]
    return "\n".join(lines)


def _corpus(tmp_path: Path) -> Path:
    doc = "\n".join(
        [
            _construct("cA", "TTGACAGCTAGC", "AGGAGGACAT", 3.0, "train"),
            _construct("cB", "TTGACGTTTTAA", "AGGAGGTTAT", 4.5, "test"),
            _construct("cC", "TTGACATATAGC", "AGGAGGCCAT", 2.5, "val"),
        ]
    )
    path = tmp_path / "kosuri.ttl"
    path.write_text(doc + "\n")
    return path


def test_label_and_split_annotations_round_trip(tmp_path):
    designs = sorted(SbolFileCorpus(_corpus(tmp_path), label_key="prot"), key=lambda d: d.iri)
    assert len(designs) == 3

    a = designs[0]
    assert a.label == 3.0
    assert a.raw.get("split_random") == "train"
    assert len(a.features) == 2
    assert a.features[0].roles == (SO_PROMOTER,)
    assert a.features[1].roles == (SO_RBS,)
    # Composition graph: root + two features + two parts + sequence.
    assert a.neighbors is not None and len(a.neighbors.nodes) == 6


def test_column_split_reads_sbol_annotation(tmp_path):
    designs = list(SbolFileCorpus(_corpus(tmp_path), label_key="prot"))
    assignments = [d.raw.get("split_random") for d in designs]
    assert set(assignments) == {"train", "val", "test"}
    split = split_from_assignments(assignments)
    assert len(split.train) == 1
    assert len(split.val) == 1
    assert len(split.test) == 1


def test_annotations_do_not_overwrite_core_fields(tmp_path):
    # The label predicate is also surfaced as a raw field, but core keys like the
    # sequence and features are untouched.
    design = next(iter(SbolFileCorpus(_corpus(tmp_path), label_key="prot")))
    assert design.raw.get("prot") is not None
    assert design.sequence is not None and design.sequence.elements
    assert design.raw["iri"].startswith(NS)
