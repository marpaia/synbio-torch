"""Synthetic SBOL fixture generator.

Produces deterministic synthetic transcriptional units — a promoter, RBS, CDS,
and terminator composed into a parent Component — as rich ``Design`` records:
sequence, features (sub-components with Range locations, roles, orientation), and
a composition ``GraphSlice``. Parts are drawn from a shared catalog so the same
part is reused across components, giving the composition graphs real structure.

Used to develop and test the structure-aware and graph encoders without a
populated sbol-db. ``write_sbol_turtle`` serializes to SBOL3 RDF for file-based
corpus and ingestion round-trips.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Iterator

from synbiotorch.types import Alphabet, Design, Feature, GraphEdge, GraphNode, GraphSlice, Location, Sequence

SBOL3 = "http://sbols.org/v3#"
NS = "https://synbiotorch.test/"

# Sequence Ontology roles.
ROLE = {
    "promoter": "https://identifiers.org/SO:0000167",
    "rbs": "https://identifiers.org/SO:0000139",
    "cds": "https://identifiers.org/SO:0000316",
    "terminator": "https://identifiers.org/SO:0000141",
}
ENGINEERED_REGION = "https://identifiers.org/SO:0000804"
ORIENTATION_INLINE = f"{SBOL3}inline"
ORIENTATION_RC = f"{SBOL3}reverseComplement"

# Part catalog: role -> list of part display ids. Lengths are role-typical.
_CATALOG = {
    "promoter": (["J23100", "J23101", "J23106", "J23118"], 35),
    "rbs": (["B0034", "B0032", "B0030"], 20),
    "cds": (["GFP", "RFP", "BFP", "YFP"], 90),
    "terminator": (["B0015", "L3S2P21"], 40),
}
_ROLE_ORDER = ["promoter", "rbs", "cds", "terminator"]
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


class PartCatalog:
    """A fixed set of named parts with stable sequences (seeded once)."""

    def __init__(self, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.sequences: dict[str, str] = {}
        self.roles: dict[str, str] = {}
        for role, (names, length) in _CATALOG.items():
            for name in names:
                self.sequences[name] = "".join(rng.choice("ACGT") for _ in range(length))
                self.roles[name] = role

    def part_iri(self, name: str) -> str:
        return f"{NS}part/{name}"


def generate_components(n: int, *, seed: int = 0, with_labels: bool = True) -> list[Design]:
    """Generate ``n`` synthetic composite Components."""
    catalog = PartCatalog(seed)
    rng = random.Random(seed + 1)
    # A per-promoter "strength" gives a learnable supervised signal.
    strengths = {name: rng.uniform(1.0, 10.0) for name in _CATALOG["promoter"][0]}

    components: list[Design] = []
    for i in range(n):
        chosen = {role: rng.choice(names) for role, (names, _) in _CATALOG.items()}
        comp_iri = f"{NS}component/tu{i}"

        elements_parts: list[str] = []
        features: list[Feature] = []
        nodes: list[GraphNode] = [
            GraphNode(iri=comp_iri, depth=0, record_class=f"{SBOL3}Component", display_id=f"tu{i}")
        ]
        edges: list[GraphEdge] = []
        cursor = 0
        for role in _ROLE_ORDER:
            part = chosen[role]
            part_seq = catalog.sequences[part]
            reverse = rng.random() < 0.2
            placed = _reverse_complement(part_seq) if reverse else part_seq
            start = cursor + 1  # SBOL Ranges are 1-based, inclusive.
            end = cursor + len(part_seq)
            cursor = end
            elements_parts.append(placed)

            feature_iri = f"{comp_iri}/{role}"
            part_iri = catalog.part_iri(part)
            features.append(
                Feature(
                    iri=feature_iri,
                    kind="SubComponent",
                    instance_of=part_iri,
                    roles=(ROLE[role],),
                    locations=(
                        Location(
                            start=start,
                            end=end,
                            orientation=ORIENTATION_RC if reverse else ORIENTATION_INLINE,
                        ),
                    ),
                )
            )
            nodes.append(GraphNode(iri=feature_iri, depth=1, record_class=f"{SBOL3}SubComponent", display_id=role))
            nodes.append(GraphNode(iri=part_iri, depth=2, record_class=f"{SBOL3}Component", display_id=part))
            edges.append(GraphEdge(subject=comp_iri, predicate=f"{SBOL3}hasFeature", object=feature_iri, depth=1))
            edges.append(GraphEdge(subject=feature_iri, predicate=f"{SBOL3}instanceOf", object=part_iri, depth=2))

        sequence = "".join(elements_parts)
        seq_iri = f"{comp_iri}/sequence"
        nodes.append(GraphNode(iri=seq_iri, depth=1, record_class=f"{SBOL3}Sequence", display_id="sequence"))
        edges.append(GraphEdge(subject=comp_iri, predicate=f"{SBOL3}hasSequence", object=seq_iri, depth=1))

        label = strengths[chosen["promoter"]] if with_labels else None
        components.append(
            Design(
                iri=comp_iri,
                record_class=f"{SBOL3}Component",
                display_id=f"tu{i}",
                roles=(ENGINEERED_REGION,),
                types=(f"{SBOL3}DNA",),
                sequence=Sequence(elements=sequence, alphabet=Alphabet.DNA),
                features=tuple(features),
                neighbors=GraphSlice(root_iri=comp_iri, nodes=tuple(nodes), edges=tuple(edges), truncated=False),
                label=label,
            )
        )
    return components


class SyntheticCorpus:
    """An in-memory Corpus of synthetic components, for tests and local development."""

    def __init__(self, n: int = 64, *, seed: int = 0, with_labels: bool = True) -> None:
        self.n = n
        self.seed = seed
        self.with_labels = with_labels

    def __iter__(self) -> Iterator[Design]:
        return iter(generate_components(self.n, seed=self.seed, with_labels=self.with_labels))

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(f"synthetic:{self.n}:{self.seed}:{self.with_labels}".encode())
        return h.hexdigest()[:16]


_RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
_XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"


def _iri(value: str) -> str:
    return f"<{value}>"


def _literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _integer(value: int) -> str:
    return f'"{value}"^^<{_XSD_INTEGER}>'


def write_sbol_turtle(components: list[Design], path: str | Path) -> Path:
    """Serialize synthetic components to an SBOL3 Turtle document.

    Emits one triple per line (the N-Triples subset of Turtle), so it round-trips
    through the native SBOL reader without an RDF library dependency.
    """
    lines: list[str] = []

    def emit(subject: str, predicate: str, obj: str) -> None:
        lines.append(f"{subject} {predicate} {obj} .")

    for comp in components:
        comp_ref = _iri(comp.iri)
        emit(comp_ref, _RDF_TYPE, _iri(f"{SBOL3}Component"))
        for role in comp.roles:
            emit(comp_ref, _iri(f"{SBOL3}role"), _iri(role))
        if comp.sequence is not None:
            seq_ref = _iri(f"{comp.iri}/sequence")
            emit(comp_ref, _iri(f"{SBOL3}hasSequence"), seq_ref)
            emit(seq_ref, _RDF_TYPE, _iri(f"{SBOL3}Sequence"))
            emit(seq_ref, _iri(f"{SBOL3}elements"), _literal(comp.sequence.elements))
        for feature in comp.features:
            feat_ref = _iri(feature.iri)
            emit(comp_ref, _iri(f"{SBOL3}hasFeature"), feat_ref)
            emit(feat_ref, _RDF_TYPE, _iri(f"{SBOL3}SubComponent"))
            if feature.instance_of:
                emit(feat_ref, _iri(f"{SBOL3}instanceOf"), _iri(feature.instance_of))
            for role in feature.roles:
                emit(feat_ref, _iri(f"{SBOL3}role"), _iri(role))
            for index, loc in enumerate(feature.locations):
                loc_ref = _iri(f"{feature.iri}/loc{index}")
                emit(feat_ref, _iri(f"{SBOL3}hasLocation"), loc_ref)
                emit(loc_ref, _RDF_TYPE, _iri(f"{SBOL3}Range"))
                if loc.start is not None:
                    emit(loc_ref, _iri(f"{SBOL3}start"), _integer(loc.start))
                if loc.end is not None:
                    emit(loc_ref, _iri(f"{SBOL3}end"), _integer(loc.end))
                if loc.orientation:
                    emit(loc_ref, _iri(f"{SBOL3}orientation"), _iri(loc.orientation))

    out = Path(path)
    out.write_text("\n".join(lines) + "\n")
    return out
