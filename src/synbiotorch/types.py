"""Source-agnostic record types.

Every data source normalizes into ``Design`` instances. Training code consumes
only these types and never branches on where the data came from. A ``Design`` is
one biological design or sequence record: a sequence, optional annotated
features, and an optional composition graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Matches the SBOL elements/encoding predicates regardless of compaction, e.g.
# "http://sbols.org/v3#elements", "sbol:elements", or a bare "elements" key.
_LOCAL_NAME = re.compile(r"[#/:]")


def local_name(iri: str) -> str:
    """Return the local name of an IRI/CURIE (the part after the last #, / or :)."""
    return _LOCAL_NAME.split(iri.rstrip("#/"))[-1]


class Alphabet(str, Enum):
    """Sequence alphabet (DNA, RNA, protein, or other)."""

    DNA = "DNA"
    RNA = "RNA"
    PROTEIN = "PROTEIN"
    OTHER = "OTHER"

    @classmethod
    def from_encoding(cls, encoding_iri: str | None) -> "Alphabet":
        """Infer an alphabet from an SBOL encoding IRI; default to DNA."""
        if not encoding_iri:
            return cls.DNA
        name = local_name(encoding_iri).lower()
        if "protein" in name or "aminoacid" in name:
            return cls.PROTEIN
        if "rna" in name:
            return cls.RNA
        if "dna" in name:
            return cls.DNA
        return cls.OTHER


@dataclass(frozen=True)
class Sequence:
    """A biological sequence: the raw elements plus its alphabet."""

    elements: str
    alphabet: Alphabet = Alphabet.DNA
    encoding_iri: str | None = None

    def __len__(self) -> int:
        return len(self.elements)


@dataclass(frozen=True)
class Location:
    """A position within a sequence (Range or Cut), with optional orientation."""

    start: int | None = None
    end: int | None = None
    orientation: str | None = None


@dataclass(frozen=True)
class Feature:
    """A feature within a component (typically a SubComponent)."""

    iri: str
    kind: str | None = None
    instance_of: str | None = None
    roles: tuple[str, ...] = ()
    locations: tuple[Location, ...] = ()


@dataclass(frozen=True)
class GraphNode:
    iri: str
    depth: int
    record_class: str | None = None
    display_id: str | None = None


@dataclass(frozen=True)
class GraphEdge:
    subject: str
    predicate: str
    object: str
    depth: int


@dataclass(frozen=True)
class GraphSlice:
    """A bounded neighborhood of an object, used by the structure/graph encoders."""

    root_iri: str
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    truncated: bool = False


@dataclass(frozen=True)
class Design:
    """The canonical unit of data flowing through the library."""

    iri: str
    record_class: str
    display_id: str | None = None
    name: str | None = None
    roles: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    sequence: Sequence | None = None
    features: tuple[Feature, ...] = ()
    neighbors: GraphSlice | None = None
    label: float | int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def feature_to_dict(feature: Feature) -> dict[str, Any]:
    return {
        "iri": feature.iri,
        "kind": feature.kind,
        "instance_of": feature.instance_of,
        "roles": list(feature.roles),
        "locations": [
            {"start": loc.start, "end": loc.end, "orientation": loc.orientation} for loc in feature.locations
        ],
    }


def feature_from_dict(data: dict[str, Any]) -> Feature:
    return Feature(
        iri=data["iri"],
        kind=data.get("kind"),
        instance_of=data.get("instance_of"),
        roles=tuple(data.get("roles") or ()),
        locations=tuple(
            Location(start=loc.get("start"), end=loc.get("end"), orientation=loc.get("orientation"))
            for loc in data.get("locations") or ()
        ),
    )


def graph_to_dict(graph: GraphSlice) -> dict[str, Any]:
    return {
        "root_iri": graph.root_iri,
        "truncated": graph.truncated,
        "nodes": [
            {"iri": n.iri, "depth": n.depth, "record_class": n.record_class, "display_id": n.display_id}
            for n in graph.nodes
        ],
        "edges": [
            {"subject": e.subject, "predicate": e.predicate, "object": e.object, "depth": e.depth} for e in graph.edges
        ],
    }


def graph_from_dict(data: dict[str, Any]) -> GraphSlice:
    return GraphSlice(
        root_iri=data["root_iri"],
        truncated=bool(data.get("truncated", False)),
        nodes=tuple(
            GraphNode(
                iri=n["iri"], depth=n["depth"], record_class=n.get("record_class"), display_id=n.get("display_id")
            )
            for n in data.get("nodes") or ()
        ),
        edges=tuple(
            GraphEdge(subject=e["subject"], predicate=e["predicate"], object=e["object"], depth=e["depth"])
            for e in data.get("edges") or ()
        ),
    )
