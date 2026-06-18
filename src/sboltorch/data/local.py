"""Local-file corpus: FASTA and SBOL RDF, normalized to SbolObject.

This is the offline fallback to the sbol-db client. It produces the exact same
SbolObject records, so downstream code is identical regardless of source.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

from rdflib import RDF, Graph, URIRef
from rdflib.term import Node

from sboltorch.config import CorpusConfig
from sboltorch.exceptions import ParseError
from sboltorch.types import (
    Alphabet,
    Feature,
    GraphEdge,
    GraphNode,
    GraphSlice,
    Location,
    SbolObject,
    SbolSequence,
    local_name,
)

_SBOL3 = "http://sbols.org/v3#"
_SBOL_ELEMENTS = "http://sbols.org/v2#elements"
_SBOL_ELEMENTS_V3 = f"{_SBOL3}elements"
_SBOL_COMPONENT = URIRef(f"{_SBOL3}Component")


class LocalFileCorpus:
    """Reads SbolObject records from a FASTA or SBOL RDF file (or a directory of them)."""

    def __init__(self, path: str | Path, *, fmt: str = "auto", label_key: str | None = None) -> None:
        self.path = Path(path)
        self.fmt = fmt
        self.label_key = label_key

    @classmethod
    def from_config(cls, config: CorpusConfig) -> "LocalFileCorpus":
        assert config.path is not None  # guaranteed by CorpusConfig validation
        return cls(config.path, fmt=config.fmt, label_key=config.label_key)

    def _files(self) -> list[Path]:
        if self.path.is_dir():
            return sorted(p for p in self.path.rglob("*") if p.is_file())
        return [self.path]

    def _format_for(self, file: Path) -> str:
        if self.fmt != "auto":
            return self.fmt
        suffix = file.suffix.lower()
        if suffix in {".fa", ".fasta", ".fna"}:
            return "fasta"
        if suffix in {".xml", ".rdf", ".ttl", ".nt", ".sbol"}:
            return "sbol"
        raise ParseError(f"cannot infer format for {file}; set corpus.fmt explicitly")

    def __iter__(self) -> Iterator[SbolObject]:
        for file in self._files():
            fmt = self._format_for(file)
            if fmt == "fasta":
                yield from _parse_fasta(file, self.label_key)
            else:
                yield from _parse_sbol(file, self.label_key)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for file in self._files():
            stat = file.stat()
            h.update(str(file).encode())
            h.update(str(stat.st_size).encode())
            h.update(str(int(stat.st_mtime)).encode())
        h.update(repr(self.label_key).encode())
        return h.hexdigest()[:16]


def _parse_fasta(file: Path, label_key: str | None) -> Iterator[SbolObject]:
    """Parse FASTA. Labels are read from ``key=value`` tokens in the header."""
    header: str | None = None
    chunks: list[str] = []

    def flush() -> SbolObject | None:
        if header is None:
            return None
        seq_id = header.split()[0] if header.split() else header
        label = _label_from_header(header, label_key) if label_key else None
        return SbolObject(
            iri=seq_id,
            sbol_class="http://sbols.org/v3#Sequence",
            display_id=seq_id,
            sequence=SbolSequence(elements="".join(chunks), alphabet=Alphabet.DNA),
            label=label,
            raw={"header": header},
        )

    with file.open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                obj = flush()
                if obj is not None:
                    yield obj
                header = line[1:].strip()
                chunks = []
            elif line:
                chunks.append(line.strip())
    obj = flush()
    if obj is not None:
        yield obj


def _label_from_header(header: str, label_key: str) -> float | int | None:
    for token in header.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key == label_key:
                try:
                    num = float(value)
                except ValueError:
                    return None
                return int(num) if num.is_integer() else num
    return None


def _parse_sbol(file: Path, label_key: str | None) -> Iterator[SbolObject]:
    """Parse an SBOL RDF document.

    SBOL3 documents carry their structure on ``Component`` top-levels — the
    sequence (``hasSequence``), annotated features (``hasFeature`` →
    SubComponent / SequenceFeature, with roles and ``Range`` locations), and the
    composition graph. When the document has Components, one ``SbolObject`` is
    yielded per Component, populating ``features`` and ``neighbors`` so the
    structure-aware and graph modalities work on real designs.

    Documents without Components — bare ``sbol:Sequence`` subjects, including
    SBOL2 ``ComponentDefinition`` sequences — fall back to one sequence-only
    record per ``elements`` subject.
    """
    graph = Graph()
    try:
        graph.parse(str(file))
    except Exception as exc:  # rdflib raises a variety of parser errors
        raise ParseError(f"failed to parse SBOL file {file}: {exc}") from exc

    components = list(graph.subjects(RDF.type, _SBOL_COMPONENT))
    if components:
        yield from _parse_components(graph, sorted(components, key=str), file, label_key)
    else:
        yield from _parse_sequences(graph, file, label_key)


def _parse_sequences(graph: Graph, file: Path, label_key: str | None) -> Iterator[SbolObject]:
    """Yield one sequence-only SbolObject per ``elements`` subject (SBOL3 or SBOL2)."""
    for predicate in (URIRef(_SBOL_ELEMENTS_V3), URIRef(_SBOL_ELEMENTS)):
        for subject, _, elements in graph.triples((None, predicate, None)):
            iri = str(subject)
            label = _label_from_graph(graph, subject, label_key) if label_key else None
            yield SbolObject(
                iri=iri,
                sbol_class="http://sbols.org/v3#Sequence",
                display_id=local_name(iri),
                sequence=SbolSequence(elements=str(elements), alphabet=Alphabet.DNA),
                label=label,
                raw={"file": str(file)},
            )


def _parse_components(graph: Graph, components: list[Node], file: Path, label_key: str | None) -> Iterator[SbolObject]:
    """Yield one rich SbolObject per SBOL3 Component."""
    for comp in components:
        comp_iri = str(comp)
        sequence = _component_sequence(graph, comp)
        features = _component_features(graph, comp)
        neighbors = _component_graph(graph, comp, comp_iri, features, sequence is not None)
        label = _label_from_graph(graph, comp, label_key) if label_key else None
        yield SbolObject(
            iri=comp_iri,
            sbol_class=f"{_SBOL3}Component",
            display_id=_first_local(graph, comp, "displayId") or local_name(comp_iri),
            name=_first_value(graph, comp, "name"),
            roles=_values(graph, comp, "role"),
            types=_sbol_types(graph, comp),
            sequence=sequence,
            features=tuple(features),
            neighbors=neighbors,
            label=label,
            raw={"file": str(file)},
        )


def _component_sequence(graph: Graph, comp: Node) -> SbolSequence | None:
    """Resolve a Component's first ``hasSequence`` target that carries elements."""
    for seq in _objects(graph, comp, "hasSequence"):
        elements = _first_value(graph, seq, "elements")
        if not elements:
            continue
        encoding = _first_object_str(graph, seq, "encoding")
        return SbolSequence(elements=elements, alphabet=Alphabet.from_encoding(encoding), encoding_iri=encoding)
    return None


def _component_features(graph: Graph, comp: Node) -> list[Feature]:
    """Build a Feature per ``hasFeature`` target (SubComponent or SequenceFeature)."""
    features: list[Feature] = []
    for feat in _objects(graph, comp, "hasFeature"):
        locations: list[Location] = []
        for loc in _objects(graph, feat, "hasLocation"):
            locations.append(
                Location(
                    start=_first_int(graph, loc, "start"),
                    end=_first_int(graph, loc, "end"),
                    orientation=_first_object_str(graph, loc, "orientation"),
                )
            )
        features.append(
            Feature(
                iri=str(feat),
                kind=_local_type(graph, feat),
                instance_of=_first_object_str(graph, feat, "instanceOf"),
                roles=_values(graph, feat, "role"),
                locations=tuple(locations),
            )
        )
    return features


def _component_graph(
    graph: Graph, comp: Node, comp_iri: str, features: list[Feature], has_sequence: bool
) -> GraphSlice:
    """Build the composition GraphSlice rooted at a Component."""
    nodes: list[GraphNode] = [
        GraphNode(
            iri=comp_iri,
            depth=0,
            sbol_class=f"{_SBOL3}Component",
            display_id=_first_local(graph, comp, "displayId") or local_name(comp_iri),
        )
    ]
    edges: list[GraphEdge] = []
    for feature in features:
        nodes.append(
            GraphNode(
                iri=feature.iri,
                depth=1,
                sbol_class=f"{_SBOL3}{feature.kind}" if feature.kind else None,
                display_id=local_name(feature.iri),
            )
        )
        edges.append(GraphEdge(subject=comp_iri, predicate=f"{_SBOL3}hasFeature", object=feature.iri, depth=1))
        if feature.instance_of:
            nodes.append(
                GraphNode(
                    iri=feature.instance_of,
                    depth=2,
                    sbol_class=f"{_SBOL3}Component",
                    display_id=local_name(feature.instance_of),
                )
            )
            edges.append(
                GraphEdge(subject=feature.iri, predicate=f"{_SBOL3}instanceOf", object=feature.instance_of, depth=2)
            )
    if has_sequence:
        seq_iri = f"{comp_iri}/sequence"
        nodes.append(GraphNode(iri=seq_iri, depth=1, sbol_class=f"{_SBOL3}Sequence", display_id="sequence"))
        edges.append(GraphEdge(subject=comp_iri, predicate=f"{_SBOL3}hasSequence", object=seq_iri, depth=1))
    return GraphSlice(root_iri=comp_iri, nodes=tuple(nodes), edges=tuple(edges), truncated=False)


def _objects(graph: Graph, subject: Node, predicate_local: str) -> list[Node]:
    """All objects of triples on ``subject`` whose predicate has the given local name."""
    return [o for _, p, o in graph.triples((subject, None, None)) if local_name(str(p)) == predicate_local]


def _values(graph: Graph, subject: Node, predicate_local: str) -> tuple[str, ...]:
    return tuple(str(o) for o in _objects(graph, subject, predicate_local))


def _sbol_types(graph: Graph, subject: Node) -> tuple[str, ...]:
    """SBOL ``type`` objects only — ``rdf:type`` shares the local name ``type`` and is excluded."""
    return tuple(
        str(o) for _, p, o in graph.triples((subject, None, None)) if p != RDF.type and local_name(str(p)) == "type"
    )


def _first_value(graph: Graph, subject: Node, predicate_local: str) -> str | None:
    values = _values(graph, subject, predicate_local)
    return values[0] if values else None


def _first_object_str(graph: Graph, subject: Node, predicate_local: str) -> str | None:
    """First object that is an IRI (URIRef), as a string — for role/encoding/orientation refs."""
    for obj in _objects(graph, subject, predicate_local):
        if isinstance(obj, URIRef):
            return str(obj)
    return None


def _first_local(graph: Graph, subject: Node, predicate_local: str) -> str | None:
    value = _first_value(graph, subject, predicate_local)
    return value or None


def _first_int(graph: Graph, subject: Node, predicate_local: str) -> int | None:
    value = _first_value(graph, subject, predicate_local)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _local_type(graph: Graph, subject: Node) -> str | None:
    """Local name of an SBOL3 rdf:type (e.g. ``SubComponent``, ``SequenceFeature``)."""
    for obj in graph.objects(subject, RDF.type):
        name = local_name(str(obj))
        if name:
            return name
    return None


def _label_from_graph(graph: Graph, subject: Node, label_key: str) -> float | int | None:
    for _, predicate, value in graph.triples((subject, None, None)):
        if local_name(str(predicate)) != label_key:
            continue
        try:
            num = float(str(value))
        except ValueError:
            return None
        return int(num) if num.is_integer() else num
    return None
