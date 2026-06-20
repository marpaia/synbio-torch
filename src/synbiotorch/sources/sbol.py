"""SBOL ingestion, parsed in-process by the native sbol-rs binding.

SBOL 3 documents are read directly, SBOL 2 documents are upgraded to SBOL 3, and
both are flattened to ``Design`` records — sequence, annotated features, and the
composition ``GraphSlice`` — so real designs feed the ``structure_aware`` and
``graph`` modalities. ``design_from_record`` covers the separate sbol-db REST
payload shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from synbiotorch import _sbol
from synbiotorch.config import CorpusConfig
from synbiotorch.exceptions import ParseError
from synbiotorch.types import (
    Alphabet,
    Design,
    Feature,
    GraphEdge,
    GraphNode,
    GraphSlice,
    Location,
    Sequence,
    local_name,
)

from .files import fingerprint_files, list_files

SBOL3 = "http://sbols.org/v3#"
_SBOL3_EXTENSIONS = {".ttl", ".rdf", ".nt", ".xml", ".jsonld", ".sbol"}
# Extension -> the RDF format name the binding expects.
_RDF_FORMAT = {".ttl": "ttl", ".rdf": "rdf", ".xml": "rdf", ".nt": "nt", ".jsonld": "jsonld", ".sbol": "ttl"}


def records_to_designs(records: list[dict[str, Any]], label_key: str | None = None) -> Iterator[Design]:
    """Map binding records (parsed JSON) into ``Design`` instances."""
    for record in records:
        yield _record_to_design(record, label_key)


def _record_to_design(record: dict[str, Any], label_key: str | None) -> Design:
    seq = record.get("sequence")
    sequence = None
    if seq:
        encoding = seq.get("encoding")
        sequence = Sequence(
            elements=seq["elements"],
            alphabet=Alphabet.from_encoding(encoding),
            encoding_iri=encoding,
        )
    features = tuple(_feature(f) for f in record.get("features") or ())
    label = _label_from_extensions(record.get("extensions") or (), label_key) if label_key else None
    neighbors = _graph_from_record(record, features) if features or sequence else None
    return Design(
        iri=record["iri"],
        record_class=record.get("record_class", ""),
        display_id=record.get("display_id"),
        name=record.get("name"),
        roles=tuple(record.get("roles") or ()),
        types=tuple(record.get("types") or ()),
        sequence=sequence,
        features=features,
        neighbors=neighbors,
        label=label,
        raw=record,
    )


def _feature(data: dict[str, Any]) -> Feature:
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


def _graph_from_record(record: dict[str, Any], features: tuple[Feature, ...]) -> GraphSlice:
    """Build the composition GraphSlice rooted at a Component record."""
    comp_iri = record["iri"]
    nodes: list[GraphNode] = [
        GraphNode(iri=comp_iri, depth=0, record_class=f"{SBOL3}Component", display_id=record.get("display_id"))
    ]
    edges: list[GraphEdge] = []
    for feature in features:
        nodes.append(
            GraphNode(
                iri=feature.iri,
                depth=1,
                record_class=f"{SBOL3}{feature.kind}" if feature.kind else None,
                display_id=local_name(feature.iri),
            )
        )
        edges.append(GraphEdge(subject=comp_iri, predicate=f"{SBOL3}hasFeature", object=feature.iri, depth=1))
        if feature.instance_of:
            nodes.append(
                GraphNode(
                    iri=feature.instance_of,
                    depth=2,
                    record_class=f"{SBOL3}Component",
                    display_id=local_name(feature.instance_of),
                )
            )
            edges.append(
                GraphEdge(subject=feature.iri, predicate=f"{SBOL3}instanceOf", object=feature.instance_of, depth=2)
            )
    if record.get("sequence"):
        seq_iri = f"{comp_iri}/sequence"
        nodes.append(GraphNode(iri=seq_iri, depth=1, record_class=f"{SBOL3}Sequence", display_id="sequence"))
        edges.append(GraphEdge(subject=comp_iri, predicate=f"{SBOL3}hasSequence", object=seq_iri, depth=1))
    return GraphSlice(root_iri=comp_iri, nodes=tuple(nodes), edges=tuple(edges), truncated=False)


def _label_from_extensions(extensions: Any, label_key: str) -> float | int | None:
    """Read a numeric label from an annotation triple whose predicate matches by local-name."""
    for ext in extensions:
        if local_name(ext["predicate"]) != label_key:
            continue
        try:
            num = float(ext["value"])
        except (TypeError, ValueError):
            return None
        return int(num) if num.is_integer() else num
    return None


class SbolFileCorpus:
    """Reads ``Design`` records from SBOL RDF files (SBOL 2 or 3) via the binding."""

    def __init__(self, path: str | Path, *, namespace: str | None = None, label_key: str | None = None) -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.label_key = label_key

    @classmethod
    def from_config(cls, config: CorpusConfig) -> "SbolFileCorpus":
        assert config.path is not None  # guaranteed by CorpusConfig validation
        return cls(config.path, namespace=config.namespace, label_key=config.label_key)

    def _files(self) -> list[Path]:
        return [p for p in list_files(self.path) if p.suffix.lower() in _SBOL3_EXTENSIONS]

    def __iter__(self) -> Iterator[Design]:
        for file in self._files():
            text = file.read_text()
            fmt = _RDF_FORMAT.get(file.suffix.lower(), "ttl")
            try:
                if f"{SBOL3}" in text or "sbols.org/v3#" in text:
                    raw = _sbol.read_sbol3(text, fmt)
                else:
                    raw = _sbol.upgrade_sbol2(text, fmt, self.namespace)
            except ValueError as exc:
                raise ParseError(f"failed to parse SBOL file {file}: {exc}") from exc
            yield from records_to_designs(json.loads(raw), self.label_key)

    def fingerprint(self) -> str:
        return fingerprint_files(self._files(), self.namespace, self.label_key)


def design_from_record(record: dict[str, Any], label: float | int | None = None) -> Design:
    """Build a ``Design`` from an sbol-db ``SbolObjectRecord`` REST payload.

    The sequence elements live inside the lossless JSON-LD ``data`` slice under
    the ``sbol:elements`` predicate; they are extracted by local name so the code
    is robust to IRI compaction.
    """
    data = record.get("data") or {}
    return Design(
        iri=record["iri"],
        record_class=record.get("sbol_class", ""),
        display_id=record.get("display_id"),
        name=record.get("name"),
        roles=tuple(record.get("roles") or ()),
        types=tuple(record.get("types") or ()),
        sequence=_sequence_from_data(data),
        label=label,
        raw=record,
    )


def _scalar(value: Any) -> Any:
    """Unwrap JSON-LD value shapes: {"@value": x}, [x], or x -> x."""
    if isinstance(value, list):
        return _scalar(value[0]) if value else None
    if isinstance(value, dict):
        return value.get("@value", value.get("value"))
    return value


def _sequence_from_data(data: dict[str, Any]) -> Sequence | None:
    """Extract elements + encoding from a JSON-LD object slice, by local name."""
    elements: str | None = None
    encoding: str | None = None
    for key, value in data.items():
        name = local_name(key)
        if name == "elements" and elements is None:
            elements = _scalar(value)
        elif name == "encoding" and encoding is None:
            enc = _scalar(value)
            encoding = enc if isinstance(enc, str) else None
    if not elements:
        return None
    return Sequence(elements=str(elements), alphabet=Alphabet.from_encoding(encoding), encoding_iri=encoding)
