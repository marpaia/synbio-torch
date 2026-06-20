"""Graph encoder: SBOL composition graph -> PyG Data for a graph transformer.

Each object's neighborhood becomes a graph whose nodes carry a (record_class, role)
type pair and whose edges carry a predicate type (hasFeature / instanceOf /
hasSequence). Edges are added in both directions so information flows up and down
the composition hierarchy. The companion model embeds these categorical types.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from synbiotorch.encoders.structure import DEFAULT_ROLES
from synbiotorch.types import Design, local_name

_NODE_CLASSES = ("Component", "SubComponent", "Sequence")
_EDGE_PREDICATES = ("hasFeature", "instanceOf", "hasSequence")


@dataclass(frozen=True)
class GraphSpec:
    """Vocabulary sizes the graph model needs to size its embeddings."""

    num_node_classes: int
    num_roles: int
    num_edge_types: int
    num_name_buckets: int


def _indexed(values: tuple[str, ...]) -> dict[str, int]:
    # Index 0 is reserved for an out-of-vocabulary "other".
    return {v: i + 1 for i, v in enumerate(values)}


class GraphEncoder:
    def __init__(self, roles: tuple[str, ...] = DEFAULT_ROLES, *, name_buckets: int = 4096) -> None:
        self._class_ids = _indexed(_NODE_CLASSES)
        self._edge_ids = _indexed(_EDGE_PREDICATES)
        # Role 0 = "none/other" (e.g. Component and Sequence nodes carry no role).
        self._role_ids = {local_name(r): i + 1 for i, r in enumerate(roles)}
        self._name_buckets = name_buckets

    def _class_id(self, record_class: str | None) -> int:
        return self._class_ids.get(local_name(record_class or ""), 0)

    def _role_id(self, role: str | None) -> int:
        return self._role_ids.get(local_name(role or ""), 0) if role else 0

    def _edge_id(self, predicate: str) -> int:
        return self._edge_ids.get(local_name(predicate), 0)

    def _name_id(self, display_id: str | None) -> int:
        """Stable hash of a node's display id into a bounded bucket (0 = none).

        This carries part identity (e.g. which promoter) into node features, so
        the graph can learn properties that depend on which parts are present —
        without an unbounded, train-only vocabulary. Uses md5 (not Python's
        salted ``hash``) so the mapping is deterministic across processes.
        """
        if not display_id:
            return 0
        digest = hashlib.md5(display_id.encode()).hexdigest()
        return int(digest, 16) % (self._name_buckets - 1) + 1

    def encode(self, obj: Design) -> Data:
        graph = obj.neighbors
        if graph is None or not graph.nodes:
            raise ValueError(f"object {obj.iri} has no composition graph to encode")

        index = {node.iri: i for i, node in enumerate(graph.nodes)}
        role_of = {f.iri: (f.roles[0] if f.roles else None) for f in obj.features}

        x = torch.tensor(
            [
                [self._class_id(n.record_class), self._role_id(role_of.get(n.iri)), self._name_id(n.display_id)]
                for n in graph.nodes
            ],
            dtype=torch.long,
        )

        src: list[int] = []
        dst: list[int] = []
        etype: list[int] = []
        for edge in graph.edges:
            if edge.subject in index and edge.object in index:
                a, b = index[edge.subject], index[edge.object]
                kind = self._edge_id(edge.predicate)
                # Bidirectional so messages flow both ways along composition edges.
                src += [a, b]
                dst += [b, a]
                etype += [kind, kind]

        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        data.edge_type = torch.tensor(etype, dtype=torch.long) if etype else torch.zeros((0,), dtype=torch.long)
        if obj.label is not None:
            data.y = torch.tensor([float(obj.label)], dtype=torch.float)
        return data

    @property
    def spec(self) -> GraphSpec:
        return GraphSpec(
            num_node_classes=len(self._class_ids) + 1,
            num_roles=len(self._role_ids) + 1,
            num_edge_types=len(self._edge_ids) + 1,
            num_name_buckets=self._name_buckets,
        )
