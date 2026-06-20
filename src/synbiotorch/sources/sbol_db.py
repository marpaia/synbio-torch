"""Typed Python client for the sbol-db REST API.

Implements the read paths a training pipeline needs: keyset-paginated object
listing, single/bulk IRI resolution, bounded neighborhood traversal, sequence
search, and ontology descendant expansion. Endpoints and payload shapes mirror
the service's OpenAPI specification.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterator

import httpx

from synbiotorch.config import CorpusConfig
from synbiotorch.exceptions import SbolDbError
from synbiotorch.sources.sbol import design_from_record
from synbiotorch.types import Design, GraphEdge, GraphNode, GraphSlice, local_name

# sbol-db caps a listing page at 5000 objects.
_MAX_PAGE = 5000


class SbolDbClient:
    """A thin, typed wrapper over the sbol-db HTTP API.

    When iterated it streams every object matching the configured
    ``record_class`` / ``role`` / ``document_id`` filters, resolving the
    supervised label from each object's JSON-LD slice when ``label_key`` is set.
    """

    def __init__(
        self,
        base_url: str,
        *,
        auth: tuple[str, str] | None = None,
        record_class: str | None = None,
        role: str | None = None,
        document_id: str | None = None,
        label_key: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.record_class = record_class
        self.role = role
        self.document_id = document_id
        self.label_key = label_key
        self._client = client or httpx.Client(base_url=self.base_url, auth=auth, timeout=timeout)

    @classmethod
    def from_config(cls, config: CorpusConfig) -> "SbolDbClient":
        assert config.base_url is not None  # guaranteed by CorpusConfig validation
        auth = (config.username, config.password) if config.username and config.password else None
        return cls(
            config.base_url,
            auth=auth,
            record_class=config.record_class,
            role=config.role,
            document_id=config.document_id,
            label_key=config.label_key,
        )

    # -- low-level ---------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            resp = self._client.get(path, params=_clean(params))
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise SbolDbError(f"GET {path} -> {exc.response.status_code}: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise SbolDbError(f"GET {path} failed: {exc}") from exc

    def _post(self, path: str, json: dict[str, Any]) -> Any:
        try:
            resp = self._client.post(path, json=json)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise SbolDbError(f"POST {path} -> {exc.response.status_code}: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise SbolDbError(f"POST {path} failed: {exc}") from exc

    # -- object listing ----------------------------------------------------

    def list_objects(
        self,
        *,
        record_class: str | None = None,
        role: str | None = None,
        document_id: str | None = None,
        limit: int = _MAX_PAGE,
    ) -> Iterator[dict[str, Any]]:
        """Yield raw SbolObjectRecord dicts, transparently following the keyset cursor."""
        after: str | None = None
        while True:
            page = self._get(
                "/objects/list",
                {
                    "sbol_class": record_class,
                    "role": role,
                    "document_id": document_id,
                    "limit": min(limit, _MAX_PAGE),
                    "after": after,
                },
            )
            objects = page.get("objects") or []
            for record in objects:
                yield record
            after = page.get("next_cursor")
            if not after or not objects:
                return

    def get_object(self, iri: str) -> dict[str, Any]:
        return self._get("/objects", {"iri": iri})

    def lookup_objects(self, iris: list[str]) -> dict[str, Any]:
        """Resolve up to 1000 IRIs in one request."""
        return self._post("/objects/lookup", {"iris": iris})

    def neighborhood(
        self,
        iri: str,
        *,
        depth: int = 1,
        direction: str = "forward",
        predicates: list[str] | None = None,
    ) -> GraphSlice:
        params: dict[str, Any] = {"iri": iri, "depth": depth, "direction": direction}
        if predicates:
            params["predicates"] = ",".join(predicates)
        result = self._get("/objects/neighborhood", params)
        return _graph_slice(result)

    def search_sequence(self, pattern: str, *, max_hits: int = 100, forward_only: bool = False) -> list[dict[str, Any]]:
        return self._get(
            "/sequences/search",
            {"pattern": pattern, "max_hits": max_hits, "forward_only": forward_only},
        )

    def ontology_descendants(self, iri: str) -> list[Any]:
        return self._get("/ontology/descendants", {"iri": iri})

    # -- Corpus protocol ---------------------------------------------------

    def __iter__(self) -> Iterator[Design]:
        for record in self.list_objects(
            record_class=self.record_class,
            role=self.role,
            document_id=self.document_id,
        ):
            label = _extract_label(record, self.label_key) if self.label_key else None
            yield design_from_record(record, label=label)

    def fingerprint(self) -> str:
        """Identify this corpus by its source + filters (not its full contents).

        The materialization layer hashes the actual records; this is the cheap
        identity used to name the cache namespace.
        """
        h = hashlib.sha256()
        for part in (self.base_url, self.record_class, self.role, self.document_id, self.label_key):
            h.update(repr(part).encode())
        return h.hexdigest()[:16]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SbolDbClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _clean(params: dict[str, Any] | None) -> dict[str, Any]:
    """Drop None-valued query params so httpx doesn't serialize them."""
    if not params:
        return {}
    return {k: v for k, v in params.items() if v is not None}


def _graph_slice(result: dict[str, Any]) -> GraphSlice:
    nodes = tuple(
        GraphNode(
            iri=n["id"],
            depth=n["depth"],
            record_class=n.get("sbol_class"),
            display_id=n.get("display_id"),
        )
        for n in result.get("nodes", [])
    )
    edges = tuple(
        GraphEdge(
            subject=e["subject"],
            predicate=e["predicate"],
            object=str(e["object"]),
            depth=e["depth"],
        )
        for e in result.get("edges", [])
    )
    return GraphSlice(
        root_iri=result["root_iri"],
        nodes=nodes,
        edges=edges,
        truncated=bool(result.get("truncated", False)),
    )


def _extract_label(record: dict[str, Any], label_key: str) -> float | int | None:
    """Find a numeric label in the object's JSON-LD slice by predicate local-name."""
    data = record.get("data") or {}
    for key, value in data.items():
        if local_name(key) != label_key:
            continue
        scalar = value[0] if isinstance(value, list) and value else value
        if isinstance(scalar, dict):
            scalar = scalar.get("@value", scalar.get("value"))
        try:
            num = float(scalar)
        except (TypeError, ValueError):
            return None
        return int(num) if num.is_integer() else num
    return None
