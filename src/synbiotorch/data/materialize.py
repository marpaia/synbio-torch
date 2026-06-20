"""Materialize a corpus to a versioned, sharded Parquet cache.

A long training run should not depend on a live database, and two runs over the
"same data" must be byte-for-byte comparable. Materialization streams a corpus
once into fixed-size Parquet shards, hashing the contents into a fingerprint.
Re-materializing the same data is a no-op that returns the cached shards. Both
writing and reading go a shard at a time, so the cache works for a corpus larger
than memory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SHARD_SIZE = 50_000

from synbiotorch.data.corpus import Corpus
from synbiotorch.types import (
    Alphabet,
    Design,
    Sequence,
    feature_from_dict,
    feature_to_dict,
    graph_from_dict,
    graph_to_dict,
)

_SCHEMA = pa.schema(
    [
        ("iri", pa.string()),
        ("record_class", pa.string()),
        ("display_id", pa.string()),
        ("name", pa.string()),
        ("roles", pa.list_(pa.string())),
        ("types", pa.list_(pa.string())),
        ("elements", pa.string()),
        ("alphabet", pa.string()),
        ("encoding_iri", pa.string()),
        ("label", pa.float64()),
        ("features_json", pa.string()),
        ("graph_json", pa.string()),
        ("raw_json", pa.string()),
    ]
)


@dataclass(frozen=True)
class MaterializedCorpus:
    """A sharded-Parquet-backed corpus: streamable, reproducible, offline."""

    path: Path
    fingerprint: str
    count: int
    shards: tuple[str, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return self.count

    def shard_paths(self) -> list[Path]:
        if self.shards:
            return [self.path / name for name in self.shards]
        # A single legacy ``data.parquet`` written before sharding.
        legacy = self.path / "data.parquet"
        return [legacy] if legacy.exists() else []

    def labels(self) -> list[float | int | None]:
        out: list[float | int | None] = []
        for shard in self.shard_paths():
            table = pq.read_table(shard, columns=["label"])
            out.extend(table.column("label").to_pylist())
        return out

    def _iter_rows(self, shards: list[Path]) -> Iterator[Design]:
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches():
                for row in batch.to_pylist():
                    yield _row_to_object(row)

    def __iter__(self) -> Iterator[Design]:
        # Stream a shard at a time rather than reading the whole corpus into RAM.
        return self._iter_rows(self.shard_paths())

    def iter_for_worker(self, worker_id: int, num_workers: int) -> Iterator[Design]:
        """Stream only the shards assigned to one DataLoader worker.

        Whole shards are partitioned across workers round-robin, so the union over
        all workers is exactly the corpus with no record read twice.
        """
        paths = self.shard_paths()
        assigned = [p for i, p in enumerate(paths) if i % num_workers == worker_id]
        return self._iter_rows(assigned)

    def read_all(self) -> list[Design]:
        return list(self)


def materialize(
    corpus: Corpus, cache_dir: str | Path, *, force: bool = False, shard_size: int = DEFAULT_SHARD_SIZE
) -> MaterializedCorpus:
    """Stream ``corpus`` to sharded Parquet under ``cache_dir``, keyed by a content hash."""
    cache_root = Path(cache_dir)
    namespace = corpus.fingerprint()
    staging = cache_root / namespace / "staging"

    # If a completed manifest already exists for this source identity, reuse it.
    existing = _find_complete(cache_root / namespace)
    if existing is not None and not force:
        return existing

    # The target dir is named by the content hash, which we only know after the
    # full stream — so write shards to staging, then atomically rename into place.
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    count = 0
    shard_names: list[str] = []
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffer:
            return
        name = f"part-{len(shard_names):05d}.parquet"
        pq.write_table(pa.Table.from_pylist(buffer, schema=_SCHEMA), staging / name)
        shard_names.append(name)
        buffer.clear()

    for obj in corpus:
        buffer.append(_object_to_row(obj))
        hasher.update(_hash_payload(obj))
        count += 1
        if len(buffer) >= shard_size:
            flush()
    flush()
    if not shard_names:  # an empty corpus still gets one (empty) shard
        pq.write_table(_SCHEMA.empty_table(), staging / "part-00000.parquet")
        shard_names.append("part-00000.parquet")

    fingerprint = f"{namespace}-{hasher.hexdigest()[:16]}"
    target = cache_root / namespace / fingerprint
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    manifest = {"fingerprint": fingerprint, "count": count, "namespace": namespace, "shards": shard_names}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return MaterializedCorpus(path=target, fingerprint=fingerprint, count=count, shards=tuple(shard_names))


def _find_complete(namespace_dir: Path) -> MaterializedCorpus | None:
    if not namespace_dir.exists():
        return None
    for child in sorted(namespace_dir.iterdir()):
        if child.name == "staging":
            continue
        manifest = child / "manifest.json"
        if not manifest.exists():
            continue
        meta = json.loads(manifest.read_text())
        candidate = MaterializedCorpus(
            path=child,
            fingerprint=meta["fingerprint"],
            count=meta["count"],
            shards=tuple(meta.get("shards", ())),
        )
        if candidate.shard_paths():
            return candidate
    return None


def _hash_payload(obj: Design) -> bytes:
    seq = obj.sequence.elements if obj.sequence else ""
    return f"{obj.iri}\x00{seq}\x00{obj.label}".encode()


def _object_to_row(obj: Design) -> dict[str, Any]:
    seq = obj.sequence
    return {
        "iri": obj.iri,
        "record_class": obj.record_class,
        "display_id": obj.display_id,
        "name": obj.name,
        "roles": list(obj.roles),
        "types": list(obj.types),
        "elements": seq.elements if seq else None,
        "alphabet": seq.alphabet.value if seq else None,
        "encoding_iri": seq.encoding_iri if seq else None,
        "label": float(obj.label) if obj.label is not None else None,
        "features_json": json.dumps([feature_to_dict(f) for f in obj.features]) if obj.features else None,
        "graph_json": json.dumps(graph_to_dict(obj.neighbors)) if obj.neighbors is not None else None,
        "raw_json": json.dumps(obj.raw, default=str),
    }


def _row_to_object(row: dict[str, Any]) -> Design:
    elements = row.get("elements")
    sequence = None
    if elements:
        sequence = Sequence(
            elements=str(elements),
            alphabet=Alphabet(row["alphabet"]) if row.get("alphabet") else Alphabet.DNA,
            encoding_iri=row.get("encoding_iri"),  # type: ignore[arg-type]
        )
    raw_json = row.get("raw_json")
    features_json = row.get("features_json")
    graph_json = row.get("graph_json")
    return Design(
        iri=str(row["iri"]),
        record_class=str(row.get("record_class") or ""),
        display_id=row.get("display_id"),  # type: ignore[arg-type]
        name=row.get("name"),  # type: ignore[arg-type]
        roles=tuple(row.get("roles") or ()),
        types=tuple(row.get("types") or ()),
        sequence=sequence,
        features=tuple(feature_from_dict(f) for f in json.loads(features_json)) if features_json else (),
        neighbors=graph_from_dict(json.loads(graph_json)) if graph_json else None,
        label=row.get("label"),  # type: ignore[arg-type]
        raw=json.loads(raw_json) if raw_json else {},
    )
