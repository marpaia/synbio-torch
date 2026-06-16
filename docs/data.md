# Data sources

Every source normalizes to `SbolObject` records, so training code never branches
on where data came from. A source implements the `Corpus` protocol
(`__iter__ -> Iterator[SbolObject]` and `fingerprint() -> str`).

```python
from sboltorch import SbolObject  # iri, sbol_class, roles, types, sequence, features, neighbors, label, raw
```

## sbol-db (`source: sbol_db`)

`SbolDbClient` (`sboltorch.data.sbol_db`) is a typed client over the sbol-db REST
API. Iterating it streams objects matching the configured `sbol_class` / `role` /
`document_id` filters, reading the supervised label from each object's JSON-LD
slice when `label_key` is set.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `list_objects(...)` | `GET /objects/list` | Keyset-paginated stream of object records (≤5000/page). |
| `get_object(iri)` | `GET /objects` | Resolve one object by IRI. |
| `lookup_objects(iris)` | `POST /objects/lookup` | Resolve up to 1000 IRIs at once. |
| `neighborhood(iri, depth, direction, predicates)` | `GET /objects/neighborhood` | Bounded graph traversal → `GraphSlice`. |
| `search_sequence(pattern, ...)` | `GET /sequences/search` | k-mer-indexed substring + reverse-complement search. |
| `ontology_descendants(iri)` | `GET /ontology/descendants` | Transitive `is_a` expansion of a role/type term. |

```python
from sboltorch.data import SbolDbClient
with SbolDbClient("http://localhost:8080", role="https://identifiers.org/SO:0000167",
                  label_key="measure") as client:
    for obj in client:
        ...  # SbolObject with sequence + label
```

Sequence elements are extracted from the lossless JSON-LD `data` slice by
predicate local-name, so the client is robust to IRI compaction.

## Local files (`source: local`)

`LocalFileCorpus` reads a file or a directory of files:

- **FASTA** (`.fa`/`.fasta`/`.fna`) — one record per sequence; labels parsed from
  `key=value` tokens in the header when `label_key` is set.
- **SBOL RDF** (`.ttl`/`.rdf`/`.xml`/`.nt`) — parsed with rdflib; yields one
  record per `sbol:Sequence` subject (SBOL2 and SBOL3 element predicates).

```
>partA measure=12.5
ACGTACGT...
```

## Synthetic (`source: synthetic`)

`sboltorch.data.synthetic` generates deterministic transcriptional units —
promoter → RBS → CDS → terminator — as rich `SbolObject`s with sequence,
features (sub-components with `Range` locations, roles, orientation), and a
composition `GraphSlice`. Parts are drawn from a shared catalog, so the same part
recurs across components and the composition graphs have real structure. A
per-promoter "strength" provides a learnable supervised label.

```python
from sboltorch.data import generate_components, SyntheticCorpus, write_sbol_turtle
components = generate_components(128, seed=0)        # rich SbolObjects
write_sbol_turtle(components, "out.ttl")             # serialize to SBOL3 Turtle
```

This drives development and testing of the structure-aware and graph modalities
without a populated sbol-db.

## Materialization & caching

`materialize(corpus, cache_dir)` streams a corpus once into a versioned Parquet
shard under `cache_dir/<source-fingerprint>/<content-fingerprint>/`, hashing the
contents into the fingerprint. Sequence, features, and the composition graph are
all persisted, so a materialized corpus round-trips every modality. Re-running
over the same data returns the cached shard — a training run is reproducible and
offline, decoupled from a live database.

```bash
sboltorch ingest examples/configs/train_graph.yaml   # materialize without training
```

## Test fixtures

- **Synthetic** — generated in-memory; no files committed.
- **Real SBOL** — a curated subset of the SynBioDex SBOLTestSuite under
  `tests/fixtures/sbol/` (SBOL2 RDF/XML, SBOL3 Turtle/N-Triples), covering
  sequence-bearing and abstract designs. See
  [`tests/fixtures/sbol/PROVENANCE.md`](../tests/fixtures/sbol/PROVENANCE.md) for
  source, pinned commit, and licensing notes.
