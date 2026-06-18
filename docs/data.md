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
| `list_objects(...)` | `GET /objects/list` | Keyset-paginated stream of object records (up to 5000 per page). |
| `get_object(iri)` | `GET /objects` | Resolve one object by IRI. |
| `lookup_objects(iris)` | `POST /objects/lookup` | Resolve up to 1000 IRIs at once. |
| `neighborhood(iri, depth, direction, predicates)` | `GET /objects/neighborhood` | Bounded graph traversal, returning a `GraphSlice`. |
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
- **SBOL RDF** (`.ttl`/`.rdf`/`.xml`/`.nt`) — parsed with rdflib. When the
  document has SBOL3 `Component` top-levels, one record is yielded per Component,
  carrying its `sequence` (`hasSequence`), `roles`/`types`, annotated `features`
  (`hasFeature` → SubComponent / SequenceFeature, with roles and `Range`
  locations), and the composition `GraphSlice`. So real SBOL3 designs feed the
  `structure_aware` and `graph` modalities, not just `sequence`. Documents
  without Components — bare `sbol:Sequence` subjects, including SBOL2
  `ComponentDefinition` sequences — yield one sequence-only record each.

```
>partA measure=12.5
ACGTACGT...
```

### Normalizing other formats to SBOL3

GenBank and SBOL2 inputs reach the SBOL3 path through the [`sbol`](https://github.com/marpaia/sbol-rs)
CLI, run as an offline preprocessing step. Install it with Cargo:

```bash
cargo install sbol-cli
```

`scripts/normalize_sbol.sh` then converts a directory of mixed inputs into SBOL3
Turtle:

```bash
NAMESPACE=https://example.org/mydesigns scripts/normalize_sbol.sh raw/ normalized/
# then point a local corpus at normalized/ with fmt: sbol
```

The `sbol` binary is located via the `SBOL_BIN` environment variable, then an
`SBOL_BIN=` line in the repo-root `.env`, then `sbol` on `PATH` (where
`cargo install` puts it). `examples/normalize_and_ingest.py` runs this end to end
against a bundled GenBank demo.

GenBank (`.gb`/`.gbk`) is imported, SBOL2 RDF is upgraded, and existing SBOL3 is
re-serialized to Turtle. `NAMESPACE` roots the resulting top-levels and is
required for GenBank (which carries no namespace). **FASTA is excluded** — the
importer writes a header's `key=value` into `sbol:description` rather than a
numeric predicate, so labels would not survive; feed labeled FASTA directly with
`fmt: fasta`, whose header parsing reads `measure=...` correctly.

## Synthetic (`source: synthetic`)

`sboltorch.data.synthetic` generates deterministic transcriptional units, each
ordering a promoter, RBS, CDS, and terminator, as `SbolObject`s carrying
sequence, features (sub-components with `Range` locations, roles, orientation),
and a composition `GraphSlice`. Parts come from a shared catalog, so a given part
recurs across components and the composition graphs overlap rather than standing
alone. A per-promoter "strength" provides a learnable supervised label.

```python
from sboltorch.data import generate_components, SyntheticCorpus, write_sbol_turtle
components = generate_components(128, seed=0)        # SbolObjects with sequence, features, graph
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
