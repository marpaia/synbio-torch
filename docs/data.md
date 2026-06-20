# Data sources

Every source normalizes to `Design` records, so training code never branches on
where data came from. A source implements the `Corpus` protocol
(`__iter__ -> Iterator[Design]` and `fingerprint() -> str`) and lives in
`synbiotorch.sources`; the source-neutral pipeline (the `Corpus` protocol,
`build_corpus`, and materialization) lives in `synbiotorch.data`.

```python
from synbiotorch import Design  # iri, record_class, roles, types, sequence, features, neighbors, label, raw
```

The active source is `corpus.source` in the run config.

## FASTA (`source: fasta`)

`FastaCorpus` reads a `.fa`/`.fasta`/`.fna`/`.faa` file (or a directory of them),
one record per sequence. Labels are parsed from `key=value` tokens in the header
when `label_key` is set. The alphabet is auto-detected (DNA/RNA/protein) unless
`corpus.alphabet` overrides it.

```
>partA measure=12.5
ACGTACGT...
```

## Tables (`source: table`)

`TableCorpus` reads CSV/TSV — the shape most public sequence-activity datasets
ship in — one `Design` per row. Set `sequence_column` and, for supervised runs,
`label_column` (plus optional `id_column`). The delimiter follows the extension
(`.tsv` → tab). This is the most direct path for labeled sequence corpora.

```yaml
corpus:
  source: table
  path: data/promoters.csv
  sequence_column: sequence
  label_column: strength
```

## GenBank (`source: genbank`)

`GenbankCorpus` imports `.gb`/`.gbk` files to SBOL 3 in-process via the native
sbol-rs binding, yielding a `Design` per record with sequence, SO-mapped
features, and the composition graph — so GenBank feeds the `structure_aware` and
`graph` modalities, not just `sequence`. `namespace` roots the resulting
identities (GenBank carries none) and is required.

## SBOL (`source: sbol`)

`SbolFileCorpus` reads SBOL RDF (`.ttl`/`.rdf`/`.xml`/`.nt`). SBOL 3 documents are
read directly; SBOL 2 documents are upgraded to SBOL 3 — both in-process through
the binding. Documents with `Component` top-levels yield one record per Component
(sequence, annotated features, composition graph); documents of bare sequences
yield one sequence-only record each. A numeric `label_key` is read from a matching
annotation predicate on the object.

GenBank import and SBOL parsing are handled by the vendored PyO3 bindings to
[sbol-rs](https://github.com/marpaia/sbol-rs) — no external tool or RDF library is
involved.

## sbol-db (`source: sbol_db`)

`SbolDbClient` (`synbiotorch.sources.sbol_db`) is a typed client over the sbol-db
REST API. Iterating it streams objects matching the configured `record_class` /
`role` / `document_id` filters, reading the supervised label from each object's
JSON-LD slice when `label_key` is set.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `list_objects(...)` | `GET /objects/list` | Keyset-paginated stream of object records. |
| `get_object(iri)` | `GET /objects` | Resolve one object by IRI. |
| `lookup_objects(iris)` | `POST /objects/lookup` | Resolve up to 1000 IRIs at once. |
| `neighborhood(iri, ...)` | `GET /objects/neighborhood` | Bounded graph traversal, returning a `GraphSlice`. |
| `search_sequence(pattern, ...)` | `GET /sequences/search` | k-mer-indexed substring + reverse-complement search. |
| `ontology_descendants(iri)` | `GET /ontology/descendants` | Transitive `is_a` expansion of a role/type term. |

Sequence elements are extracted from the lossless JSON-LD `data` slice by
predicate local-name, so the client is robust to IRI compaction.

## Synthetic (`source: synthetic`)

`synbiotorch.sources.synthetic` generates deterministic transcriptional units,
each ordering a promoter, RBS, CDS, and terminator, as `Design`s carrying
sequence, features (sub-components with `Range` locations, roles, orientation),
and a composition `GraphSlice`. Parts come from a shared catalog, so a given part
recurs across components and the composition graphs overlap. A per-promoter
"strength" provides a learnable supervised label. This drives development and
testing of the structure-aware and graph modalities without external data.

```python
from synbiotorch.sources import generate_components, write_sbol_turtle
designs = generate_components(128, seed=0)       # Designs with sequence, features, graph
write_sbol_turtle(designs, "out.ttl")            # serialize to SBOL3 Turtle
```

## Materialization & caching

`materialize(corpus, cache_dir)` streams a corpus once into versioned Parquet
shards under `cache_dir/<source-fingerprint>/<content-fingerprint>/`, hashing the
contents into the fingerprint. Sequence, features, and the composition graph are
all persisted, so a materialized corpus round-trips every modality. Re-running
over the same data returns the cached shards — a training run is reproducible and
offline, decoupled from a live database.

Writing and reading both go a shard at a time (`corpus.shard_size` rows each), so
the cache holds a corpus larger than memory. The in-memory pipeline reads all
shards into a list and splits by index; the streaming pipeline (`streaming: true`)
iterates shards lazily, assigns each record to a partition by `hash` split, and —
under a multi-worker DataLoader — gives each worker a disjoint set of shards.

```bash
synbiotorch ingest examples/configs/ingest_genbank.yaml   # materialize without training
```

## Test fixtures

- **Synthetic** — generated in-memory; no files committed.
- **Real SBOL** — a curated subset of the SynBioDex SBOLTestSuite under
  `tests/fixtures/sbol/` (SBOL2 RDF/XML, SBOL3 Turtle/N-Triples), covering
  sequence-bearing and abstract designs. See
  [`tests/fixtures/sbol/PROVENANCE.md`](../tests/fixtures/sbol/PROVENANCE.md) for
  source, pinned commit, and licensing notes.
