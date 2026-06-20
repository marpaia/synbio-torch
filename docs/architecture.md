# Architecture

synbio-torch turns biological designs and sequences into trained transformer
models. Three ideas shape it: every source normalizes to one record type, the
parts that vary plug in behind protocols, and the training engine stays small and
explicit.

## One record type

Every data source — FASTA, CSV/TSV tables, GenBank, SBOL, the sbol-db REST API, or
the synthetic generator — is normalized into `Design` (`synbiotorch.types`):

```
Design(iri, record_class, roles, types, sequence, features, neighbors, label, raw)
```

Training code consumes only `Design`s and never branches on provenance. A
source is anything satisfying the `Corpus` protocol (`__iter__` yielding
`Design`s, plus a `fingerprint()` for caching).

## Swappable plug points

Three independent axes are each a `Protocol` with interchangeable implementations,
selected by configuration:

- **Tokenizer** — how a sequence becomes tokens (`hf`, `kmer`, `char`).
- **Encoder** — the input modality, turning an `Design` into model input
  (`sequence`, `structure_aware`, `graph`).
- **Task** — the training objective, owning loss and metrics (`supervised`,
  `frozen`, `mlm`, `causal`).

Adding an implementation and registering it in the matching `build_*` factory
extends a capability without touching the engine. See [extending.md](extending.md).

## The training engine

The training loop (`synbiotorch.engine`) is plain PyTorch: AMP (`fp16`/`bf16`),
gradient accumulation and clipping, a linear warmup/decay schedule, optional
gradient checkpointing and `torch.compile`, and a list of callbacks
(`EarlyStopping`, `ModelCheckpoint`, `PeriodicCheckpoint`, `MetricLogger`,
`WandbLogger`). A run is bounded by epochs or by a step budget (`max_steps`), and
checkpoints carry full optimizer/scheduler/scaler/RNG state so a run resumes after
an interruption. The loop learns the batch shape only through a `BatchAdapter`, so
it trains a sequence model (tensor-dict batches) or a graph model (PyG `Batch`
objects), and it wraps the model for data-parallel (DDP) training when configured.

## Data flow

A `RunConfig` drives the whole pipeline. The configured `Corpus` source is
materialized to sharded Parquet and split into train/val/test. The default path
loads it into memory and splits by index; the streaming path iterates the shards
lazily and assigns each record to a partition by a hash split, optionally packing
tokens into fixed-length blocks. The `Encoder` turns each `Design` into model
input for its modality, using the `Tokenizer`, and a `DataLoader` batches the
result through a collator (padding, MLM masking, or causal next-token shift). The
`Trainer` then runs the loop under the `Task` and `BatchAdapter`, writing
checkpoints and metrics.

## Layers

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Config | `synbiotorch.config` | One Pydantic `RunConfig` per run; validated, serialized. |
| Sources | `synbiotorch.sources` | Corpus sources (FASTA, table, GenBank, SBOL, sbol-db, synthetic), each normalizing to `Design`. |
| Data | `synbiotorch.data` | The `Corpus` protocol, `build_corpus`, and sharded Parquet materialization. |
| Tokenize | `synbiotorch.tokenize` | `hf` / `kmer` / `char` behind one protocol (encode + decode). |
| Encoders | `synbiotorch.encoders` | Turn a `Design` into model input, per modality. |
| Datasets | `synbiotorch.datasets` | Map-style and streaming `Dataset`s, token packing, padding / MLM / causal collators, seeded and hash splits. |
| Models | `synbiotorch.models` | Backbone (pretrained or from-scratch) + pooling + head; MLM, causal, and graph models. |
| Tasks | `synbiotorch.tasks` | Loss, metrics, label dtype, target transform. |
| Engine | `synbiotorch.engine` | Training loop, callbacks, batch adapters. |
| Distributed | `synbiotorch.distributed` | Process-group setup, rank-aware data/IO, metric reduction (DDP). |
| Generate | `synbiotorch.generate` | Autoregressive sampling and design completion from a causal backbone. |
| Pipeline | `synbiotorch.pipeline` | Wires the layers from a `RunConfig`. |

## Key protocols

- `Corpus`: `__iter__() -> Iterator[Design]`, `fingerprint() -> str`.
- `Tokenizer`: `encode`, `tokenize_content`, `vocab_size`, `pad_token_id`,
  `mask_token_id`, `special_token_ids`, `max_length`.
- `Encoder`: `encode(Design) -> ModelInput`, `output_spec -> EncoderSpec`.
- `Task`: `loss`, `predict`, `epoch_metrics`, `primary_metric`, `label_dtype`.
- `BatchAdapter`: `to_device`, `forward(model, batch)`, `labels(batch)`.
- `Callback`: `on_train_start`, `on_epoch_end`, `on_train_end`.

## Reproducibility

- A run is fully specified by its `RunConfig`; the resolved config is written to
  `<output_dir>/config.resolved.yaml`.
- `seed` seeds Python, NumPy, and torch. The in-memory split is a pure function of
  `(n, ratios, seed, strategy)`; the streaming `hash` split is a pure function of
  `(iri, ratios, seed)` per record, stable as the corpus grows.
- Corpora are materialized to content-fingerprinted sharded Parquet, so a run is
  offline and byte-for-byte comparable across executions. See [data.md](data.md).
- Checkpoints carry optimizer/scheduler/scaler/RNG state, so a resumed run
  continues equivalently to an uninterrupted one.

## Scaling

For corpora and models that outgrow a single in-memory, single-device run:

- **Streaming** (`streaming: true`) iterates sharded Parquet a shard at a time
  with a hash split, so the corpus need not fit in RAM. **Packing** concatenates
  tokenized documents into fixed-length blocks for LM pretraining.
- **Long context** comes from RoPE architectures (`gpt_neox`/`llama`/`modernbert`)
  plus SDPA/FlashAttention; see [backbones.md](backbones.md).
- **DDP** (`train.distributed.strategy: ddp`, launched with `torchrun`) replicates
  the model and all-reduces gradients across ranks, with rank-aware data sharding,
  rank-0-only IO, and cross-rank metric reduction. It is data-parallel only (no
  parameter sharding); see [configuration.md](configuration.md).

## Consuming sbol-db

`SbolDbClient` reads designs over the sbol-db REST API: keyset-paginated object
listing, single/bulk IRI resolution, bounded neighborhood traversal, sequence
search, and ontology descendant expansion. Sequence elements are read from each
object's JSON-LD slice by predicate local-name. Details in [data.md](data.md).
