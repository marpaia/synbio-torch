# Configuration reference

A run is fully specified by one YAML file validated into a `RunConfig`
(`synbiotorch.config`). Every field has a default, so configs only need to state
what differs. The resolved config is written to `<output_dir>/config.resolved.yaml`
at the start of each run.

```python
from synbiotorch import RunConfig
config = RunConfig.from_yaml("examples/configs/train_graph.yaml")
```

## Top level

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `seed` | int | `42` | Seeds Python, NumPy, and torch; also seeds the split. |
| `output_dir` | str | `runs/default` | Resolved config, `metrics.jsonl`, `best.pt`, `final_metrics.json` (and `backbone/` for MLM) are written here. |
| `streaming` | bool | `false` | Stream the corpus from sharded Parquet instead of loading it into RAM. Requires `splits.strategy: hash` and `train.max_steps`. Sequence/MLM modalities only (not graph). |
| `corpus` | object | — | **Required.** Where training data comes from. |
| `tokenizer` | object | defaults | Sequence tokenization (ignored by the graph encoder). |
| `encoder` | object | defaults | Input modality. |
| `model` | object | defaults | Backbone + architecture. |
| `task` | object | defaults | Training objective. |
| `packing` | object | defaults | Token packing for LM pretraining (off by default). |
| `splits` | object | defaults | Train/val/test partitioning. |
| `train` | object | defaults | Optimization loop. |
| `wandb` | object | defaults | Weights & Biases tracking (disabled by default). |

## `corpus`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `source` | `fasta` \| `sbol` \| `genbank` \| `table` \| `synthetic` \| `sbol_db` | `synthetic` | Data source. |
| `base_url` | str | `null` | sbol-db base URL. **Required** when `source: sbol_db`. |
| `username` / `password` | str | `null` | Optional basic auth for sbol-db. |
| `record_class` | str | `null` | Filter to one SBOL class IRI (e.g. `http://sbols.org/v3#Sequence`). |
| `role` | str | `null` | Filter to one role IRI (e.g. a Sequence Ontology term). |
| `document_id` | str | `null` | Filter to one source document. |
| `path` | str | `null` | File or directory. **Required** for `fasta`/`sbol`/`genbank`/`table`. |
| `namespace` | str | `null` | Roots identities for GenBank import and the SBOL2→3 upgrade. **Required** when `source: genbank`. |
| `alphabet` | `auto` \| `dna` \| `rna` \| `protein` | `auto` | FASTA/table alphabet; `auto` detects from sequence content. |
| `sequence_column` | str | `null` | Table column holding the sequence. **Required** when `source: table`. |
| `label_column` | str | `null` | Table column holding the numeric label. |
| `id_column` | str | `null` | Table column holding the record id (else `<file>:<row>`). |
| `n` | int | `64` | Number of components when `source: synthetic`. |
| `synthetic_seed` | int | `0` | Seed for the synthetic generator. |
| `label_key` | str | `null` | Where the supervised label comes from: an sbol-db predicate local-name, a FASTA header `key=value`, or an SBOL annotation predicate. `null` means unlabeled (pretraining). For the synthetic source, any non-null value enables labels. |
| `cache_dir` | str | `.synbiotorch_cache` | Where the materialized Parquet corpus is stored. |
| `shard_size` | int | `50000` | Rows per Parquet shard. Sharding keeps materializing and streaming memory-bounded for a corpus larger than RAM. |

See [data.md](data.md) for the corpus sources in depth.

## `tokenizer`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `kind` | `hf` \| `kmer` \| `char` | `hf` | `hf` wraps any HuggingFace `AutoTokenizer`; `kmer` is overlapping k-mers; `char` is IUPAC character-level. |
| `k` | int | `6` | k-mer size (`kmer` only). |
| `stride` | int | `1` | k-mer stride (`kmer` only). |
| `max_length` | int | `512` | Max tokens per sequence. |
| `model_name` | str | `zhihan1996/DNABERT-2-117M` | Hub id for the `hf` tokenizer. |
| `alphabet` | `dna` \| `protein` | `dna` | Character alphabet (`char` only): nucleotide or amino acids. |

## `encoder`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `kind` | `sequence` \| `structure_aware` \| `graph` | `sequence` | Input modality. |
| `roles` | list[str] | `null` | `structure_aware`: role IRIs that get dedicated boundary markers. `null` falls back to a default Sequence Ontology set (promoter/RBS/CDS/terminator). |
| `mark_orientation` | bool | `true` | `structure_aware`: emit a marker for reverse-complement features. |

## `model`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `backbone` | str | `zhihan1996/DNABERT-2-117M` | Hub id **or** local path to a saved backbone (e.g. from an MLM run). Used when `from_scratch: false`. |
| `hidden_size` | int | `768` | Hidden width; for graphs must be divisible by `arch.num_attention_heads`. |
| `dropout` | float | `0.1` | Head dropout. |
| `from_scratch` | bool | `false` | Build an untrained encoder from `arch` + the tokenizer/encoder vocab instead of loading pretrained weights. |
| `arch` | object | defaults | Architecture for `from_scratch` (and the graph model). |

### `model.arch`

Used when `model.from_scratch: true`. `model_type` selects any HuggingFace
architecture — `bert`/`modernbert` for MLM, `gpt2`/`gpt_neox`/`llama` for causal.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model_type` | str | `bert` | HuggingFace architecture id. RoPE types (`modernbert`, `gpt_neox`, `llama`) extrapolate past `max_position_embeddings`; absolute-position types (`bert`, `gpt2`) are capped at it. |
| `num_hidden_layers` | int | `6` | |
| `num_attention_heads` | int | `6` | |
| `intermediate_size` | int | `1536` | |
| `max_position_embeddings` | int | `2048` | Context length. |
| `attn_implementation` | `sdpa` \| `eager` \| `flash_attention_2` | `sdpa` | SDPA uses PyTorch's fused attention (FlashAttention kernels on CUDA) and works on CPU; `flash_attention_2` needs the flash-attn package + CUDA. |
| `rope_theta` | float | `null` | Rotary base for RoPE architectures; `null` uses the architecture default. |

## `task`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `kind` | `supervised` \| `frozen` \| `mlm` \| `causal` | `frozen` | `frozen` trains a head over a frozen backbone; `supervised` fine-tunes end to end; `mlm` is masked-LM pretraining; `causal` is decoder next-token pretraining (set a decoder `model.arch.model_type`, e.g. `gpt2`). |
| `objective` | `regression` \| `classification` | `regression` | Supervised head + loss. |
| `num_classes` | int | `null` | **Required** for `classification`. |
| `target_transform` | `none` \| `log1p` | `none` | `log1p` trains regression in log space and reports metrics in the original space. |
| `mlm_probability` | float | `0.15` | Fraction of content tokens masked (`mlm` only). |

See [capabilities.md](capabilities.md) for how modality and objective combine.

## `packing`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `enabled` | bool | `false` | Concatenate tokenized documents into fixed-length blocks with no padding, so every position carries signal. The training unit becomes a block, not a document. For `task.kind: mlm` or `causal`, used with `streaming`. |
| `block_size` | int | `512` | Tokens per packed block. Must be ≤ `model.arch.max_position_embeddings`. |

## `splits`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `strategy` | `random` \| `stratified` \| `hash` | `random` | `stratified` balances label bins across partitions (requires labels); `hash` assigns each record by hashing its IRI — stable as the corpus grows and required for `streaming`. |
| `ratios` | [float, float, float] | `[0.8, 0.1, 0.1]` | train/val/test; must sum to 1.0. |

The `random`/`stratified` index splits are pure functions of `(n, ratios, seed,
strategy)`; the `hash` split is a pure function of `(iri, ratios, seed)` per
record — both reproducible across runs.

## `train`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `batch_size` | int | `16` | |
| `epochs` | int | `10` | |
| `lr` | float | `2e-5` | AdamW learning rate. |
| `weight_decay` | float | `0.01` | |
| `grad_accum` | int | `1` | Gradient accumulation steps. |
| `max_grad_norm` | float | `1.0` | Gradient clipping. |
| `amp` | bool | `true` | Mixed precision (active on CUDA). |
| `precision` | `fp16` \| `bf16` \| `fp32` | `fp16` | Autocast dtype when `amp` is on and the device is CUDA. `bf16` needs no loss scaler and is the stable choice at scale; `fp32` disables autocast even with `amp` on. |
| `num_workers` | int | `0` | DataLoader workers. |
| `max_steps` | int | `null` | Step budget. When set, optimizer steps (not `epochs`) end the run, and evaluation/checkpointing can follow a step cadence — the mode for pretraining over a large corpus. |
| `eval_every_n_steps` | int | `null` | Validate every N optimizer steps instead of at epoch boundaries. |
| `checkpoint_every_n_steps` | int | `null` | Write a rolling, resumable `last.pt` every N steps. |
| `gradient_checkpointing` | bool | `false` | Trade compute for memory in the transformer (HuggingFace backbones/LMs). |
| `compile` | bool | `false` | Wrap the model in `torch.compile`. |
| `distributed` | object | defaults | Multi-GPU / multi-node strategy (single-process by default). |
| `early_stop` | object | `null` | Omit to disable. |

Checkpoints (`best.pt`, `last.pt`) carry full optimizer/scheduler/scaler/RNG
state. Resume a run with `synbiotorch train <config> --resume <output_dir>/last.pt`;
it continues from the next epoch boundary after the checkpointed one, with the
step counter and LR schedule intact.

### `train.distributed`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `strategy` | `none` \| `ddp` | `none` | `ddp` replicates the model and all-reduces gradients across ranks (works on CPU/gloo). It does not shard parameters, so it gives no memory saving. |
| `find_unused_parameters` | bool | `false` | DDP only; enable when some parameters get no gradient (e.g. an unused pooler). |

Launch a distributed run with `torchrun`, which sets the rank/world environment:

```bash
torchrun --nproc_per_node=<gpus> -m synbiotorch.cli train <config>   # set train.distributed.strategy: ddp
```

Each rank reads a disjoint slice of the data (a `DistributedSampler` for in-memory
corpora; rank×worker shard assignment for `streaming`). Only rank 0 writes
checkpoints, `metrics.jsonl`, and W&B; validation metrics are averaged across
ranks so metric-driven decisions stay consistent.

`ddp` is for data-parallel scaling, not for fitting a model larger than one
device. Sharded training (FSDP/ZeRO), which shards parameters/grads/optimizer
state across GPUs, is a planned addition that needs validation on real GPU
hardware and is not part of this layer yet.

### `train.early_stop`

| Field | Type | Default |
|-------|------|---------|
| `monitor` | str | `val_loss` |
| `patience` | int | `5` |
| `mode` | `min` \| `max` | `min` |
| `min_delta` | float | `0.0` |

Monitor names are `train_loss`, `val_loss`, and the task's validation metrics
(`val_mae` / `val_mse` / `val_r2`, `val_accuracy`, or `val_masked_accuracy`).

## `wandb`

Tracking is off unless `enabled: true`. When on, the `WandbLogger` callback logs
per-step `train/step_loss` and `train/lr`, per-epoch `train/*` and `val/*`
metrics, the resolved `RunConfig` as the run config, and the corpus fingerprint,
object count, split sizes, and seed as run-summary lineage. The API key is read
from the `WANDB_API_KEY` environment variable and is never a config field.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `enabled` | bool | `false` | Turn tracking on. |
| `project` | str | `null` | W&B project. |
| `entity` | str | `null` | W&B entity (team/user). |
| `mode` | `online` \| `offline` \| `disabled` | `online` | `offline` logs locally for later sync; `disabled` is a no-op backend. |
| `run_name` | str | `null` | Run display name. |
| `tags` | list[str] | `[]` | Run tags. |
| `group` | str | `null` | Run group. |
| `job_type` | str | `null` | Run job type. |
| `watch_model` | bool | `false` | Log gradient/parameter histograms via `wandb.watch` (heavy on large backbones). |
| `log_freq` | int | `100` | Steps between per-step metric flushes and gradient logs. |
| `log_model` | bool | `true` | Push `best.pt` + `config.resolved.yaml` as a `model` Artifact (aliased `best` and the corpus fingerprint) at train end. |
