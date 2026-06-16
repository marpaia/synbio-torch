# Configuration reference

A run is fully specified by one YAML file validated into a `RunConfig`
(`sboltorch.config`). Every field has a default, so configs only need to state
what differs. The resolved config is written to `<output_dir>/config.resolved.yaml`
at the start of each run.

```python
from sboltorch import RunConfig
config = RunConfig.from_yaml("examples/configs/train_graph.yaml")
```

## Top level

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `seed` | int | `42` | Seeds Python, NumPy, and torch; also seeds the split. |
| `output_dir` | str | `runs/default` | Resolved config, `metrics.jsonl`, `best.pt`, `final_metrics.json` (and `backbone/` for MLM) are written here. |
| `corpus` | object | — | **Required.** Where training data comes from. |
| `tokenizer` | object | defaults | Sequence tokenization (ignored by the graph encoder). |
| `encoder` | object | defaults | Input modality. |
| `model` | object | defaults | Backbone + architecture. |
| `task` | object | defaults | Training objective. |
| `splits` | object | defaults | Train/val/test partitioning. |
| `train` | object | defaults | Optimization loop. |

## `corpus`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `source` | `sbol_db` \| `local` \| `synthetic` | `sbol_db` | Data source. |
| `base_url` | str | `null` | sbol-db base URL. **Required** when `source: sbol_db`. |
| `username` / `password` | str | `null` | Optional basic auth for sbol-db. |
| `sbol_class` | str | `null` | Filter to one SBOL class IRI (e.g. `http://sbols.org/v3#Sequence`). |
| `role` | str | `null` | Filter to one role IRI (e.g. a Sequence Ontology term). |
| `document_id` | str | `null` | Filter to one source document. |
| `path` | str | `null` | File or directory. **Required** when `source: local`. |
| `fmt` | `fasta` \| `sbol` \| `auto` | `auto` | Local file format; `auto` infers from extension. |
| `n` | int | `64` | Number of components when `source: synthetic`. |
| `synthetic_seed` | int | `0` | Seed for the synthetic generator. |
| `label_key` | str | `null` | Where the supervised label comes from: an sbol-db predicate local-name, or a FASTA header `key=value`. `null` ⇒ unlabeled (pretraining). For the synthetic source, any non-null value enables labels. |
| `cache_dir` | str | `.sboltorch_cache` | Where the materialized Parquet corpus is stored. |

See [data.md](data.md) for the corpus sources in depth.

## `tokenizer`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `kind` | `hf` \| `kmer` \| `char` | `hf` | `hf` wraps any HuggingFace `AutoTokenizer`; `kmer` is overlapping k-mers; `char` is IUPAC character-level. |
| `k` | int | `6` | k-mer size (`kmer` only). |
| `stride` | int | `1` | k-mer stride (`kmer` only). |
| `max_length` | int | `512` | Max tokens per sequence. |
| `model_name` | str | `zhihan1996/DNABERT-2-117M` | Hub id for the `hf` tokenizer. |

## `encoder`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `kind` | `sequence` \| `structure_aware` \| `graph` | `sequence` | Input modality. |
| `roles` | list[str] | `null` | `structure_aware`: role IRIs that get dedicated boundary markers. `null` ⇒ a default Sequence Ontology set (promoter/RBS/CDS/terminator). |
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

| Field | Type | Default |
|-------|------|---------|
| `model_type` | str | `bert` |
| `num_hidden_layers` | int | `6` |
| `num_attention_heads` | int | `6` |
| `intermediate_size` | int | `1536` |
| `max_position_embeddings` | int | `1024` |

## `task`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `kind` | `supervised` \| `frozen` \| `mlm` | `frozen` | `frozen` trains a head over a frozen backbone; `supervised` fine-tunes end to end; `mlm` is masked-language-model pretraining. |
| `objective` | `regression` \| `classification` | `regression` | Supervised head + loss. |
| `num_classes` | int | `null` | **Required** for `classification`. |
| `target_transform` | `none` \| `log1p` | `none` | `log1p` trains regression in log space and reports metrics in the original space. |
| `mlm_probability` | float | `0.15` | Fraction of content tokens masked (`mlm` only). |

See [capabilities.md](capabilities.md) for how modality × objective combine.

## `splits`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `strategy` | `random` \| `stratified` | `random` | `stratified` balances label bins across partitions (requires labels). |
| `ratios` | [float, float, float] | `[0.8, 0.1, 0.1]` | train/val/test; must sum to 1.0. |

Splitting is a pure function of `(n, ratios, seed, strategy)` — reproducible across runs.

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
| `num_workers` | int | `0` | DataLoader workers. |
| `early_stop` | object | `null` | Omit to disable. |

### `train.early_stop`

| Field | Type | Default |
|-------|------|---------|
| `monitor` | str | `val_loss` |
| `patience` | int | `5` |
| `mode` | `min` \| `max` | `min` |
| `min_delta` | float | `0.0` |

Monitor names are `train_loss`, `val_loss`, and the task's validation metrics
(`val_mae` / `val_mse` / `val_r2`, `val_accuracy`, or `val_masked_accuracy`).
