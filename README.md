# sbol-torch

A PyTorch library for synthetic biology and biodesign automation.

Installed as `sbol-torch`, imported as `sboltorch` (commonly `import sboltorch as st`).

sboltorch pulls designs from a running [sbol-db](https://github.com/marpaia/sbol-db)
instance (or local SBOL/FASTA files), normalizes them into a single record type,
and trains transformer models against them. The input modality, tokenizer, and
training objective are all configuration — not forks of the pipeline.

## Capabilities

| Axis | Options |
|------|---------|
| **Data sources** | sbol-db REST API · local SBOL/FASTA files · synthetic generator |
| **Tokenizers** | pretrained HuggingFace (`hf`) · overlapping k-mer · IUPAC character |
| **Modalities** | `sequence` · `structure_aware` (feature boundaries) · `graph` (PyG composition transformer) |
| **Objectives** | `supervised` fine-tuning · `frozen`-backbone head · `mlm` pretraining (from-scratch & continued) |
| **Engine** | raw-PyTorch loop · early stopping · checkpointing · AMP · LR schedule · gradient accumulation |
| **Reproducibility** | one validated config per run · seeded splits · content-fingerprinted Parquet cache |

## Install

```bash
pip install sbol-torch
```

For development:

```bash
uv venv
uv pip install -e '.[dev]'
```

## Quickstart

A run is fully specified by one YAML config. From the command line:

```bash
# Materialize a corpus to the local Parquet cache (offline, reproducible).
sboltorch ingest examples/configs/finetune_expression.yaml

# Train. Resolved config, per-epoch metrics.jsonl, and best.pt land in output_dir.
sboltorch train examples/configs/finetune_expression.yaml
```

Or from Python:

```python
import sboltorch as st

config = st.RunConfig.from_yaml("examples/configs/train_graph.yaml")
metrics = st.run_training(config)
```

### Example configs

| Config | What it does |
|--------|--------------|
| [`finetune_expression.yaml`](examples/configs/finetune_expression.yaml) | Frozen DNABERT-2 backbone → regression head. |
| [`pretrain_mlm.yaml`](examples/configs/pretrain_mlm.yaml) | From-scratch masked-LM pretraining; writes a reusable backbone. |
| [`finetune_structure_aware.yaml`](examples/configs/finetune_structure_aware.yaml) | Sequence + feature-boundary markers. |
| [`train_graph.yaml`](examples/configs/train_graph.yaml) | Graph transformer over the composition graph. |

## Documentation

| Doc | Contents |
|-----|----------|
| [architecture.md](docs/architecture.md) | How the system is built — record type, plug points, engine, data flow. |
| [capabilities.md](docs/capabilities.md) | Modalities, objectives, tokenizers, metrics. |
| [configuration.md](docs/configuration.md) | Complete `RunConfig` reference. |
| [data.md](docs/data.md) | Data sources, the sbol-db client, materialization, fixtures. |
| [backbones.md](docs/backbones.md) | Choosing/loading backbones and environment constraints. |
| [extending.md](docs/extending.md) | Adding a tokenizer, encoder, task, callback, or data source. |

## Develop

```bash
uv run pytest
pre-commit run --all-files
```
