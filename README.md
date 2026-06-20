# synbio-torch

A PyTorch library for synthetic biology and biodesign machine learning.

Installed as `synbio-torch`, imported as `synbiotorch` (commonly `import synbiotorch as st`).

synbio-torch ingests biological designs and sequences from many sources — labeled
FASTA, CSV/TSV tables, GenBank, SBOL, an [sbol-db](https://github.com/marpaia/sbol-db)
instance, or a synthetic generator — normalizes them into a single record type
(`Design`), and trains transformer models against them. The input modality,
tokenizer, and training objective are all set in configuration, so trying a new
combination never means forking the pipeline. GenBank and SBOL are parsed
in-process by native [sbol-rs](https://github.com/marpaia/sbol-rs) bindings.

## Capabilities

| Axis | Options |
|------|---------|
| **Data sources** | labeled **FASTA**, CSV/TSV **tables**, **GenBank**, **SBOL** (2 & 3), the sbol-db REST API, or a **synthetic** generator; loaded in-memory or **streamed** from sharded Parquet for corpora larger than RAM |
| **Tokenizers** | pretrained HuggingFace (`hf`), overlapping k-mer, or character-level over a nucleotide **or protein** alphabet (encode + decode) |
| **Modalities** | `sequence`, `structure_aware` (feature boundaries), `graph` (PyG composition transformer) |
| **Objectives** | `supervised` / `frozen` heads, `mlm` and `causal` pretraining (from-scratch or continued) |
| **Architectures** | from-scratch or pretrained; absolute or **RoPE** positions (`gpt_neox`/`llama`/`modernbert`), SDPA/FlashAttention, configurable context length |
| **Generation** | autoregressive sampling (temperature / top-k / top-p) and design completion from a causal backbone (`synbiotorch generate`) |
| **Engine** | raw-PyTorch loop, epoch- or **step-budgeted**; AMP (`fp16`/`bf16`), gradient accumulation/clipping, gradient checkpointing, `torch.compile`; **resumable** checkpoints; early stopping; LR schedule |
| **Scaling** | token **packing**, multi-GPU **DDP** (data-parallel) via `torchrun` |
| **Tracking** | per-epoch `metrics.jsonl`, optional [Weights & Biases](https://docs.wandb.ai/) (scalars, config, lineage, model artifact) |
| **Reproducibility** | one validated config per run, seeded / hash splits, content-fingerprinted sharded Parquet cache, resumable runs |

## Install

```bash
pip install synbio-torch
```

synbio-torch ships a native extension (PyO3 bindings to the sbol-rs Rust crates).
Building from source needs a Rust toolchain (≥ 1.93); a prebuilt wheel needs none.

For development, build the extension into the venv with maturin:

```bash
uv venv
uv pip install -e '.[dev]'    # compiles the Rust extension on install
# After editing Rust under rust/, rebuild with:
uv run maturin develop
```

## Quickstart

A run is fully specified by one YAML config. From the command line:

```bash
# Materialize a corpus to the local Parquet cache (offline, reproducible).
synbiotorch ingest examples/configs/finetune_expression.yaml

# Train. Resolved config, per-epoch metrics.jsonl, and best.pt land in output_dir.
synbiotorch train examples/configs/finetune_expression.yaml

# Resume an interrupted run from its rolling checkpoint (needs checkpoint_every_n_steps).
synbiotorch train examples/configs/pretrain_mlm.yaml --resume runs/pretrain_mlm/last.pt

# Generate from a trained causal backbone — point model.backbone at the run's
# backbone/ (with from_scratch: false), then complete a design from a prompt.
synbiotorch generate my_causal_run.yaml --prompt ATGCGT --max-new-tokens 200 --temperature 0.8
```

Train multi-GPU with `torchrun` and `train.distributed.strategy: ddp`:

```bash
torchrun --nproc_per_node=<gpus> -m synbiotorch.cli train examples/configs/pretrain_causal_long.yaml
```

Or from Python:

```python
import synbiotorch as st

config = st.RunConfig.from_yaml("examples/configs/train_graph.yaml")
metrics = st.run_training(config)
```

### Example configs

| Config | What it does |
|--------|--------------|
| [`finetune_expression.yaml`](https://github.com/marpaia/synbio-torch/blob/master/examples/configs/finetune_expression.yaml) | Frozen DNABERT-2 backbone feeding a regression head. |
| [`pretrain_mlm.yaml`](https://github.com/marpaia/synbio-torch/blob/master/examples/configs/pretrain_mlm.yaml) | From-scratch masked-LM pretraining; writes a reusable backbone. |
| [`finetune_structure_aware.yaml`](https://github.com/marpaia/synbio-torch/blob/master/examples/configs/finetune_structure_aware.yaml) | Sequence + feature-boundary markers. |
| [`train_graph.yaml`](https://github.com/marpaia/synbio-torch/blob/master/examples/configs/train_graph.yaml) | Graph transformer over the composition graph. |
| [`pretrain_causal_long.yaml`](https://github.com/marpaia/synbio-torch/blob/master/examples/configs/pretrain_causal_long.yaml) | Long-context causal pretraining: RoPE decoder, SDPA, streamed + packed corpus. |
| [`finetune_protein.yaml`](https://github.com/marpaia/synbio-torch/blob/master/examples/configs/finetune_protein.yaml) | Protein regression from a labeled CSV table with the protein char tokenizer. |
| [`benchmark_dna_classification.yaml`](https://github.com/marpaia/synbio-torch/blob/master/examples/configs/benchmark_dna_classification.yaml) | Genomics-ML benchmark shape: a labeled table fed to a pretrained DNA backbone for classification. |
| [`ingest_genbank.yaml`](https://github.com/marpaia/synbio-torch/blob/master/examples/configs/ingest_genbank.yaml) | Import GenBank to the Parquet cache via the native binding. |

## Experiment tracking

The two synthetic-data configs ([`train_graph.yaml`](https://github.com/marpaia/synbio-torch/blob/master/examples/configs/train_graph.yaml)
and [`finetune_structure_aware.yaml`](https://github.com/marpaia/synbio-torch/blob/master/examples/configs/finetune_structure_aware.yaml))
ship with [Weights & Biases](https://docs.wandb.ai/) enabled. Set `WANDB_API_KEY`
in a `.env` at the repo root and run both:

```bash
python examples/run_wandb_examples.py
```

Each run logs per-step loss and learning rate, per-epoch train/val metrics, the
resolved config, the corpus fingerprint and split sizes as lineage, and the best
checkpoint as a model artifact.

| Graph transformer | Structure-aware sequence |
|-------------------|--------------------------|
| ![train_graph W&B run](https://raw.githubusercontent.com/marpaia/synbio-torch/master/docs/images/wandb_train_graph.png) | ![structure_aware W&B run](https://raw.githubusercontent.com/marpaia/synbio-torch/master/docs/images/wandb_structure_aware.png) |

## Documentation

| Doc | Contents |
|-----|----------|
| [architecture.md](https://github.com/marpaia/synbio-torch/blob/master/docs/architecture.md) | How the system is built — record type, plug points, engine, data flow. |
| [capabilities.md](https://github.com/marpaia/synbio-torch/blob/master/docs/capabilities.md) | Modalities, objectives, tokenizers, metrics. |
| [configuration.md](https://github.com/marpaia/synbio-torch/blob/master/docs/configuration.md) | Complete `RunConfig` reference. |
| [data.md](https://github.com/marpaia/synbio-torch/blob/master/docs/data.md) | Data sources, native parsing, materialization, fixtures. |
| [backbones.md](https://github.com/marpaia/synbio-torch/blob/master/docs/backbones.md) | Choosing/loading backbones and environment constraints. |
| [extending.md](https://github.com/marpaia/synbio-torch/blob/master/docs/extending.md) | Adding a tokenizer, encoder, task, callback, or data source. |

Release history is in [CHANGELOG.md](https://github.com/marpaia/synbio-torch/blob/master/CHANGELOG.md).

## Develop

```bash
uv run maturin develop      # rebuild the Rust extension after editing rust/
uv run pytest
pre-commit run --all-files
```
