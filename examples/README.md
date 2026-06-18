# Examples

Runnable example configs and a Python quickstart for sbol-torch.

## Quickstart (offline)

Train a graph transformer on synthetic SBOL data — no external services or
downloads, runs in seconds:

```bash
python examples/quickstart.py
```

## Example configs

Run any config with the CLI:

```bash
sboltorch train examples/configs/<name>.yaml
```

| Config | Task | Data | Prerequisites |
|--------|------|------|---------------|
| [`train_graph.yaml`](configs/train_graph.yaml) | Graph transformer regression | synthetic | `WANDB_API_KEY` (or set `wandb.enabled: false`) |
| [`finetune_structure_aware.yaml`](configs/finetune_structure_aware.yaml) | Structure-aware regression (from scratch) | synthetic | `WANDB_API_KEY` (or set `wandb.enabled: false`) |
| [`pretrain_mlm.yaml`](configs/pretrain_mlm.yaml) | From-scratch MLM pretraining | sbol-db | a running sbol-db at `base_url` |
| [`finetune_expression.yaml`](configs/finetune_expression.yaml) | Frozen DNABERT-2 → regression | sbol-db | a running sbol-db, plus DNABERT-2 (Linux/GPU — see [backbones](../docs/backbones.md)) |
| [`ingest_local_sbol.yaml`](configs/ingest_local_sbol.yaml) | Structure-aware on normalized SBOL3 | local files | the [`sbol`](https://github.com/marpaia/sbol-rs) CLI (`SBOL_BIN`) |

The two `synthetic`-source configs are the quickest way to see the full pipeline
end to end. To point a config at your own data, change the `corpus` section (see
[configuration](../docs/configuration.md) and [data sources](../docs/data.md)).

## Normalizing other formats to SBOL3

`normalize_and_ingest.py` bridges a GenBank file into a materialized,
structure-aware corpus: it converts [`data/demo_tu.gb`](data/demo_tu.gb) to SBOL3
with the [`sbol`](https://github.com/marpaia/sbol-rs) CLI, then parses the
sequence, features, and composition graph back out. Install the CLI with Cargo:

```bash
cargo install sbol-cli
```

```bash
python examples/normalize_and_ingest.py
```

The `sbol` binary is located via the `SBOL_BIN` environment variable or an
`SBOL_BIN=` line in the repo-root `.env` (the same file that holds
`WANDB_API_KEY`); `cargo install` puts it on `PATH`. The conversion writes
`data/normalized/`, which
[`ingest_local_sbol.yaml`](configs/ingest_local_sbol.yaml) then reads:

```bash
sboltorch train examples/configs/ingest_local_sbol.yaml
```

## Weights & Biases

The two synthetic configs have `wandb.enabled: true`. Put `WANDB_API_KEY` in a
`.env` at the repo root and run both:

```bash
python examples/run_wandb_examples.py
```

The runner loads the key from `.env`, trains each config online, and prints the
project workspace URL. To run without an account, set `wandb.enabled: false` (or
`wandb.mode: offline`) in the config.

To refresh the screenshots in the top-level README, capture these panels from
each run and save them as `docs/images/wandb_train_graph.png` and
`docs/images/wandb_structure_aware.png`:

- `train/step_loss` (per-step training loss)
- `val/loss` and `val/r2` (per-epoch validation)
- optionally a `gradients/*` histogram (logged by `watch_model`)
