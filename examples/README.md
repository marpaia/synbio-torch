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
| [`train_graph.yaml`](configs/train_graph.yaml) | Graph transformer regression | synthetic | none — runs out of the box |
| [`finetune_structure_aware.yaml`](configs/finetune_structure_aware.yaml) | Structure-aware regression (from scratch) | synthetic | none — runs out of the box |
| [`pretrain_mlm.yaml`](configs/pretrain_mlm.yaml) | From-scratch MLM pretraining | sbol-db | a running sbol-db at `base_url` |
| [`finetune_expression.yaml`](configs/finetune_expression.yaml) | Frozen DNABERT-2 → regression | sbol-db | a running sbol-db, plus DNABERT-2 (Linux/GPU — see [backbones](../docs/backbones.md)) |

The two `synthetic`-source configs are the quickest way to see the full pipeline
end to end. To point a config at your own data, change the `corpus` section (see
[configuration](../docs/configuration.md) and [data sources](../docs/data.md)).
