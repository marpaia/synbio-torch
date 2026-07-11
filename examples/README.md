# Examples

Runnable example configs and a Python quickstart for synbio-torch.

## Quickstart (offline)

Train a graph transformer on synthetic SBOL data — no external services or
downloads, runs in seconds:

```bash
python examples/quickstart.py
```

## Example configs

Run any config with the CLI:

```bash
synbiotorch train examples/configs/<name>.yaml
```

| Config | Task | Data | Prerequisites |
|--------|------|------|---------------|
| [`train_graph.yaml`](configs/train_graph.yaml) | Graph transformer regression | synthetic | `WANDB_API_KEY` (or set `wandb.enabled: false`) |
| [`finetune_structure_aware.yaml`](configs/finetune_structure_aware.yaml) | Structure-aware regression (from scratch) | synthetic | `WANDB_API_KEY` (or set `wandb.enabled: false`) |
| [`pretrain_mlm.yaml`](configs/pretrain_mlm.yaml) | From-scratch MLM pretraining | sbol-db | a running sbol-db at `base_url` |
| [`finetune_expression.yaml`](configs/finetune_expression.yaml) | Frozen DNABERT-2 → regression | sbol-db | a running sbol-db, plus DNABERT-2 (Linux/GPU — see [backbones](../docs/backbones.md)) |
| [`finetune_protein.yaml`](configs/finetune_protein.yaml) | Protein regression (from scratch) | [`data/protein_activity.csv`](data/protein_activity.csv) | none — runs offline |
| [`benchmark_dna_classification.yaml`](configs/benchmark_dna_classification.yaml) | DNA sequence classification | [`data/promoter_strength.csv`](data/promoter_strength.csv) | DNABERT-2 from the hub (network) |
| [`ingest_genbank.yaml`](configs/ingest_genbank.yaml) | Import GenBank to the Parquet cache | [`data/demo_tu.gb`](data/demo_tu.gb) | none — runs offline |

The `synthetic` and CSV configs are the quickest way to see the full pipeline end
to end. To point a config at your own data, change the `corpus` section (see
[configuration](../docs/configuration.md) and [data sources](../docs/data.md)).

## GenBank and SBOL

GenBank and SBOL are parsed in-process by the native sbol-rs binding — no external
tool. Import a GenBank file to the Parquet cache with:

```bash
synbiotorch ingest examples/configs/ingest_genbank.yaml
```

Point `corpus.path` at a directory of `.gb`/`.gbk` (or `.ttl`/`.xml` for SBOL)
files to build a real corpus; `corpus.namespace` roots the imported identities.

## Höllerer RBS demonstration (paper Table 1)

The configuration sweep on the Höllerer et al. *E. coli* RBS dataset. All three
variants share one corpus, task, and the published held-out split; they differ
only in the tokenizer and model blocks.

```bash
# 1. download the SAPIENs arrays (pinned commit) and build the split CSV
python examples/prepare_hollerer_rbs.py
# 2. build a Triton-free DNABERT-2 copy (needed on Apple Silicon / CPU)
python examples/prepare_dnabert2.py
# 3. train the three sweep variants
synbiotorch train examples/configs/hollerer_scratch_char.yaml
synbiotorch train examples/configs/hollerer_scratch_kmer.yaml
synbiotorch train examples/configs/hollerer_finetune_dnabert2.yaml
# 4. score on the fixed test split, archive metrics + predictions, write the figure
env -u VIRTUAL_ENV uv run --with matplotlib python examples/eval_hollerer.py \
    --out ../research/synbio-torch/figures/rbs_scatter.pdf
```

Step 4 scores each run on the exact SAPIENs test partition (`split == "test"`,
27,654 variants) and writes:

- [`hollerer_test_metrics.json`](hollerer_test_metrics.json) — the held-out test
  R², MAE, and bootstrap CI for each variant (the source of the paper's Table 1),
- `hollerer_predictions/<run>.npz` — per-variant measured and predicted values.

These held-out **test** metrics are distinct from the **validation** metrics in
each run's `runs/<run>/final_metrics.json`.

## Kosuri composability demonstration (design-native encoders)

An encoder ablation on the Kosuri et al. 2013 promoter × RBS composability
library, where each construct is a real composition of two annotated parts. The
constructs are ingested as SBOL 3 through the native binding, and the same corpus,
task, and split are held fixed while only the encoder varies — flat `sequence`,
`structure_aware`, or `graph` — so the design-native modalities are exercised on
real data. Two splits run for each encoder: `random` (in-distribution) and
`partout`, a held-out-parts split whose test constructs use a promoter or RBS
never seen in training, which measures generalization to novel part combinations.

The three PNAS Supporting Information files are not open-access and are not
redistributed. Download them once from
[PMC3752251](https://pmc.ncbi.nlm.nih.gov/articles/PMC3752251/) (Supplementary
Materials) into `data/kosuri/` (`sd01.xls`, `sd02.xls`, `sd03.xls`), then:

```bash
# 1. build the SBOL 3 corpus (log10 protein label + both split annotations)
env -u VIRTUAL_ENV uv run --with pandas --with xlrd python examples/prepare_kosuri.py
# 2. train the six-cell sweep (3 encoders x 2 splits)
for enc in seq structure graph; do for split in random partout; do
  synbiotorch train examples/configs/kosuri_${enc}_${split}.yaml
done; done
# 3. score each run on its test split, archive metrics + predictions
env -u VIRTUAL_ENV uv run python examples/eval_kosuri.py
```

Step 3 writes [`kosuri_test_metrics.json`](kosuri_test_metrics.json) (test R²,
MAE, and bootstrap CI per encoder × split) and `kosuri_predictions/<run>.npz`.

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
