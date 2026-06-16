# Capabilities

sboltorch trains transformer models on SBOL data across three input modalities
and three training objectives, over three data sources, with three tokenizers.
Each axis is a configuration choice, not a separate code path.

## Input modalities (`encoder.kind`)

| Modality | Consumes | Model | Config |
|----------|----------|-------|--------|
| `sequence` | raw sequence elements | pretrained or from-scratch encoder + pooling + head | [finetune_expression.yaml](../examples/configs/finetune_expression.yaml) |
| `structure_aware` | sequence + feature boundaries (role, orientation) | same, over an extended vocabulary | [finetune_structure_aware.yaml](../examples/configs/finetune_structure_aware.yaml) |
| `graph` | the SBOL composition graph | PyG graph transformer (`TransformerConv`) | [train_graph.yaml](../examples/configs/train_graph.yaml) |

- **Sequence** tokenizes the object's elements directly.
- **Structure-aware** wraps each annotated feature span with role-keyed boundary
  markers (e.g. `[promoter] … [/promoter]`) and a reverse-complement marker,
  injected inline as tokens. The markers extend the base tokenizer's vocabulary,
  so the model sees SBOL structure alongside sequence. Pairs naturally with a
  `from_scratch` model (or a pretrained backbone with resized embeddings).
- **Graph** turns each object's neighborhood into a graph: nodes carry a
  `(sbol_class, role, identity)` feature triple, edges carry a predicate type,
  edges are bidirectional, and a global mean pool feeds the task head.

All three produce batches consumed by one training engine through a
`BatchAdapter`, so the loop is modality-agnostic (see [extending.md](extending.md)).

## Objectives (`task.kind`)

| Objective | Head / loss | Metrics | Label |
|-----------|-------------|---------|-------|
| `frozen` | regression or classification head; backbone frozen | `val_mae`/`val_mse`/`val_r2` or `val_accuracy` | required |
| `supervised` | same, fine-tuned end to end | same | required |
| `mlm` | tied LM head; masked cross-entropy | `val_loss` (= masked CE), `val_masked_accuracy` | none (self-supervised) |

- **Supervised regression** supports `target_transform: log1p` for expression/
  fitness-style targets; metrics are reported back in the original space.
- **Classification** requires `task.num_classes`.
- **MLM** masks ~`mlm_probability` of content tokens (80% `<mask>`, 10% random,
  10% unchanged), never masking special tokens.

## Pretrain → fine-tune

An `mlm` run writes its trained encoder to `<output_dir>/backbone/` in
HuggingFace format. A later `supervised`/`frozen` run loads it by setting
`model.backbone` to that directory — the masked-LM pretraining and the
downstream task share one backbone.

MLM supports two modes via `model.from_scratch`:

- `true` — build a fresh architecture from `model.arch` + the tokenizer vocab
  (pretrain a DNA LM on the SBOL corpus with the k-mer/char tokenizer).
- `false` — continued pretraining of a pretrained `model.backbone`.

## Tokenizers (`tokenizer.kind`)

| Tokenizer | Description |
|-----------|-------------|
| `hf` | Any pretrained HuggingFace `AutoTokenizer` (DNABERT-2, Nucleotide Transformer, …), wrapped behind the library's tokenizer protocol. |
| `kmer` | Overlapping k-mers over `{A,C,G,T}`; ambiguous bases map to `<unk>`. |
| `char` | IUPAC character-level. |

All expose the same protocol (`vocab_size`, `pad_token_id`, `mask_token_id`,
`special_token_ids`, `tokenize_content`, `encode`), so tokenizers, encoders, and
objectives mix freely.

## Metrics

- Regression: MAE, MSE, R² (`val_mae`, `val_mse`, `val_r2`).
- Classification: accuracy (`val_accuracy`).
- MLM: masked cross-entropy as `val_loss` (perplexity is `exp(val_loss)`) and
  `val_masked_accuracy`.

Per-epoch metrics are printed and appended to `<output_dir>/metrics.jsonl`; the
best checkpoint (by the task's primary metric) is saved to `best.pt`.

## What the test suite verifies

`tests/test_learning.py` trains each capability on a learnable signal and asserts
the loss drops and the model generalizes — not merely that the pipeline runs:

| Capability | Asserted |
|------------|----------|
| Supervised sequence (from scratch) | val_loss ↓, `val_r2 > 0.5` |
| MLM (from scratch) | val_loss ↓, masked accuracy ↑ |
| Structure-aware | val_loss ↓, `val_r2 > 0.5` |
| Graph transformer | val_loss ↓, `val_r2 > 0.5` |

Continued pretraining and pretrained-backbone loading are exercised against a
real hub model in the data/model tests.
