# Capabilities

synbio-torch trains transformer models on SBOL data. The input modality, training
objective, data source, and tokenizer are independent axes, each picked in
configuration. Changing one reuses the same code path rather than branching it.

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
  so the model sees SBOL structure alongside sequence. Use it with a
  `from_scratch` model, or a pretrained backbone whose embeddings you've resized.
- **Graph** turns each object's neighborhood into a graph: nodes carry a
  `(record_class, role, identity)` feature triple, edges carry a predicate type,
  edges are bidirectional, and a global mean pool feeds the task head.

All three produce batches consumed by one training engine through a
`BatchAdapter`, so the loop is modality-agnostic (see [extending.md](extending.md)).

## Objectives (`task.kind`)

| Objective | Head / loss | Metrics | Label |
|-----------|-------------|---------|-------|
| `frozen` | regression or classification head; backbone frozen | `val_mae`/`val_mse`/`val_r2` or `val_accuracy` | required |
| `supervised` | same, fine-tuned end to end | same | required |
| `mlm` | tied LM head; masked cross-entropy | `val_loss` (= masked CE), `val_masked_accuracy` | none (self-supervised) |
| `causal` | decoder LM head; next-token cross-entropy | `val_loss` (perplexity = `exp(val_loss)`), `val_next_token_accuracy` | none (self-supervised) |

- **Supervised regression** supports `target_transform: log1p` for expression/
  fitness-style targets; metrics are reported back in the original space.
- **Classification** requires `task.num_classes`.
- **MLM** masks ~`mlm_probability` of content tokens (80% `<mask>`, 10% random,
  10% unchanged), never masking special tokens.
- **Causal** trains a decoder (e.g. `model.arch.model_type: gpt2`, or a RoPE type
  like `gpt_neox` for long context) on next-token prediction. Pair it with
  `packing` to train on fixed-length blocks. The run writes its model to
  `<output_dir>/backbone/`; to generate, point `model.backbone` at that directory
  with `from_scratch: false` and run `synbiotorch generate <config> --prompt <seq>`,
  which completes a design from a prefix (greedy with `--temperature 0`, or sampled
  with `--temperature`/`--top-k`/`--top-p`). The same is available in Python as
  `st.generate_sequence(model, tokenizer, prompt, max_new_tokens=...)`.

## Pretrain, then fine-tune

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
| `char` | Character-level over a nucleotide (IUPAC) or protein alphabet (`tokenizer.alphabet`). |

All expose the same protocol (`vocab_size`, `pad_token_id`, `mask_token_id`,
`special_token_ids`, `tokenize_content`, `encode`), so tokenizers, encoders, and
objectives mix freely.

## Metrics

- Regression: MAE, MSE, R² (`val_mae`, `val_mse`, `val_r2`).
- Classification: accuracy (`val_accuracy`).
- MLM: masked cross-entropy as `val_loss` (perplexity is `exp(val_loss)`) and
  `val_masked_accuracy`.
- Causal: next-token cross-entropy as `val_loss` (perplexity is `exp(val_loss)`)
  and `val_next_token_accuracy`.

Metrics are printed and appended to `<output_dir>/metrics.jsonl` (per epoch, or
per `eval_every_n_steps` for step-budgeted runs), each row tagged with the global
step; the best checkpoint (by the task's primary metric) is saved to `best.pt`,
and a rolling resumable `last.pt` if `checkpoint_every_n_steps` is set.

## What the test suite verifies

`tests/test_learning.py` trains each capability on a learnable signal and asserts
that the loss drops and the model generalizes, not just that the code runs:

| Capability | Asserted |
|------------|----------|
| Supervised sequence (from scratch) | val_loss falls, `val_r2 > 0.5` |
| MLM (from scratch) | val_loss falls, masked accuracy rises |
| Structure-aware | val_loss falls, `val_r2 > 0.5` |
| Graph transformer | val_loss falls, `val_r2 > 0.5` |

Continued pretraining and pretrained-backbone loading are exercised against a
real hub model in the data/model tests. The features added on top of these are
held to the same "prove it works, not just runs" bar in their own suites:
resumable training reproduces an uninterrupted run (`test_resume.py`); a streaming
+ packed MLM run learns (`test_streaming.py`); a causal LM learns next-token
prediction and a model trained on a motif regenerates it (`test_causal.py`); a
RoPE decoder learns and runs past an absolute model's context limit
(`test_long_context.py`); and 2-rank DDP matches single-process training
bit-for-bit (`test_distributed.py`).
