# Backbones & environment

The backbone is any HuggingFace encoder, or one built from scratch. The `hf`
tokenizer is a generic `AutoTokenizer` adapter, so any model that accepts a raw
sequence string works by pointing `model.backbone` and `tokenizer.model_name` at
the same id.

## Choosing a backbone

| Goal | `model` config |
|------|----------------|
| Fine-tune a pretrained DNA model | `backbone: <hub-id>`, `from_scratch: false` |
| Train a head over a frozen backbone | as above, with `task.kind: frozen` |
| Pretrain / train from scratch | `from_scratch: true` + `model.arch` (sized to the tokenizer vocab) |
| Reuse a model you pretrained | `backbone: <output_dir>/backbone` (a local path) |

`model.backbone` accepts either a hub id or a local directory. A masked-LM run
writes its encoder to `<output_dir>/backbone/`, which a later supervised run loads
directly.

## transformers version

sbol-torch pins `transformers` to the 4.x line. The 5.x line changes the ESM and
custom modeling code that the pretrained DNA backbones rely on, so it is not
compatible with them.

## Known backbone constraints

- **DNABERT-2 (`zhihan1996/DNABERT-2-117M`)** — its remote modeling code requires
  `triton`, which has no macOS wheels, so DNABERT-2 runs on Linux/GPU only.
  `einops` (also required by that code) is a declared dependency. The DNABERT-2
  tokenizer loads on any platform; only the model weights need Linux/GPU.
- **Nucleotide Transformer v2** — the tokenizer loads everywhere; the checkpoint
  needs a transformers version whose ESM implementation matches its gated-MLP
  shapes. Verify the pairing before relying on it.

For local development on CPU/macOS, use a `from_scratch` model with the `kmer` or
`char` tokenizer, or a small standard encoder; run bio-specific backbones like
DNABERT-2 on a Linux/GPU host.

## Long context & modern attention

For long sequences, pick a RoPE architecture and SDPA attention in `model.arch`:

| Goal | `model.arch` |
|------|--------------|
| Long-context causal pretraining | `model_type: gpt_neox` (or `llama`), large `max_position_embeddings` |
| Long-context MLM | `model_type: modernbert` (RoPE encoder with local/global attention) |
| Fused/fast attention | `attn_implementation: sdpa` (default) — FlashAttention kernels on CUDA, and works on CPU |

RoPE architectures carry no absolute position-embedding table, so they run on
sequences longer than `max_position_embeddings` (an absolute-position model like
`bert`/`gpt2` is hard-capped at it). Pair a causal RoPE model with `packing` and
`streaming` for block pretraining — see
[`pretrain_causal_long.yaml`](../examples/configs/pretrain_causal_long.yaml).

State-space / convolutional long-context models (e.g. Mamba, the Evo/Hyena route)
are selectable via `model_type` when the architecture and its kernels are
installed; they are not a bundled dependency.

## Structure-aware backbones

The structure-aware encoder adds feature-boundary markers to the vocabulary, so
its `output_spec.vocab_size` exceeds a base tokenizer's. A `from_scratch` model is
sized to that vocabulary automatically. To use a pretrained backbone with the
structure-aware encoder, resize its token embeddings to the encoder's vocab size
first.
