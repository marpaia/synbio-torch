"""Stage-3 generative path: causal-LM objective, decoder model, and sampling.

The discipline: the collator's next-token shift is checked directly; a tiny causal
LM is shown to actually learn next-token prediction; and a model trained on a
periodic motif is shown to *generate* that motif back — generation works end to
end, not merely without error.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader

from sboltorch.config import (
    ArchConfig,
    CorpusConfig,
    EncoderConfig,
    ModelConfig,
    PackingConfig,
    RunConfig,
    SplitConfig,
    TaskConfig,
    TokenizerConfig,
    TrainConfig,
)
from sboltorch.datasets.causal_collator import CausalCollator
from sboltorch.datasets.mlm_collator import IGNORE_INDEX
from sboltorch.datasets.packing import PackedDataset
from sboltorch.encoders.base import ModelInput
from sboltorch.engine.trainer import Callback, Trainer
from sboltorch.generate import generate, generate_sequence
from sboltorch.models import build_model
from sboltorch.pipeline import run_training
from sboltorch.reproducibility import set_seed
from sboltorch.tasks.base import build_task
from sboltorch.tasks.causal import CausalLMTask
from sboltorch.tokenize.char import CharTokenizer
from sboltorch.tokenize.kmer import KmerTokenizer
from sboltorch.types import Alphabet, SbolObject, SbolSequence

CPU = torch.device("cpu")


def _gpt2_arch(max_positions: int = 64) -> ArchConfig:
    return ArchConfig(
        model_type="gpt2",
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=96,
        max_position_embeddings=max_positions,
    )


def _motif_objects(n: int, motif: str) -> list[SbolObject]:
    return [
        SbolObject(
            iri=f"https://ex/c{i}",
            sbol_class="http://sbols.org/v3#Sequence",
            sequence=SbolSequence(elements=motif, alphabet=Alphabet.DNA),
        )
        for i in range(n)
    ]


class _History(Callback):
    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        self.rows.append(dict(metrics))


# --- collator ---------------------------------------------------------------


def test_causal_collator_shifts_targets():
    collator = CausalCollator(pad_token_id=0)
    batch = [ModelInput([5, 6, 7], [1, 1, 1]), ModelInput([8, 9], [1, 1])]
    out = collator(batch)
    # Row 0: targets are the next token, last position ignored.
    assert out["labels"][0].tolist() == [6, 7, IGNORE_INDEX]
    # Row 1 is padded to width 3; predicting the pad token is ignored too.
    assert out["labels"][1].tolist() == [9, IGNORE_INDEX, IGNORE_INDEX]


def test_build_task_causal():
    assert isinstance(build_task(TaskConfig(kind="causal")), CausalLMTask)


# --- decode -----------------------------------------------------------------


def test_kmer_decode_roundtrips():
    tok = KmerTokenizer(k=3, max_length=512)
    seq = "ACGTACGTACGTAC"
    assert tok.decode(tok.tokenize_content(seq)) == seq


def test_char_decode_roundtrips():
    tok = CharTokenizer(max_length=512)
    seq = "ACGTNRYACGT"
    assert tok.decode(tok.tokenize_content(seq)) == seq


# --- learning ---------------------------------------------------------------


def test_causal_lm_learns_next_token():
    set_seed(0)
    tok = KmerTokenizer(k=3, max_length=512)
    objs = _motif_objects(120, "ACGT" * 30) + _motif_objects(120, "GGCC" * 30)
    collator = CausalCollator(tok.pad_token_id)
    train = DataLoader(PackedDataset(objs, tok, block_size=32, which="train"), batch_size=8, collate_fn=collator)
    val = DataLoader(PackedDataset(objs, tok, block_size=32, which="val"), batch_size=8, collate_fn=collator)
    model = build_model(
        ModelConfig(from_scratch=True, hidden_size=48, arch=_gpt2_arch()),
        TaskConfig(kind="causal"),
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_token_id,
    )
    history = _History()
    cfg = TrainConfig(batch_size=8, lr=5e-3, amp=False, max_steps=80, eval_every_n_steps=20)
    Trainer(model, CausalLMTask(), cfg, callbacks=[history], device=CPU).fit(train, val)

    assert len(history.rows) >= 2
    assert history.rows[-1]["val_loss"] < history.rows[0]["val_loss"]
    assert history.rows[-1]["val_next_token_accuracy"] > history.rows[0]["val_next_token_accuracy"]


# --- generation -------------------------------------------------------------


def test_generate_shapes_and_determinism():
    set_seed(0)
    tok = CharTokenizer(max_length=128)
    model = build_model(
        ModelConfig(from_scratch=True, hidden_size=32, arch=_gpt2_arch()),
        TaskConfig(kind="causal"),
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_token_id,
    )
    prompt = tok.tokenize_content("ACGT")
    out = generate(model, prompt, max_new_tokens=10)
    assert len(out) == len(prompt) + 10
    assert all(0 <= i < tok.vocab_size for i in out)
    # Greedy is deterministic; seeded sampling is reproducible.
    assert generate(model, prompt, max_new_tokens=10, temperature=0.0) == generate(
        model, prompt, max_new_tokens=10, temperature=0.0
    )
    assert generate(model, prompt, max_new_tokens=10, temperature=1.0, seed=7) == generate(
        model, prompt, max_new_tokens=10, temperature=1.0, seed=7
    )


def test_trained_causal_lm_generates_its_motif():
    set_seed(0)
    tok = CharTokenizer(max_length=256)
    # A single ACGT cycle: each base determines the next, so a trained LM should
    # regenerate the cycle under greedy decoding.
    objs = _motif_objects(200, "ACGT" * 40)
    collator = CausalCollator(tok.pad_token_id)
    train = DataLoader(PackedDataset(objs, tok, block_size=16, which=None), batch_size=16, collate_fn=collator)
    model = build_model(
        ModelConfig(from_scratch=True, hidden_size=32, arch=_gpt2_arch()),
        TaskConfig(kind="causal"),
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_token_id,
    )
    cfg = TrainConfig(batch_size=16, lr=5e-3, amp=False, max_steps=200)
    Trainer(model, CausalLMTask(), cfg, device=CPU).fit(train)

    completion = generate_sequence(model, tok, "ACGT", max_new_tokens=16, temperature=0.0, max_context=64)
    assert "ACGTACGT" in completion


# --- pipeline ---------------------------------------------------------------


def test_pipeline_causal_streaming_packed_runs(tmp_path):
    config = RunConfig(
        streaming=True,
        output_dir=str(tmp_path / "run"),
        corpus=CorpusConfig(source="synthetic", n=96, shard_size=16, cache_dir=str(tmp_path / "cache")),
        tokenizer=TokenizerConfig(kind="kmer", k=3, max_length=64),
        encoder=EncoderConfig(kind="sequence"),
        model=ModelConfig(from_scratch=True, hidden_size=32, arch=_gpt2_arch()),
        task=TaskConfig(kind="causal"),
        packing=PackingConfig(enabled=True, block_size=32),
        splits=SplitConfig(strategy="hash"),
        train=TrainConfig(batch_size=8, amp=False, max_steps=10, eval_every_n_steps=5),
    )
    metrics = run_training(config)
    assert "val_loss" in metrics and math.isfinite(metrics["val_loss"])
    assert (tmp_path / "run" / "backbone").exists()


def test_packing_rejects_supervised(tmp_path):
    config = RunConfig(
        streaming=True,
        output_dir=str(tmp_path / "run"),
        corpus=CorpusConfig(source="synthetic", n=32, shard_size=16, cache_dir=str(tmp_path / "cache")),
        tokenizer=TokenizerConfig(kind="kmer", k=3, max_length=64),
        model=ModelConfig(from_scratch=True, hidden_size=32),
        task=TaskConfig(kind="supervised"),
        packing=PackingConfig(enabled=True, block_size=32),
        splits=SplitConfig(strategy="hash"),
        train=TrainConfig(batch_size=8, amp=False, max_steps=5),
    )
    with pytest.raises(Exception):
        run_training(config)
