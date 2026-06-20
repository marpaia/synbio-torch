"""Masked-LM pretraining: collator masking, task metrics, end-to-end from-scratch run."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from synbiotorch.config import ArchConfig, ModelConfig, TaskConfig, TrainConfig
from synbiotorch.datasets.dataset import EncodedDataset
from synbiotorch.datasets.mlm_collator import IGNORE_INDEX, MlmCollator
from synbiotorch.encoders.sequence import SequenceEncoder
from synbiotorch.engine.callbacks import MetricLogger
from synbiotorch.engine.trainer import Trainer
from synbiotorch.models import build_model
from synbiotorch.tasks.mlm import MlmTask
from synbiotorch.tokenize.kmer import KmerTokenizer
from synbiotorch.types import Alphabet, Design, Sequence


def _objects(n: int = 32, length: int = 40):
    bases = "ACGT"
    out = []
    for i in range(n):
        seq = "".join(bases[(i + j) % 4] for j in range(length))
        out.append(
            Design(
                iri=f"s{i}",
                record_class="http://sbols.org/v3#Sequence",
                sequence=Sequence(elements=seq, alphabet=Alphabet.DNA),
            )
        )
    return out


def _batch(tokenizer):
    encoder = SequenceEncoder(tokenizer)
    return [encoder.encode(o) for o in _objects(4)]


def test_masking_is_deterministic_with_seed():
    tok = KmerTokenizer(k=3, max_length=64)
    batch = _batch(tok)
    a = MlmCollator(tok, seed=123)(list(batch))
    b = MlmCollator(tok, seed=123)(list(batch))
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert torch.equal(a["labels"], b["labels"])


def test_special_tokens_are_never_masked():
    from synbiotorch.datasets.dataset import pad_token_batch

    tok = KmerTokenizer(k=3, max_length=64)
    batch = _batch(tok)
    original_ids, _ = pad_token_batch(batch, tok.pad_token_id)
    out = MlmCollator(tok, seed=7)(batch)
    special = torch.tensor(sorted(tok.special_token_ids))
    # A position whose ORIGINAL token was special (cls/sep/pad) is never selected,
    # so its label stays ignored.
    was_special = torch.isin(original_ids, special)
    assert torch.all(out["labels"][was_special] == IGNORE_INDEX)


def test_some_positions_are_masked_and_labelled():
    tok = KmerTokenizer(k=3, max_length=64)
    out = MlmCollator(tok, seed=1, mlm_probability=0.15)(_batch(tok))
    scored = (out["labels"] != IGNORE_INDEX).sum().item()
    assert scored > 0
    # Every row with content gets at least one target (no all-ignored row).
    per_row = (out["labels"] != IGNORE_INDEX).sum(dim=1)
    assert torch.all(per_row >= 1)


def test_mlm_task_loss_and_metrics():
    task = MlmTask()
    logits = torch.randn(2, 5, 10)
    labels = torch.tensor([[IGNORE_INDEX, 3, IGNORE_INDEX, 1, IGNORE_INDEX], [2, IGNORE_INDEX, 4, IGNORE_INDEX, 0]])
    loss = task.loss(logits, labels)
    assert torch.isfinite(loss)
    preds = task.predict(logits)
    assert preds.shape == (2, 5)
    metrics = task.epoch_metrics(preds.numpy().ravel(), labels.numpy().ravel())
    assert 0.0 <= metrics["masked_accuracy"] <= 1.0


def test_from_scratch_mlm_trains_end_to_end(tmp_path):
    tok = KmerTokenizer(k=3, max_length=64)
    encoder = SequenceEncoder(tok)
    model_config = ModelConfig(
        from_scratch=True,
        hidden_size=48,
        arch=ArchConfig(num_hidden_layers=2, num_attention_heads=4, intermediate_size=96, max_position_embeddings=64),
    )
    task_config = TaskConfig(kind="mlm")
    model = build_model(model_config, task_config, vocab_size=tok.vocab_size, pad_token_id=tok.pad_token_id)
    task = MlmTask()

    collator = MlmCollator(tok, mlm_probability=0.15)
    train_loader = DataLoader(EncodedDataset(_objects(32), encoder), batch_size=8, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(EncodedDataset(_objects(8), encoder), batch_size=8, collate_fn=collator)

    trainer = Trainer(
        model,
        task,
        TrainConfig(epochs=3, lr=5e-3, amp=False),
        callbacks=[MetricLogger(tmp_path)],
        device=torch.device("cpu"),
    )
    metrics = trainer.fit(train_loader, val_loader)
    assert "val_loss" in metrics and np.isfinite(metrics["val_loss"])
    assert "val_masked_accuracy" in metrics
    # Pretrained MLM saves an HF backbone for downstream fine-tuning.
    model.save_pretrained(tmp_path / "backbone")
    assert (tmp_path / "backbone" / "config.json").exists()
