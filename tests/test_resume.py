"""Stage-1 robustness: precision resolution, step budgeting, mid-epoch eval, and
the resume guarantee — a checkpoint-and-resume run reproduces the uninterrupted
one. These run on CPU with a tiny dropout-free model so the equivalence is exact.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from synbiotorch.config import TrainConfig
from synbiotorch.datasets.dataset import Collator, EncodedDataset
from synbiotorch.encoders.sequence import SequenceEncoder
from synbiotorch.engine.callbacks import MetricLogger, PeriodicCheckpoint
from synbiotorch.engine.trainer import Callback, Trainer, resolve_precision
from synbiotorch.reproducibility import set_seed
from synbiotorch.tasks.supervised import SupervisedTask
from synbiotorch.tokenize.kmer import KmerTokenizer
from synbiotorch.types import Alphabet, Design, Sequence

CPU = torch.device("cpu")


class TinyModel(nn.Module):
    """Embedding + masked mean pool + linear head — no dropout, so runs are exact."""

    def __init__(self, vocab_size: int, dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, input_ids, attention_mask):
        emb = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).type_as(emb)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled).squeeze(-1)


class StopAfter(Callback):
    """Stops the run after ``n`` epochs, simulating an interruption."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.count = 0

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        self.count += 1
        if self.count >= self.n:
            trainer.should_stop = True


def _objects(n: int = 64) -> list[Design]:
    objs = []
    for i in range(n):
        seq = ("GC" * 12) if i % 2 == 0 else ("AT" * 12)
        objs.append(
            Design(
                iri=f"s{i}",
                record_class="http://sbols.org/v3#Sequence",
                sequence=Sequence(elements=seq, alphabet=Alphabet.DNA),
                label=1.0 if i % 2 == 0 else 0.0,
            )
        )
    return objs


def _loaders():
    tok = KmerTokenizer(k=3, max_length=64)
    enc = SequenceEncoder(tok)
    collator = Collator(tok.pad_token_id, with_labels=True, label_dtype="float")
    objs = _objects(64)
    # shuffle=False keeps batch order deterministic, so resume equivalence is exact.
    train = DataLoader(EncodedDataset(objs[:48], enc), batch_size=16, shuffle=False, collate_fn=collator)
    val = DataLoader(EncodedDataset(objs[48:], enc), batch_size=16, collate_fn=collator)
    return tok, train, val


def _weights(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_resolve_precision_matrix():
    # Off whenever amp is disabled, the device is not CUDA, or precision is fp32.
    assert resolve_precision(True, "fp16", "cpu") == (False, torch.float32, False)
    assert resolve_precision(False, "fp16", "cuda") == (False, torch.float32, False)
    assert resolve_precision(True, "fp32", "cuda") == (False, torch.float32, False)
    # On CUDA, bf16 autocasts without a scaler; fp16 autocasts with one.
    assert resolve_precision(True, "bf16", "cuda") == (True, torch.bfloat16, False)
    assert resolve_precision(True, "fp16", "cuda") == (True, torch.float16, True)


def test_resume_reproduces_uninterrupted_run(tmp_path):
    tok, train, val = _loaders()
    cfg = TrainConfig(batch_size=16, epochs=4, lr=1e-2, amp=False)

    set_seed(0)
    full = Trainer(TinyModel(tok.vocab_size), SupervisedTask("regression"), cfg, device=CPU)
    full.fit(train, val)
    expected = _weights(full._base_model)

    # Interrupt after 2 epochs, checkpoint, then resume in a fresh trainer.
    set_seed(0)
    part = Trainer(TinyModel(tok.vocab_size), SupervisedTask("regression"), cfg, callbacks=[StopAfter(2)], device=CPU)
    part.fit(train, val)
    ckpt = tmp_path / "last.pt"
    part.save_checkpoint(ckpt, epoch=1, metrics={})

    set_seed(0)
    resumed = Trainer(TinyModel(tok.vocab_size), SupervisedTask("regression"), cfg, device=CPU)
    resumed.fit(train, val, resume_from=ckpt)

    assert resumed.start_epoch == 2
    assert resumed.global_step == full.global_step
    for name, tensor in expected.items():
        torch.testing.assert_close(resumed._base_model.state_dict()[name], tensor, rtol=1e-5, atol=1e-6)


def test_max_steps_ends_the_run(tmp_path):
    tok, train, val = _loaders()
    # epochs is large; max_steps must be what stops the run, mid-epoch.
    cfg = TrainConfig(batch_size=16, epochs=100, max_steps=5, lr=1e-2, amp=False)
    trainer = Trainer(TinyModel(tok.vocab_size), SupervisedTask("regression"), cfg, device=CPU)
    trainer.fit(train, val)
    assert trainer.global_step == 5


def test_step_based_eval_writes_multiple_records(tmp_path):
    tok, train, val = _loaders()
    cfg = TrainConfig(batch_size=16, epochs=2, eval_every_n_steps=2, lr=1e-2, amp=False)
    trainer = Trainer(
        TinyModel(tok.vocab_size), SupervisedTask("regression"), cfg, callbacks=[MetricLogger(tmp_path)], device=CPU
    )
    trainer.fit(train, val)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(records) >= 2
    assert all("step" in r and "val_loss" in r for r in records)


def test_periodic_checkpoint_is_resumable(tmp_path):
    tok, train, val = _loaders()
    cfg = TrainConfig(batch_size=16, epochs=1, checkpoint_every_n_steps=2, lr=1e-2, amp=False)
    trainer = Trainer(
        TinyModel(tok.vocab_size),
        SupervisedTask("regression"),
        cfg,
        callbacks=[PeriodicCheckpoint(tmp_path, 2)],
        device=CPU,
    )
    trainer.fit(train, val)
    ckpt = tmp_path / "last.pt"
    assert ckpt.exists()
    payload = torch.load(ckpt, weights_only=False)
    assert payload["optimizer_state"] is not None
    assert payload["scheduler_state"] is not None
    assert "rng" in payload and payload["global_step"] > 0
