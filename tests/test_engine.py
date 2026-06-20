"""End-to-end engine test with a tiny local model (no model download)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from synbiotorch.config import TrainConfig
from synbiotorch.datasets.dataset import Collator, EncodedDataset
from synbiotorch.encoders.sequence import SequenceEncoder
from synbiotorch.engine.callbacks import EarlyStopping, MetricLogger, ModelCheckpoint
from synbiotorch.engine.trainer import Trainer
from synbiotorch.tasks.supervised import SupervisedTask
from synbiotorch.tokenize.kmer import KmerTokenizer
from synbiotorch.types import Alphabet, Design, Sequence


class TinyModel(nn.Module):
    """Embedding + masked mean pool + linear head — trains in milliseconds on CPU."""

    def __init__(self, vocab_size: int, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, input_ids, attention_mask):
        emb = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).type_as(emb)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled).squeeze(-1)


def _objects(n: int = 40):
    objs = []
    for i in range(n):
        # Label correlates with GC content so the model has something to learn.
        seq = ("GC" * 10) if i % 2 == 0 else ("AT" * 10)
        objs.append(
            Design(
                iri=f"s{i}",
                record_class="http://sbols.org/v3#Sequence",
                sequence=Sequence(elements=seq, alphabet=Alphabet.DNA),
                label=1.0 if i % 2 == 0 else 0.0,
            )
        )
    return objs


def test_training_loop_runs_and_logs(tmp_path):
    tokenizer = KmerTokenizer(k=3, max_length=64)
    encoder = SequenceEncoder(tokenizer)
    task = SupervisedTask(objective="regression")
    model = TinyModel(tokenizer.vocab_size)

    collator = Collator(tokenizer.pad_token_id, with_labels=True, label_dtype="float")
    train_ds = EncodedDataset(_objects(40), encoder)
    val_ds = EncodedDataset(_objects(10), encoder)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collator)

    config = TrainConfig(batch_size=8, epochs=3, lr=1e-2, amp=False)
    callbacks = [
        MetricLogger(tmp_path),
        ModelCheckpoint(tmp_path, monitor="val_mae", mode="min"),
        EarlyStopping(monitor="val_mae", mode="min", patience=10),
    ]
    trainer = Trainer(model, task, config, callbacks=callbacks, device=torch.device("cpu"))
    metrics = trainer.fit(train_loader, val_loader)

    assert "val_mae" in metrics
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "best.pt").exists()


def test_collator_pads_to_longest():
    tokenizer = KmerTokenizer(k=3, max_length=64)
    encoder = SequenceEncoder(tokenizer)
    short = Design(iri="a", record_class="c", sequence=Sequence(elements="ACGTAC"))
    long = Design(iri="b", record_class="c", sequence=Sequence(elements="ACGTACGTACGTACGT"))
    collator = Collator(tokenizer.pad_token_id, with_labels=False)
    batch = collator([encoder.encode(short), encoder.encode(long)])
    assert batch["input_ids"].shape[0] == 2
    # Shorter sequence is padded out to the longer one's length.
    assert batch["input_ids"].shape[1] == max(len(encoder.encode(short).input_ids), len(encoder.encode(long).input_ids))


def test_tensor_batch_adapter_contract():
    from synbiotorch.engine.batch import TensorBatchAdapter

    adapter = TensorBatchAdapter()
    captured = {}

    class Recorder(nn.Module):
        def forward(self, **inputs):
            captured.update(inputs)
            return torch.zeros(1)

    batch = {
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3),
        "labels": torch.tensor([1.0, 2.0]),
    }
    on_device = adapter.to_device(batch, torch.device("cpu"))
    adapter.forward(Recorder(), on_device)
    # 'labels' is not forwarded to the model; the rest is passed by keyword.
    assert set(captured.keys()) == {"input_ids", "attention_mask"}
    assert torch.equal(adapter.labels(on_device), torch.tensor([1.0, 2.0]))
