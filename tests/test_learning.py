"""Learning tests: train each capability on a learnable signal and assert the
loss actually goes down (and the headline metric improves) — not merely that the
pipeline runs. Models are tiny and trained from scratch so these stay fast,
deterministic, and offline.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from synbiotorch.config import ArchConfig, ModelConfig, TaskConfig, TrainConfig
from synbiotorch.datasets.dataset import Collator, EncodedDataset
from synbiotorch.datasets.mlm_collator import MlmCollator
from synbiotorch.encoders.sequence import SequenceEncoder
from synbiotorch.encoders.structure import StructureAwareEncoder
from synbiotorch.engine.callbacks import Callback
from synbiotorch.engine.trainer import Trainer
from synbiotorch.models import build_model
from synbiotorch.reproducibility import set_seed
from synbiotorch.sources.synthetic import generate_components
from synbiotorch.tasks.mlm import MlmTask
from synbiotorch.tasks.supervised import SupervisedTask
from synbiotorch.tokenize.kmer import KmerTokenizer
from synbiotorch.types import Alphabet, Design, Sequence

CPU = torch.device("cpu")


class History(Callback):
    """Records each epoch's metrics so a test can compare first vs last."""

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        self.rows.append(dict(metrics))


def _tiny_arch() -> ArchConfig:
    return ArchConfig(num_hidden_layers=2, num_attention_heads=4, intermediate_size=96, max_position_embeddings=256)


def _gc_objects(n: int) -> list[Design]:
    """GC-rich sequences -> label 1, AT-rich -> label 0 (a strong learnable signal)."""
    objs = []
    for i in range(n):
        gc = i % 2 == 0
        seq = ("GC" * 30) if gc else ("AT" * 30)
        objs.append(
            Design(
                iri=f"s{i}",
                record_class="http://sbols.org/v3#Sequence",
                sequence=Sequence(elements=seq, alphabet=Alphabet.DNA),
                label=1.0 if gc else 0.0,
            )
        )
    return objs


def _run(model, task, train, val, collator, *, epochs: int, lr: float, adapter=None) -> History:
    history = History()
    trainer = Trainer(
        model,
        task,
        TrainConfig(epochs=epochs, lr=lr, amp=False),
        callbacks=[history],
        device=CPU,
        batch_adapter=adapter,
    )
    trainer.fit(train, val)
    return history


def test_supervised_sequence_learns():
    set_seed(0)
    tok = KmerTokenizer(k=3, max_length=256)
    enc = SequenceEncoder(tok)
    objs = _gc_objects(120)
    model = build_model(
        ModelConfig(from_scratch=True, hidden_size=48, arch=_tiny_arch()),
        TaskConfig(kind="supervised", objective="regression"),
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_token_id,
    )
    collator = Collator(tok.pad_token_id)
    train = DataLoader(EncodedDataset(objs[:100], enc), batch_size=16, shuffle=True, collate_fn=collator)
    val = DataLoader(EncodedDataset(objs[100:], enc), batch_size=16, collate_fn=collator)
    h = _run(model, SupervisedTask("regression"), train, val, collator, epochs=15, lr=5e-3)
    assert h.rows[-1]["val_loss"] < h.rows[0]["val_loss"] * 0.7
    assert h.rows[-1]["val_r2"] > 0.5


def test_mlm_loss_decreases():
    set_seed(0)
    tok = KmerTokenizer(k=3, max_length=128)
    enc = SequenceEncoder(tok)
    # Periodic motifs make masked k-mers predictable from context -> loss must drop.
    motifs = ["ACGT" * 15, "GGCC" * 15, "TTAA" * 15]
    objs = [
        Design(iri=f"s{i}", record_class="c", sequence=Sequence(elements=motifs[i % 3], alphabet=Alphabet.DNA))
        for i in range(90)
    ]
    model = build_model(
        ModelConfig(from_scratch=True, hidden_size=48, arch=_tiny_arch()),
        TaskConfig(kind="mlm"),
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_token_id,
    )
    collator = MlmCollator(tok, mlm_probability=0.15, seed=0)
    train = DataLoader(EncodedDataset(objs[:72], enc), batch_size=12, shuffle=True, collate_fn=collator)
    val = DataLoader(EncodedDataset(objs[72:], enc), batch_size=12, collate_fn=collator)
    h = _run(model, MlmTask(), train, val, collator, epochs=20, lr=5e-3)
    assert h.rows[-1]["val_loss"] < h.rows[0]["val_loss"] * 0.7
    assert h.rows[-1]["val_masked_accuracy"] > h.rows[0]["val_masked_accuracy"]


def test_structure_aware_learns():
    set_seed(0)
    tok = KmerTokenizer(k=3, max_length=256)
    enc = StructureAwareEncoder(tok)
    objs = generate_components(160, seed=0)  # label = promoter strength
    model = build_model(
        ModelConfig(from_scratch=True, hidden_size=48, arch=_tiny_arch()),
        TaskConfig(kind="supervised", objective="regression"),
        vocab_size=enc.output_spec.vocab_size,
        pad_token_id=enc.output_spec.pad_token_id,
    )
    collator = Collator(tok.pad_token_id)
    train = DataLoader(EncodedDataset(objs[:130], enc), batch_size=16, shuffle=True, collate_fn=collator)
    val = DataLoader(EncodedDataset(objs[130:], enc), batch_size=16, collate_fn=collator)
    h = _run(model, SupervisedTask("regression"), train, val, collator, epochs=40, lr=5e-3)
    assert h.rows[-1]["val_loss"] < h.rows[0]["val_loss"] * 0.7
    assert h.rows[-1]["val_r2"] > 0.5  # genuinely generalizes, not just regressing to the mean


def test_graph_learns():
    from torch_geometric.loader import DataLoader as GeoLoader

    from synbiotorch.encoders.graph import GraphEncoder
    from synbiotorch.engine.batch import GraphBatchAdapter
    from synbiotorch.models.graph import build_graph_model

    set_seed(0)
    enc = GraphEncoder()
    objs = generate_components(200, seed=0)  # label = promoter strength
    model = build_graph_model(
        ModelConfig(hidden_size=32, arch=ArchConfig(num_hidden_layers=2, num_attention_heads=4)),
        TaskConfig(kind="supervised", objective="regression"),
        enc.spec,
    )
    train = GeoLoader([enc.encode(o) for o in objs[:160]], batch_size=16, shuffle=True)
    val = GeoLoader([enc.encode(o) for o in objs[160:]], batch_size=16)
    h = _run(model, SupervisedTask("regression"), train, val, None, epochs=40, lr=5e-3, adapter=GraphBatchAdapter())
    assert h.rows[-1]["val_loss"] < h.rows[0]["val_loss"] * 0.7
    assert h.rows[-1]["val_r2"] > 0.5  # the node identity feature lets the graph learn part-dependent labels
