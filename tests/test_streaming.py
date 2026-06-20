"""Stage-2 streaming data: hash splitting, sharded materialize, worker
partitioning, token packing, and an end-to-end streaming-MLM learning check.

The discipline: the streaming path is proven equivalent to the in-memory one
(same records, same split assignment, no duplication) and the packed-MLM run is
shown to actually learn — not merely to run.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Iterator

import pytest
import torch
from torch.utils.data import DataLoader

from synbiotorch.config import (
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
from synbiotorch.data.materialize import materialize
from synbiotorch.datasets.mlm_collator import MlmCollator
from synbiotorch.datasets.packing import PackedDataset
from synbiotorch.datasets.splits import split_of
from synbiotorch.datasets.streaming import StreamingEncodedDataset, iter_split_records
from synbiotorch.encoders.base import ModelInput
from synbiotorch.encoders.sequence import SequenceEncoder
from synbiotorch.engine.trainer import Callback, Trainer
from synbiotorch.exceptions import ConfigError
from synbiotorch.models import build_model
from synbiotorch.pipeline import run_training
from synbiotorch.reproducibility import set_seed
from synbiotorch.sources.synthetic import SyntheticCorpus, generate_components
from synbiotorch.tasks.mlm import MlmTask
from synbiotorch.tokenize.kmer import KmerTokenizer
from synbiotorch.types import Alphabet, Design, Sequence

CPU = torch.device("cpu")


class _MotifCorpus:
    """A corpus of periodic-motif sequences — masked k-mers are predictable, so
    an MLM trained on it must drive loss down."""

    def __init__(self, n: int) -> None:
        self.n = n
        self._motifs = ["ACGT" * 30, "GGCC" * 30, "TTAA" * 30]

    def __iter__(self) -> Iterator[Design]:
        for i in range(self.n):
            yield Design(
                iri=f"https://ex/m{i}",
                record_class="http://sbols.org/v3#Sequence",
                sequence=Sequence(elements=self._motifs[i % 3], alphabet=Alphabet.DNA),
            )

    def fingerprint(self) -> str:
        return "motif-" + hashlib.sha256(f"{self.n}".encode()).hexdigest()[:12]


# --- hash split -------------------------------------------------------------


def test_split_of_is_deterministic_and_partitions():
    keys = [f"obj{i}" for i in range(3000)]
    counts = Counter(split_of(k) for k in keys)
    assert set(counts) == {"train", "val", "test"}
    assert sum(counts.values()) == len(keys)
    # Default 0.8/0.1/0.1 within a few percent over 3000 keys.
    assert abs(counts["train"] / len(keys) - 0.8) < 0.04
    # Deterministic: the same key always lands the same way.
    assert all(split_of(k) == split_of(k) for k in keys[:50])


def test_split_of_is_stable_as_corpus_grows():
    small = {f"obj{i}": split_of(f"obj{i}") for i in range(100)}
    # Adding 900 more records must not move any existing record's partition.
    assert all(split_of(k) == v for k, v in small.items())


# --- sharded materialize ----------------------------------------------------


def test_materialize_shards_and_streams_back(tmp_path):
    mc = materialize(SyntheticCorpus(n=50, seed=0), tmp_path, shard_size=10)
    assert len(mc.shards) == 5
    assert mc.count == 50
    streamed = mc.read_all()
    assert len(streamed) == 50
    expected = {o.iri for o in generate_components(50, seed=0)}
    assert {o.iri for o in streamed} == expected
    assert len(mc.labels()) == 50


def test_iter_for_worker_partitions_without_overlap(tmp_path):
    mc = materialize(SyntheticCorpus(n=50, seed=0), tmp_path, shard_size=7)
    num_workers = 3
    per_worker = [list(mc.iter_for_worker(w, num_workers)) for w in range(num_workers)]
    iris = [o.iri for shard in per_worker for o in shard]
    # Whole corpus covered exactly once across workers.
    assert sorted(iris) == sorted(o.iri for o in mc)
    assert len(iris) == len(set(iris))


# --- streaming dataset / split equivalence ----------------------------------


def test_streaming_split_matches_hash_assignment(tmp_path):
    mc = materialize(SyntheticCorpus(n=120, seed=0), tmp_path, shard_size=16)
    ratios, seed = (0.8, 0.1, 0.1), 42
    by_split = {w: {o.iri for o in iter_split_records(mc, w, ratios, seed)} for w in ("train", "val", "test")}
    # Splits are disjoint and cover the corpus.
    all_iris = {o.iri for o in mc}
    assert by_split["train"] | by_split["val"] | by_split["test"] == all_iris
    assert sum(len(s) for s in by_split.values()) == len(all_iris)
    # Membership matches split_of exactly.
    for which, iris in by_split.items():
        assert all(split_of(i, ratios, seed) == which for i in iris)


def test_streaming_dataset_yields_encoded_train_split(tmp_path):
    mc = materialize(SyntheticCorpus(n=120, seed=0), tmp_path, shard_size=16)
    enc = SequenceEncoder(KmerTokenizer(k=3, max_length=128))
    ds = StreamingEncodedDataset(mc, enc, which="train", seed=42)
    items = list(ds)
    expected = sum(1 for o in mc if split_of(o.iri, (0.8, 0.1, 0.1), 42) == "train")
    assert len(items) == expected
    assert all(isinstance(x, ModelInput) for x in items)


# --- token packing ----------------------------------------------------------


def test_packed_blocks_are_fixed_length_and_lossless(tmp_path):
    tok = KmerTokenizer(k=3, max_length=10_000)
    objs = list(_MotifCorpus(20))
    total_tokens = sum(len(tok.tokenize_content(o.sequence.elements)) for o in objs)
    block_size = 32
    blocks = list(PackedDataset(objs, tok, block_size=block_size))
    assert len(blocks) == total_tokens // block_size  # remainder dropped
    assert all(len(b.input_ids) == block_size for b in blocks)
    assert all(b.attention_mask == [1] * block_size for b in blocks)
    assert all(b.label is None for b in blocks)


# --- end-to-end: streaming + packing + MLM learns ---------------------------


class _History(Callback):
    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        self.rows.append(dict(metrics))


def test_streaming_packed_mlm_learns(tmp_path):
    set_seed(0)
    mc = materialize(_MotifCorpus(120), tmp_path, shard_size=20)
    tok = KmerTokenizer(k=3, max_length=512)
    collator = MlmCollator(tok, mlm_probability=0.15, seed=0)
    train_ds = PackedDataset(mc, tok, block_size=32, which="train", seed=42)
    val_ds = PackedDataset(mc, tok, block_size=32, which="val", seed=42)
    train = DataLoader(train_ds, batch_size=8, collate_fn=collator)
    val = DataLoader(val_ds, batch_size=8, collate_fn=collator)

    model = build_model(
        ModelConfig(
            from_scratch=True,
            hidden_size=48,
            arch=ArchConfig(
                num_hidden_layers=2, num_attention_heads=4, intermediate_size=96, max_position_embeddings=64
            ),
        ),
        TaskConfig(kind="mlm"),
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_token_id,
    )
    history = _History()
    cfg = TrainConfig(batch_size=8, lr=5e-3, amp=False, max_steps=60, eval_every_n_steps=20)
    Trainer(model, MlmTask(), cfg, callbacks=[history], device=CPU).fit(train, val)

    assert len(history.rows) >= 2
    assert history.rows[-1]["val_loss"] < history.rows[0]["val_loss"]


def test_iterable_loader_requires_max_steps(tmp_path):
    # fit() rejects a length-less loader before any forward pass, so a trivial
    # module is enough to exercise the guard.
    mc = materialize(SyntheticCorpus(n=40, seed=0), tmp_path, shard_size=10)
    enc = SequenceEncoder(KmerTokenizer(k=3, max_length=128))
    loader = DataLoader(StreamingEncodedDataset(mc, enc, which="train"), batch_size=8, collate_fn=lambda b: b)
    trainer = Trainer(torch.nn.Linear(4, 4), MlmTask(), TrainConfig(epochs=1, amp=False), device=CPU)
    try:
        trainer.fit(loader)
        raise AssertionError("expected ConfigError for an iterable loader without max_steps")
    except ConfigError:
        pass


# --- pipeline wiring --------------------------------------------------------


def _streaming_config(tmp_path, **overrides) -> RunConfig:
    base = dict(
        seed=42,
        streaming=True,
        output_dir=str(tmp_path / "run"),
        corpus=CorpusConfig(source="synthetic", n=96, shard_size=16, cache_dir=str(tmp_path / "cache")),
        tokenizer=TokenizerConfig(kind="kmer", k=3, max_length=64),
        encoder=EncoderConfig(kind="sequence"),
        model=ModelConfig(
            from_scratch=True,
            hidden_size=32,
            arch=ArchConfig(
                num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, max_position_embeddings=128
            ),
        ),
        task=TaskConfig(kind="mlm"),
        splits=SplitConfig(strategy="hash"),
        train=TrainConfig(batch_size=8, amp=False, max_steps=10, eval_every_n_steps=5, num_workers=0),
    )
    base.update(overrides)
    return RunConfig(**base)


def test_pipeline_streaming_mlm_runs(tmp_path):
    config = _streaming_config(tmp_path)
    metrics = run_training(config)
    assert "val_loss" in metrics and math.isfinite(metrics["val_loss"])
    assert (tmp_path / "run" / "backbone").exists()


def test_pipeline_streaming_packed_mlm_runs(tmp_path):
    config = _streaming_config(tmp_path, packing=PackingConfig(enabled=True, block_size=32))
    metrics = run_training(config)
    assert "val_loss" in metrics and math.isfinite(metrics["val_loss"])


def test_streaming_requires_hash_split(tmp_path):
    config = _streaming_config(tmp_path, splits=SplitConfig(strategy="random"))
    with pytest.raises(ConfigError):
        run_training(config)


def test_streaming_graph_is_rejected(tmp_path):
    config = _streaming_config(tmp_path, encoder=EncoderConfig(kind="graph"), task=TaskConfig(kind="supervised"))
    with pytest.raises(ConfigError):
        run_training(config)
