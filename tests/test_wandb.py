"""W&B integration tests. All offline: a fake ``wandb`` is patched onto the
callbacks module so nothing touches the network, auth, or the real backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from synbiotorch.config import (
    ArchConfig,
    CorpusConfig,
    EncoderConfig,
    ModelConfig,
    RunConfig,
    TaskConfig,
    TokenizerConfig,
    TrainConfig,
    WandbConfig,
)
from synbiotorch.data.materialize import MaterializedCorpus
from synbiotorch.datasets.dataset import Collator, EncodedDataset
from synbiotorch.datasets.splits import Split
from synbiotorch.encoders.sequence import SequenceEncoder
from synbiotorch.engine.callbacks import Callback, WandbLogger, _namespaced
from synbiotorch.engine.trainer import Trainer
from synbiotorch.pipeline import run_training
from synbiotorch.tasks.supervised import SupervisedTask
from synbiotorch.tokenize.kmer import KmerTokenizer
from synbiotorch.types import Alphabet, Design, Sequence

CPU = torch.device("cpu")


class FakeArtifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class FakeRun:
    def __init__(self) -> None:
        self.id = "run123"
        self.summary: dict[str, Any] = {}
        self.logged: list[dict[str, Any]] = []
        self.artifacts: list[tuple[FakeArtifact, list[str]]] = []
        self.finished = False

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        self.logged.append({"step": step, **data})

    def log_artifact(self, artifact: FakeArtifact, aliases: list[str]) -> None:
        self.artifacts.append((artifact, aliases))

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    """Stands in for the ``wandb`` module."""

    def __init__(self) -> None:
        self.init_kwargs: dict[str, Any] | None = None
        self.run: FakeRun | None = None
        self.watched: list[Any] = []

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_kwargs = kwargs
        self.run = FakeRun()
        return self.run

    def watch(self, model: Any, **kwargs: Any) -> None:
        self.watched.append(model)

    def Artifact(self, name: str, type: str) -> FakeArtifact:  # noqa: N802 - mirror wandb API
        return FakeArtifact(name, type)


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> FakeWandb:
    fake = FakeWandb()
    monkeypatch.setattr("synbiotorch.engine.callbacks.wandb", fake)
    return fake


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).type_as(emb)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled).squeeze(-1)


def _objects(n: int = 32) -> list[Design]:
    objs = []
    for i in range(n):
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


def _run_with_logger(output_dir: Path, wandb_cfg: WandbConfig, *, epochs: int = 2) -> WandbLogger:
    tok = KmerTokenizer(k=3, max_length=64)
    enc = SequenceEncoder(tok)
    task = SupervisedTask(objective="regression")
    model = TinyModel(tok.vocab_size)
    collator = Collator(tok.pad_token_id, with_labels=True, label_dtype="float")
    train = DataLoader(EncodedDataset(_objects(32), enc), batch_size=8, shuffle=True, collate_fn=collator)
    val = DataLoader(EncodedDataset(_objects(8), enc), batch_size=8, collate_fn=collator)

    config = RunConfig(corpus=CorpusConfig(source="synthetic"), wandb=wandb_cfg)
    corpus = MaterializedCorpus(path=output_dir, fingerprint="syn-deadbeef", count=40)
    split = Split(train=tuple(range(32)), val=tuple(range(32, 40)), test=())
    # The checkpoint artifact reads these from output_dir.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "best.pt").write_bytes(b"")
    (output_dir / "config.resolved.yaml").write_text(config.to_yaml())

    logger = WandbLogger(config, corpus, split, output_dir)
    trainer = Trainer(model, task, TrainConfig(epochs=epochs, lr=1e-2, amp=False), callbacks=[logger], device=CPU)
    trainer.fit(train, val)
    return logger


def test_wandbconfig_defaults_and_roundtrip():
    cfg = RunConfig(corpus=CorpusConfig(source="synthetic"))
    assert cfg.wandb.enabled is False
    # A config without a wandb block still validates and round-trips.
    restored = RunConfig.model_validate(cfg.model_dump(mode="json"))
    assert restored.wandb.enabled is False
    assert "wandb" in cfg.to_yaml()


def test_namespacing():
    out = _namespaced({"train_loss": 1.0, "val_loss": 0.5, "val_mae": 0.2})
    assert out == {"train/train_loss": 1.0, "val/loss": 0.5, "val/mae": 0.2}


def test_logger_lifecycle(tmp_path, fake_wandb):
    cfg = WandbConfig(enabled=True, project="p", mode="offline", log_freq=1, watch_model=True, log_model=True)
    _run_with_logger(tmp_path, cfg, epochs=2)

    run = fake_wandb.run
    assert run is not None
    # Resolved config is sent as the run config, including the wandb block.
    assert fake_wandb.init_kwargs is not None
    assert fake_wandb.init_kwargs["config"]["wandb"]["enabled"] is True
    assert fake_wandb.init_kwargs["mode"] == "offline"
    # Lineage in the summary.
    assert run.summary["corpus_fingerprint"] == "syn-deadbeef"
    assert run.summary["n_train"] == 32
    assert run.summary["n_val"] == 8
    # Model was watched.
    assert fake_wandb.watched
    # Per-step and per-epoch metrics, namespaced.
    step_logs = [r for r in run.logged if "train/step_loss" in r]
    epoch_logs = [r for r in run.logged if "epoch" in r]
    assert step_logs and all("train/lr" in r for r in step_logs)
    assert len(epoch_logs) == 2
    assert all("val/loss" in r and "train/train_loss" in r for r in epoch_logs)
    # Best checkpoint logged as an artifact aliased by fingerprint, then finish.
    assert len(run.artifacts) == 1
    artifact, aliases = run.artifacts[0]
    assert aliases == ["best", "syn-deadbeef"]
    assert any(f.endswith("best.pt") for f in artifact.files)
    assert run.finished is True


def test_log_freq_throttles_step_logs(tmp_path, fake_wandb):
    # With log_freq far larger than the step count, no per-step logs are emitted.
    cfg = WandbConfig(enabled=True, mode="offline", log_freq=1000, log_model=False)
    _run_with_logger(tmp_path, cfg, epochs=1)
    run = fake_wandb.run
    assert run is not None
    assert not [r for r in run.logged if "train/step_loss" in r]


def test_on_train_end_runs_even_when_epoch_raises():
    """The fit() try/finally must always tear callbacks down."""

    class Boom(Callback):
        def on_epoch_end(self, trainer, epoch, metrics):
            raise ValueError("boom")

    class Spy(Callback):
        def __init__(self) -> None:
            self.ended = False

        def on_train_end(self, trainer) -> None:
            self.ended = True

    tok = KmerTokenizer(k=3, max_length=64)
    enc = SequenceEncoder(tok)
    collator = Collator(tok.pad_token_id, with_labels=True, label_dtype="float")
    train = DataLoader(EncodedDataset(_objects(16), enc), batch_size=8, collate_fn=collator)
    spy = Spy()
    trainer = Trainer(
        TinyModel(tok.vocab_size),
        SupervisedTask(objective="regression"),
        TrainConfig(epochs=1, lr=1e-2, amp=False),
        callbacks=[Boom(), spy],
        device=CPU,
    )
    with pytest.raises(ValueError, match="boom"):
        trainer.fit(train)
    assert spy.ended is True


def test_real_wandb_disabled_mode_conformance(tmp_path, monkeypatch):
    """Drive the *real* wandb library (mode=disabled: no network/auth/files) so
    every call we make is validated against the genuine API, not the fake."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")
    cfg = WandbConfig(enabled=True, project="conformance", mode="disabled", log_freq=1, watch_model=True)
    logger = _run_with_logger(tmp_path, cfg, epochs=2)
    # The real run object accepted init kwargs, summary writes, log, watch,
    # Artifact/add_file/log_artifact, and finish without raising. It even reads
    # our lineage back in-process.
    assert logger._run is not None
    assert logger._run.summary["corpus_fingerprint"] == "syn-deadbeef"
    assert logger._run.summary["n_train"] == 32


def test_run_training_end_to_end_offline(tmp_path, monkeypatch):
    """The whole pipeline wires WandbLogger correctly and a wandb-enabled run
    completes fully offline (from-scratch model + kmer tokenizer => no download)."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")
    config = RunConfig(
        seed=0,
        output_dir=str(tmp_path / "run"),
        corpus=CorpusConfig(source="synthetic", n=40, label_key="measure", cache_dir=str(tmp_path / "cache")),
        tokenizer=TokenizerConfig(kind="kmer", k=3, max_length=64),
        encoder=EncoderConfig(kind="sequence"),
        model=ModelConfig(
            from_scratch=True,
            hidden_size=48,
            arch=ArchConfig(
                num_hidden_layers=2, num_attention_heads=4, intermediate_size=96, max_position_embeddings=128
            ),
        ),
        task=TaskConfig(kind="supervised", objective="regression"),
        train=TrainConfig(epochs=2, batch_size=8, lr=1e-3, amp=False),
        wandb=WandbConfig(enabled=True, mode="disabled", project="e2e", log_freq=1),
    )
    metrics = run_training(config)
    assert "val_loss" in metrics
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "config.resolved.yaml").exists()
