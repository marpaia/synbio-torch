"""Stage-5 distributed orchestration (DDP), validated GPU-free on gloo.

The discipline: the rank-aware data partition is proven disjoint+covering, and the
headline — a 2-rank DDP run produces the same weights as single-process training
over the same global data, including the grad-accumulation no_sync path — is proven
with real gloo processes.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader

from sboltorch.config import (
    ArchConfig,
    CorpusConfig,
    DistributedConfig,
    EncoderConfig,
    ModelConfig,
    RunConfig,
    SplitConfig,
    TaskConfig,
    TokenizerConfig,
    TrainConfig,
)
from sboltorch.datasets.dataset import Collator, EncodedDataset
from sboltorch.distributed import (
    DistContext,
    broadcast_flag,
    cleanup,
    reduce_mean,
    setup_distributed,
    single_process_context,
    worker_shard,
)
from sboltorch.encoders.sequence import SequenceEncoder
from sboltorch.engine.trainer import Trainer, _wrap_distributed
from sboltorch.exceptions import ConfigError
from sboltorch.pipeline import run_training
from sboltorch.reproducibility import set_seed
from sboltorch.tasks.supervised import SupervisedTask
from sboltorch.tokenize.kmer import KmerTokenizer
from sboltorch.types import SbolObject, SbolSequence

CPU = torch.device("cpu")


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, input_ids, attention_mask):
        emb = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).type_as(emb)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled).squeeze(-1)


def _objects(n: int = 8) -> list[SbolObject]:
    return [
        SbolObject(
            iri=f"s{i}",
            sbol_class="http://sbols.org/v3#Sequence",
            sequence=SbolSequence(elements=("GC" * 8) if i % 2 == 0 else ("AT" * 8)),
            label=float(i),
        )
        for i in range(n)
    ]


# --- unit tests (no subprocesses) -------------------------------------------


def test_worker_shard_partitions_globally():
    world, num_workers = 2, 3
    ids = [worker_shard(r, world, w, num_workers) for r in range(world) for w in range(num_workers)]
    assert all(count == world * num_workers for _, count in ids)
    assert sorted(gid for gid, _ in ids) == list(range(world * num_workers))
    # Applied to 20 shards: every (rank, worker) reads a disjoint slice, union = all.
    shards = list(range(20))
    assigned = [
        s
        for r in range(world)
        for w in range(num_workers)
        for i, s in enumerate(shards)
        if i % (world * num_workers) == worker_shard(r, world, w, num_workers)[0]
    ]
    assert sorted(assigned) == shards


def test_single_process_helpers_are_noops():
    ctx = single_process_context(CPU)
    assert not ctx.is_distributed and ctx.is_main
    assert reduce_mean({"a": 2.0}, ctx) == {"a": 2.0}
    assert broadcast_flag(True, ctx) is True


def test_unknown_strategy_raises():
    ctx = DistContext(rank=0, world_size=2, local_rank=0, backend="gloo", device=CPU)
    cfg = TrainConfig(distributed=DistributedConfig())
    object.__setattr__(cfg.distributed, "strategy", "bogus")  # bypass enum validation
    with pytest.raises(ConfigError):
        _wrap_distributed(nn.Linear(2, 2), cfg, ctx)


# --- multiprocess gloo tests ------------------------------------------------


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _set_env(rank: int, world: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK="0"
    )


def _rank_worker(rank: int, world: int, port: int, out_dir: str) -> None:
    _set_env(rank, world, port)
    ctx = setup_distributed("ddp")
    reduced = reduce_mean({"v": float(rank + 1)}, ctx)["v"]
    Path(out_dir, f"rank{rank}.txt").write_text(f"{ctx.rank},{ctx.world_size},{reduced}")
    cleanup(ctx)


def test_ranks_and_metric_reduction(tmp_path):
    mp.spawn(_rank_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    results = {}
    for r in range(2):
        rank, world, val = Path(tmp_path, f"rank{r}.txt").read_text().split(",")
        results[int(rank)] = (int(world), float(val))
    assert set(results) == {0, 1}
    assert all(world == 2 for world, _ in results.values())
    # mean of per-rank values 1 and 2 is 1.5, identical on every rank.
    assert all(abs(val - 1.5) < 1e-6 for _, val in results.values())


def _train_single(batch_size: int, grad_accum: int) -> dict[str, torch.Tensor]:
    set_seed(0)
    tok = KmerTokenizer(k=3, max_length=32)
    enc = SequenceEncoder(tok)
    model = TinyModel(tok.vocab_size)
    loader = DataLoader(
        EncodedDataset(_objects(8), enc),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Collator(tok.pad_token_id, label_dtype="float"),
    )
    cfg = TrainConfig(batch_size=batch_size, epochs=1, lr=0.1, amp=False, grad_accum=grad_accum)
    Trainer(model, SupervisedTask("regression"), cfg, device=CPU).fit(loader)
    return {k: v.clone() for k, v in model.state_dict().items()}


def _ddp_worker(rank: int, world: int, port: int, out_dir: str, batch_size: int, grad_accum: int) -> None:
    _set_env(rank, world, port)
    set_seed(0)
    ctx = setup_distributed("ddp")
    tok = KmerTokenizer(k=3, max_length=32)
    enc = SequenceEncoder(tok)
    model = TinyModel(tok.vocab_size)
    shard = _objects(8)[rank * 4 : (rank + 1) * 4]
    loader = DataLoader(
        EncodedDataset(shard, enc),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Collator(tok.pad_token_id, label_dtype="float"),
    )
    cfg = TrainConfig(
        batch_size=batch_size,
        epochs=1,
        lr=0.1,
        amp=False,
        grad_accum=grad_accum,
        distributed=DistributedConfig(strategy="ddp"),
    )
    Trainer(model, SupervisedTask("regression"), cfg, device=CPU, dist_ctx=ctx).fit(loader)
    if ctx.is_main:
        torch.save(model.state_dict(), Path(out_dir, "ddp.pt"))
    cleanup(ctx)


def test_ddp_matches_single_process(tmp_path):
    # Two ranks over halves of the data, gradients all-reduced, must equal one
    # process over the whole batch — the core correctness proof for DDP.
    single = _train_single(batch_size=8, grad_accum=1)
    mp.spawn(_ddp_worker, args=(2, _free_port(), str(tmp_path), 4, 1), nprocs=2, join=True)
    ddp = torch.load(Path(tmp_path, "ddp.pt"), weights_only=True)
    for name, tensor in single.items():
        torch.testing.assert_close(ddp[name], tensor, rtol=1e-4, atol=1e-5)


def test_ddp_grad_accum_matches_single_process(tmp_path):
    # Same proof with grad_accum=2, which exercises the DDP no_sync() path on the
    # accumulation micro-step (no all-reduce until the update step).
    single = _train_single(batch_size=4, grad_accum=2)
    mp.spawn(_ddp_worker, args=(2, _free_port(), str(tmp_path), 2, 2), nprocs=2, join=True)
    ddp = torch.load(Path(tmp_path, "ddp.pt"), weights_only=True)
    for name, tensor in single.items():
        torch.testing.assert_close(ddp[name], tensor, rtol=1e-4, atol=1e-5)


def _pipeline_worker(rank: int, world: int, port: int, tmp_path: str) -> None:
    _set_env(rank, world, port)
    config = RunConfig(
        seed=0,
        output_dir=str(Path(tmp_path, "run")),
        corpus=CorpusConfig(source="synthetic", n=64, label_key="strength", cache_dir=str(Path(tmp_path, "cache"))),
        tokenizer=TokenizerConfig(kind="kmer", k=3, max_length=64),
        encoder=EncoderConfig(kind="sequence"),
        model=ModelConfig(
            from_scratch=True,
            hidden_size=32,
            arch=ArchConfig(
                num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, max_position_embeddings=128
            ),
        ),
        task=TaskConfig(kind="supervised", objective="regression"),
        splits=SplitConfig(strategy="random"),
        train=TrainConfig(
            batch_size=8,
            epochs=1,
            amp=False,
            lr=1e-3,
            # The from-scratch BERT's pooler isn't used by mean-pooling, so some
            # params get no gradient — exactly the case this knob exists for.
            distributed=DistributedConfig(strategy="ddp", find_unused_parameters=True),
        ),
    )
    run_training(config)


def test_pipeline_ddp_smoke(tmp_path):
    mp.spawn(_pipeline_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    final = Path(tmp_path, "run", "final_metrics.json")
    assert final.exists()  # only rank 0 writes it
    assert "val_loss" in json.loads(final.read_text())
