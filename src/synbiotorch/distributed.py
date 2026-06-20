"""Distributed training context: process group, rank-aware helpers, reductions.

Everything the training loop needs to run under ``torchrun`` lives here, kept
separate from the loop itself. The backend is chosen from the device — NCCL on
CUDA, gloo otherwise — so the orchestration (rank-aware data, gradient sync,
rank-0 side effects, metric reduction) is exercisable on a CPU-only machine with
``torchrun --nproc_per_node=N``.

Outside a distributed launch (``WORLD_SIZE`` unset or 1) every helper degrades to
a no-op single-process path, so non-distributed runs are unaffected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int
    world_size: int
    local_rank: int
    backend: str
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def env_rank_world() -> tuple[int, int]:
    """Read ``(rank, world_size)`` from the environment (torchrun sets these).

    Safe to call inside DataLoader worker subprocesses, which inherit the env but
    do not join the process group.
    """
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def single_process_context(device: torch.device | None = None) -> DistContext:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistContext(rank=0, world_size=1, local_rank=0, backend="none", device=device)


def setup_distributed(strategy: str) -> DistContext:
    """Initialize the process group if launched distributed; else single-process."""
    rank, world_size = env_rank_world()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if strategy == "none" or world_size <= 1:
        return single_process_context()

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return DistContext(rank=rank, world_size=world_size, local_rank=local_rank, backend=backend, device=device)


def worker_shard(rank: int, world_size: int, worker_id: int, num_workers: int) -> tuple[int, int]:
    """Combine data-parallel rank and DataLoader worker into one global partition.

    Returns ``(global_id, global_count)`` so each (rank, worker) pair reads a
    disjoint set of shards and the union across all pairs is the whole corpus.
    """
    return rank * num_workers + worker_id, world_size * num_workers


def reduce_mean(values: dict[str, float], ctx: DistContext) -> dict[str, float]:
    """Average scalar metrics across ranks (a no-op when not distributed)."""
    if not ctx.is_distributed:
        return values
    keys = sorted(values)
    tensor = torch.tensor([values[k] for k in keys], dtype=torch.float64, device=ctx.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= ctx.world_size
    return {k: float(tensor[i]) for i, k in enumerate(keys)}


def broadcast_flag(flag: bool, ctx: DistContext) -> bool:
    """Broadcast a boolean from rank 0 so all ranks stay in lockstep (e.g. stop)."""
    if not ctx.is_distributed:
        return flag
    tensor = torch.tensor([1 if flag else 0], dtype=torch.int, device=ctx.device)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())


def barrier(ctx: DistContext) -> None:
    if ctx.is_distributed:
        dist.barrier()


def cleanup(ctx: DistContext) -> None:
    if ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()
