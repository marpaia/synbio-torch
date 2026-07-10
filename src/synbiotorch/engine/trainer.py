"""A minimal, explicit raw-PyTorch training loop.

This is the only place the library re-implements training infrastructure.
It is deliberately small and boring: AMP, gradient accumulation, gradient
clipping, a linear warmup/decay schedule, and a list of callbacks for early
stopping / checkpointing / logging. Everything model- or task-specific lives
behind the Task and model abstractions.

The loop is step-budgeted: with ``train.max_steps`` set, optimizer steps (not
epochs) end the run and evaluation/checkpointing follow a step cadence — the mode
used for pretraining over a corpus too large to think about in epochs. Checkpoints
carry full optimizer/scheduler/scaler/RNG state, so a run resumes from the next
epoch boundary after an interruption.
"""

from __future__ import annotations

import itertools
from contextlib import nullcontext
from pathlib import Path
from typing import ContextManager, Iterable, Sequence

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from synbiotorch.config import TrainConfig
from synbiotorch.distributed import DistContext, broadcast_flag, reduce_mean, single_process_context
from synbiotorch.engine.batch import BatchAdapter, TensorBatchAdapter
from synbiotorch.exceptions import ConfigError
from synbiotorch.reproducibility import rng_state, set_rng_state
from synbiotorch.tasks.base import Task


class Callback:
    """The hooks the training loop invokes. Concrete callbacks live in
    ``synbiotorch.engine.callbacks``."""

    def on_train_start(self, trainer: Trainer) -> None:
        return None

    def on_step_end(self, trainer: Trainer, step: int, logs: dict[str, float]) -> None:
        return None

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        return None

    def on_train_end(self, trainer: Trainer) -> None:
        return None


def resolve_precision(amp: bool, precision: str, device_type: str) -> tuple[bool, torch.dtype, bool]:
    """Resolve ``(autocast_enabled, autocast_dtype, scaler_enabled)`` for a run.

    Autocast and the fp16 gradient scaler engage only on CUDA; on CPU/MPS a run is
    full fp32 regardless of ``precision``. bf16 autocasts without a loss scaler.
    """
    if not amp or device_type != "cuda" or precision == "fp32":
        return False, torch.float32, False
    if precision == "bf16":
        return True, torch.bfloat16, False
    return True, torch.float16, True  # fp16: autocast + loss scaler


def _linear_schedule(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _wrap_distributed(model: torch.nn.Module, config: TrainConfig, ctx: DistContext) -> torch.nn.Module:
    """Wrap the model for the configured distribution strategy (no-op if single-process)."""
    strategy = config.distributed.strategy
    if strategy == "none" or not ctx.is_distributed:
        return model
    if strategy == "ddp":
        device_ids = [ctx.local_rank] if ctx.device.type == "cuda" else None
        return DistributedDataParallel(
            model, device_ids=device_ids, find_unused_parameters=config.distributed.find_unused_parameters
        )
    raise ConfigError(f"unknown distributed strategy: {strategy}")


def _enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Turn on HuggingFace gradient checkpointing wherever the transformer lives.

    The transformer is the model itself, an MLM model's ``lm``, or a sequence
    model's ``backbone``; a model without the hook (e.g. the graph model) is a
    no-op.
    """
    for module in (model, getattr(model, "lm", None), getattr(model, "backbone", None)):
        enable = getattr(module, "gradient_checkpointing_enable", None)
        if callable(enable):
            enable()
            return


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        task: Task,
        config: TrainConfig,
        *,
        callbacks: Sequence[Callback] | None = None,
        device: torch.device | None = None,
        batch_adapter: BatchAdapter | None = None,
        dist_ctx: DistContext | None = None,
    ) -> None:
        self.task = task
        self.config = config
        self.callbacks = list(callbacks or [])
        self.dist = dist_ctx or single_process_context(device)
        self.device = device or self.dist.device
        self.adapter = batch_adapter or TensorBatchAdapter()
        self.should_stop = False
        self.global_step = 0
        self.start_epoch = 0
        self.current_epoch = 0

        if config.gradient_checkpointing:
            _enable_gradient_checkpointing(model)
        model.to(self.device)
        # ``_base_model`` owns the parameters and state dict (and stays unwrapped
        # so checkpoints are provenance-clean); ``model`` is what we call forward
        # through — a torch.compile and/or DDP wrapper sharing the same params.
        self._base_model = model
        # torch.compile returns an nn.Module at runtime but is typed as Callable.
        core: torch.nn.Module = torch.compile(model) if config.compile else model  # type: ignore[assignment]
        self.model: torch.nn.Module = _wrap_distributed(core, config, self.dist)

        self.autocast_enabled, self.autocast_dtype, self.scaler_enabled = resolve_precision(
            config.amp, config.precision, self.device.type
        )

        # Training-state handles populated in fit() and serialized by checkpoints.
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self._val_loader: DataLoader | None = None
        self._last_eval_metrics: dict[str, float] = {}

    def _trainable_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self._base_model.parameters() if p.requires_grad]

    def _total_steps(self, train_loader: DataLoader) -> int:
        """Total optimizer steps for the LR schedule.

        A step budget always wins. Otherwise it comes from the loader length —
        which a streaming (IterableDataset) loader does not have, so streaming
        runs must set ``max_steps``.
        """
        if self.config.max_steps:
            return self.config.max_steps
        try:
            steps_per_epoch = max(1, len(train_loader) // self.config.grad_accum)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ConfigError(
                "train.max_steps is required when the dataloader has no length (streaming/iterable datasets)"
            ) from exc
        return steps_per_epoch * self.config.epochs

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        resume_from: str | Path | None = None,
    ) -> dict[str, float]:
        self._val_loader = val_loader
        self.optimizer = torch.optim.AdamW(
            self._trainable_params(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        total_steps = self._total_steps(train_loader)
        self.scheduler = _linear_schedule(self.optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.scaler_enabled)

        if resume_from is not None:
            self._load_state(resume_from)

        for cb in self.callbacks:
            cb.on_train_start(self)

        step_based_eval = self.config.eval_every_n_steps is not None
        last_metrics: dict[str, float] = {}
        # With a step budget, run epochs until max_steps is hit; otherwise the
        # epoch count bounds the run (resuming past any already-completed epochs).
        epochs: Iterable[int] = (
            itertools.count(self.start_epoch) if self.config.max_steps else range(self.start_epoch, self.config.epochs)
        )
        try:
            for epoch in epochs:
                self.current_epoch = epoch
                train_loss = self._train_epoch(train_loader, epoch)
                if step_based_eval:
                    last_metrics = self._last_eval_metrics or last_metrics
                else:
                    metrics = {"train_loss": train_loss}
                    if val_loader is not None:
                        metrics.update(self._validate(val_loader))
                    last_metrics = metrics
                    self._dispatch_epoch_end(epoch, metrics)
                # Agree on stopping across ranks so none deadlocks at the next collective.
                self.should_stop = broadcast_flag(self.should_stop, self.dist)
                if self.should_stop:
                    break
        finally:
            # Always run teardown — closes log handles and finishes the W&B run
            # even if an epoch raises.
            for cb in self.callbacks:
                cb.on_train_end(self)
        return last_metrics

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        assert self.optimizer is not None and self.scheduler is not None and self.scaler is not None
        self.model.train()
        self.optimizer.zero_grad()
        total = 0.0
        count = 0
        eval_every = self.config.eval_every_n_steps
        for step, batch in enumerate(loader):
            batch = self.adapter.to_device(batch, self.device)
            labels = self.adapter.labels(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype if self.autocast_enabled else None,
                enabled=self.autocast_enabled,
            ):
                logits = self.adapter.forward(self.model, batch)
                loss = self.task.loss(logits, labels) / self.config.grad_accum
            is_update_step = (step + 1) % self.config.grad_accum == 0
            # Under DDP, skip the gradient all-reduce on accumulation micro-steps;
            # sync only on the step that actually applies the update.
            with self._grad_sync_context(is_update_step):
                self.scaler.scale(loss).backward()
            if is_update_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self._trainable_params(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                logs: dict[str, float] = {
                    "step_loss": float(loss.item() * self.config.grad_accum),
                    "lr": float(self.scheduler.get_last_lr()[0]),
                }
                for cb in self.callbacks:
                    cb.on_step_end(self, self.global_step, logs)
                if eval_every and self.global_step % eval_every == 0 and self._val_loader is not None:
                    self._last_eval_metrics = self._validate(self._val_loader)
                    self._dispatch_epoch_end(epoch, self._last_eval_metrics)
                    self.model.train()
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    self.should_stop = True
            total += loss.item() * self.config.grad_accum
            count += 1
            if self.should_stop:
                break
        return total / max(1, count)

    def _grad_sync_context(self, is_update_step: bool) -> ContextManager:
        if not is_update_step and isinstance(self.model, DistributedDataParallel):
            return self.model.no_sync()
        return nullcontext()

    def _dispatch_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(self, epoch, metrics)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        losses: list[float] = []
        preds: list[np.ndarray] = []
        labels_all: list[np.ndarray] = []
        for batch in loader:
            batch = self.adapter.to_device(batch, self.device)
            labels = self.adapter.labels(batch)
            logits = self.adapter.forward(self.model, batch)
            losses.append(self.task.loss(logits, labels).item())
            # Ravel to 1-D so batches of differing sequence length (MLM) still
            # concatenate; regression/classification predictions are already 1-D.
            preds.append(self.task.predict(logits).detach().cpu().numpy().ravel())
            labels_all.append(labels.detach().cpu().numpy().ravel())
        metrics = {"val_loss": float(np.mean(losses)) if losses else 0.0}
        if preds:
            metrics.update(self.task.epoch_metrics(np.concatenate(preds), np.concatenate(labels_all)))
        named = {f"val_{k}" if not k.startswith("val_") else k: v for k, v in metrics.items()}
        # Average across ranks so every rank sees identical metrics — this keeps
        # metric-driven decisions (early stop, best-checkpoint) consistent without
        # extra coordination. Each rank validates its own data shard.
        return reduce_mean(named, self.dist)

    def state_dict(self) -> dict[str, object]:
        """The full training state needed to resume: weights, optimizer, schedule,
        scaler, step counter, and RNG."""
        return {
            "model_state": self._base_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "global_step": self.global_step,
            "rng": rng_state(),
        }

    def save_checkpoint(self, path: str | Path, *, epoch: int, metrics: dict[str, float]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = self.state_dict()
        payload["epoch"] = epoch
        payload["metrics"] = metrics
        torch.save(payload, path)

    def _load_state(self, path: str | Path) -> None:
        """Restore training state saved by :meth:`save_checkpoint`.

        The run continues at the epoch after the checkpointed one; ``global_step``
        and the LR schedule carry over so the step budget stays correct.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._base_model.load_state_dict(ckpt["model_state"])
        if self.optimizer is not None and ckpt.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler is not None and ckpt.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.scaler is not None and ckpt.get("scaler_state"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        if ckpt.get("rng"):
            set_rng_state(ckpt["rng"])
