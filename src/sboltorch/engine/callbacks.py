"""Training callbacks.

Raw PyTorch gives us no early stopping or checkpointing, so we provide small,
explicit callbacks instead of a heavyweight framework. Each is independently
testable and the Trainer simply invokes them in order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import wandb

from sboltorch.config import RunConfig
from sboltorch.data.materialize import MaterializedCorpus
from sboltorch.datasets.splits import Split
from sboltorch.engine.trainer import Callback, Trainer


def _is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


class EarlyStopping(Callback):
    def __init__(self, monitor: str, mode: str = "min", patience: int = 5, min_delta: float = 0.0) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf") if mode == "min" else float("-inf")
        self._waited = 0

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor]
        if _is_improvement(current, self._best, self.mode, self.min_delta):
            self._best = current
            self._waited = 0
        else:
            self._waited += 1
            if self._waited >= self.patience:
                trainer.should_stop = True


class ModelCheckpoint(Callback):
    """Saves the model state dict whenever the monitored metric improves.

    Under distributed training only the main rank writes; every rank tracks the
    best metric identically (metrics are reduced before callbacks see them)."""

    def __init__(self, output_dir: str | Path, monitor: str, mode: str = "min", *, is_main: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.monitor = monitor
        self.mode = mode
        self.is_main = is_main
        self._best = float("inf") if mode == "min" else float("-inf")
        self.best_path: Path | None = None

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor]
        if not _is_improvement(current, self._best, self.mode, min_delta=0.0):
            return
        self._best = current
        if not self.is_main:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.output_dir / "best.pt"
        trainer.save_checkpoint(self.best_path, epoch=epoch, metrics=metrics)


class PeriodicCheckpoint(Callback):
    """Writes a full, resumable training checkpoint every ``every_n_steps`` steps.

    Where ModelCheckpoint keeps the *best* model by a metric, this keeps a rolling
    ``last.pt`` carrying optimizer/scheduler/scaler/RNG state, so a long run can
    resume after an interruption. Resume continues from the next epoch boundary.
    """

    def __init__(
        self, output_dir: str | Path, every_n_steps: int, filename: str = "last.pt", *, is_main: bool = True
    ) -> None:
        self.output_dir = Path(output_dir)
        self.every_n_steps = every_n_steps
        self.filename = filename
        self.is_main = is_main

    def on_step_end(self, trainer: Trainer, step: int, logs: dict[str, float]) -> None:
        if not self.is_main or self.every_n_steps <= 0 or step % self.every_n_steps != 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(self.output_dir / self.filename, epoch=trainer.current_epoch, metrics={})


class MetricLogger(Callback):
    """Prints per-epoch metrics and appends them to ``metrics.jsonl`` (main rank only)."""

    def __init__(self, output_dir: str | Path, *, is_main: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.is_main = is_main
        self._handle: Any = None

    def on_train_start(self, trainer: Trainer) -> None:
        if not self.is_main:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handle = (self.output_dir / "metrics.jsonl").open("a")

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        if not self.is_main:
            return
        record = {"epoch": epoch, "step": trainer.global_step, **metrics}
        line = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items())
        print(line)
        if self._handle is not None:
            self._handle.write(json.dumps(record) + "\n")
            self._handle.flush()

    def on_train_end(self, trainer: Trainer) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _namespaced(metrics: dict[str, float]) -> dict[str, float]:
    """Group metrics under ``val/`` or ``train/`` for the W&B dashboard."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith("val_"):
            out[f"val/{key[len('val_'):]}"] = value
        else:
            out[f"train/{key}"] = value
    return out


class WandbLogger(Callback):
    """Logs the run to Weights & Biases.

    The resolved ``RunConfig`` is sent as the run config and the corpus
    fingerprint is attached as an artifact alias, extending the library's
    reproducibility story onto the dashboard.

    Under distributed training only the main rank logs; on other ranks every hook
    is a no-op (``_run`` stays None).
    """

    def __init__(
        self,
        config: RunConfig,
        corpus: MaterializedCorpus,
        split: Split,
        output_dir: str | Path,
        *,
        is_main: bool = True,
    ) -> None:
        self.config = config
        self.corpus = corpus
        self.split = split
        self.output_dir = Path(output_dir)
        self.is_main = is_main
        self._run: Any = None

    def on_train_start(self, trainer: Trainer) -> None:
        if not self.is_main:
            return
        cfg = self.config.wandb
        self._run = wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            name=cfg.run_name,
            mode=cfg.mode,
            tags=list(cfg.tags) or None,
            group=cfg.group,
            job_type=cfg.job_type,
            config=self.config.model_dump(mode="json"),
        )
        self._run.summary["corpus_fingerprint"] = self.corpus.fingerprint
        self._run.summary["corpus_count"] = self.corpus.count
        self._run.summary["n_train"] = len(self.split.train)
        self._run.summary["n_val"] = len(self.split.val)
        self._run.summary["n_test"] = len(self.split.test)
        self._run.summary["seed"] = self.config.seed
        if cfg.watch_model:
            wandb.watch(trainer.model, log="gradients", log_freq=cfg.log_freq)

    def on_step_end(self, trainer: Trainer, step: int, logs: dict[str, float]) -> None:
        if self._run is None or step % self.config.wandb.log_freq != 0:
            return
        self._run.log({f"train/{k}": v for k, v in logs.items()}, step=step)

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        if self._run is None:
            return
        payload = _namespaced(metrics)
        payload["epoch"] = epoch
        self._run.log(payload, step=trainer.global_step)
        name, mode = trainer.task.primary_metric
        monitored = f"val_{name}"
        if monitored in metrics:
            self._run.summary[f"best/{monitored}"] = (
                min(self._run.summary.get(f"best/{monitored}", float("inf")), metrics[monitored])
                if mode == "min"
                else max(self._run.summary.get(f"best/{monitored}", float("-inf")), metrics[monitored])
            )

    def on_train_end(self, trainer: Trainer) -> None:
        if self._run is None:
            return
        if self.config.wandb.log_model:
            self._log_model_artifact()
        self._run.finish()

    def _log_model_artifact(self) -> None:
        best = self.output_dir / "best.pt"
        resolved = self.output_dir / "config.resolved.yaml"
        if not best.exists():
            return
        artifact = wandb.Artifact(f"{self._run.id}-model", type="model")
        artifact.add_file(str(best))
        if resolved.exists():
            artifact.add_file(str(resolved))
        self._run.log_artifact(artifact, aliases=["best", self.corpus.fingerprint])
