"""End-to-end orchestration: config -> corpus -> dataset -> model -> training.

This wires the layers together for the supervised / frozen-backbone sequence
path. It is intentionally a straight-line function so the flow is readable; each
step delegates to a swappable component built from config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from torch.utils.data import DataLoader, DistributedSampler

from synbiotorch.config import RunConfig
from synbiotorch.data.corpus import build_corpus
from synbiotorch.data.materialize import MaterializedCorpus, materialize
from synbiotorch.datasets.causal_collator import CausalCollator
from synbiotorch.datasets.dataset import Collator, EncodedDataset
from synbiotorch.datasets.mlm_collator import MlmCollator
from synbiotorch.datasets.packing import PackedDataset
from synbiotorch.datasets.splits import Split, make_split, split_from_assignments
from synbiotorch.datasets.streaming import StreamingEncodedDataset
from synbiotorch.distributed import DistContext, barrier, cleanup, setup_distributed
from synbiotorch.encoders.base import build_encoder
from synbiotorch.engine.callbacks import (
    Callback,
    EarlyStopping,
    MetricLogger,
    ModelCheckpoint,
    PeriodicCheckpoint,
    WandbLogger,
)
from synbiotorch.engine.trainer import Trainer
from synbiotorch.exceptions import ConfigError
from synbiotorch.models import build_model
from synbiotorch.models.causal import CausalLMModel
from synbiotorch.models.mlm import MaskedLMModel
from synbiotorch.reproducibility import set_seed
from synbiotorch.tasks.base import Task, build_task
from synbiotorch.tokenize.base import build_tokenizer
from synbiotorch.types import Design


@dataclass
class PreparedData:
    corpus: MaterializedCorpus
    objects: list[Design]
    split: Split


def prepare_data(config: RunConfig) -> PreparedData:
    """Materialize the corpus and compute the seeded split."""
    corpus = build_corpus(config.corpus)
    materialized = materialize(corpus, config.corpus.cache_dir, shard_size=config.corpus.shard_size)
    objects = materialized.read_all()

    if config.splits.strategy == "column":
        split = split_from_assignments([o.raw.get(config.splits.column) for o in objects])
    else:
        supervised = config.task.kind in ("supervised", "frozen")
        labels = [o.label for o in objects] if supervised and config.splits.strategy == "stratified" else None
        if labels is not None and any(label is None for label in labels):
            labels = None  # cannot stratify on partially-unlabeled data
        split = make_split(
            len(objects),
            ratios=config.splits.ratios,
            seed=config.seed,
            labels=labels,  # type: ignore[arg-type]
            strategy=config.splits.strategy,
        )
    return PreparedData(corpus=materialized, objects=objects, split=split)


def _loader(
    objects: list[Design],
    indices: tuple[int, ...],
    encoder: object,
    collator: Callable[[list[Any]], Any],
    config: RunConfig,
    ctx: DistContext,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = EncodedDataset([objects[i] for i in indices], encoder)  # type: ignore[arg-type]
    # Under DDP a DistributedSampler gives each rank a disjoint slice; it owns the
    # shuffle, so the DataLoader's own shuffle is off when a sampler is present.
    sampler: DistributedSampler | None = (
        DistributedSampler(dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=shuffle, seed=config.seed)
        if ctx.is_distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=config.train.num_workers,
        collate_fn=collator,
    )


def _collator_for(config: RunConfig, tokenizer: Any, task: Task) -> Callable[[list[Any]], Any]:
    """Pick the collator for the objective: MLM masking, causal next-token shift,
    or supervised padding+labels."""
    if config.task.kind == "mlm":
        return MlmCollator(tokenizer, mlm_probability=config.task.mlm_probability)
    if config.task.kind == "causal":
        return CausalCollator(tokenizer.pad_token_id)
    return Collator(tokenizer.pad_token_id, with_labels=True, label_dtype=task.label_dtype)


def _build_sequence_run(config: RunConfig, data: PreparedData, task: Task, ctx: DistContext) -> tuple:
    """Build (model, train_loader, val_loader, adapter) for the sequence/MLM/causal path."""
    tokenizer = build_tokenizer(config.tokenizer)
    encoder = build_encoder(config.encoder, tokenizer)
    spec = encoder.output_spec
    model = build_model(config.model, config.task, vocab_size=spec.vocab_size, pad_token_id=spec.pad_token_id)
    collator = _collator_for(config, tokenizer, task)
    train_loader = _loader(data.objects, data.split.train, encoder, collator, config, ctx, shuffle=True)
    val_loader = _loader(data.objects, data.split.val, encoder, collator, config, ctx, shuffle=False)
    return model, train_loader, val_loader, None


def _build_streaming_run(config: RunConfig, materialized: MaterializedCorpus, task: Task) -> tuple:
    """Build (model, train_loader, val_loader, adapter) for the streaming path.

    Train/val are streamed straight from the sharded Parquet via a hash split, so
    no full corpus is held in memory. Packing yields fixed-size LM blocks; the
    unpacked path encodes per object with a shuffle buffer for the train stream.
    """
    tokenizer = build_tokenizer(config.tokenizer)
    ratios = config.splits.ratios
    seed = config.seed
    collator = _collator_for(config, tokenizer, task)

    if config.packing.enabled:
        if config.task.kind not in ("mlm", "causal"):
            raise ConfigError("packing is supported only for task.kind: mlm or causal")
        model = build_model(
            config.model, config.task, vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id
        )
        block = config.packing.block_size
        train_ds: object = PackedDataset(
            materialized, tokenizer, block_size=block, which="train", ratios=ratios, seed=seed
        )
        val_ds: object = PackedDataset(materialized, tokenizer, block_size=block, which="val", ratios=ratios, seed=seed)
    else:
        encoder = build_encoder(config.encoder, tokenizer)
        spec = encoder.output_spec
        model = build_model(config.model, config.task, vocab_size=spec.vocab_size, pad_token_id=spec.pad_token_id)
        # A shuffle window over the train stream; val stays in shard order.
        shuffle_buffer = max(64, config.train.batch_size * 64)
        train_ds = StreamingEncodedDataset(
            materialized, encoder, which="train", ratios=ratios, seed=seed, shuffle_buffer=shuffle_buffer
        )
        val_ds = StreamingEncodedDataset(materialized, encoder, which="val", ratios=ratios, seed=seed)

    def loader(dataset: object) -> DataLoader:
        # IterableDataset forbids shuffle=True; shuffling is the dataset's job.
        return DataLoader(
            dataset,  # type: ignore[arg-type]
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            collate_fn=collator,
        )

    return model, loader(train_ds), loader(val_ds), None


def _build_graph_run(config: RunConfig, data: PreparedData) -> tuple:
    """Build (model, train_loader, val_loader, adapter) for the graph path."""
    from torch_geometric.loader import DataLoader as GeoLoader

    from .encoders.graph import GraphEncoder
    from .encoders.structure import DEFAULT_ROLES
    from .engine.batch import GraphBatchAdapter
    from .models.graph import build_graph_model

    encoder = GraphEncoder(roles=config.encoder.roles or DEFAULT_ROLES)
    model = build_graph_model(config.model, config.task, encoder.spec)

    def loader(indices: tuple[int, ...], *, shuffle: bool) -> GeoLoader:
        dataset = EncodedDataset([data.objects[i] for i in indices], encoder)
        return GeoLoader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=shuffle,
            num_workers=config.train.num_workers,
        )

    return model, loader(data.split.train, shuffle=True), loader(data.split.val, shuffle=False), GraphBatchAdapter()


def run_training(config: RunConfig, *, resume_from: str | Path | None = None) -> dict[str, float]:
    """Run the full training pipeline and return the final epoch's metrics."""
    ctx = setup_distributed(config.train.distributed.strategy)
    try:
        return _run(config, ctx, resume_from)
    finally:
        cleanup(ctx)


def _run(config: RunConfig, ctx: DistContext, resume_from: str | Path | None) -> dict[str, float]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.resolved.yaml").write_text(config.to_yaml())

    task = build_task(config.task)
    streaming = config.streaming or config.packing.enabled

    # Rank 0 populates the shared Parquet cache first; the others wait, then read
    # it — otherwise ranks race to write the same shards. Later materialize() calls
    # hit the cache and are no-ops.
    if ctx.is_distributed:
        if ctx.is_main:
            materialize(build_corpus(config.corpus), config.corpus.cache_dir, shard_size=config.corpus.shard_size)
        barrier(ctx)

    if config.encoder.kind == "graph":
        if streaming:
            raise ConfigError("streaming/packing is not supported for the graph modality")
        if ctx.is_distributed:
            raise ConfigError("distributed training is not supported for the graph modality")
        data = prepare_data(config)
        model, train_loader, val_loader, adapter = _build_graph_run(config, data)
        corpus_ref, split_ref = data.corpus, data.split
    elif streaming:
        if config.splits.strategy != "hash":
            raise ConfigError("streaming requires splits.strategy: hash")
        corpus = build_corpus(config.corpus)
        materialized = materialize(corpus, config.corpus.cache_dir, shard_size=config.corpus.shard_size)
        model, train_loader, val_loader, adapter = _build_streaming_run(config, materialized, task)
        # Streaming computes the split lazily, so partition sizes are not known up front.
        corpus_ref, split_ref = materialized, Split((), (), ())
    else:
        data = prepare_data(config)
        model, train_loader, val_loader, adapter = _build_sequence_run(config, data, task, ctx)
        corpus_ref, split_ref = data.corpus, data.split

    metric_name, mode = task.primary_metric
    monitored = f"val_{metric_name}"
    # Writing callbacks act only on the main rank; metric-driven ones run on all
    # ranks (metrics are reduced to identical values, so decisions agree).
    callbacks: list[Callback] = [
        MetricLogger(output_dir, is_main=ctx.is_main),
        ModelCheckpoint(output_dir, monitor=monitored, mode=mode, is_main=ctx.is_main),
    ]
    if config.train.checkpoint_every_n_steps:
        callbacks.append(PeriodicCheckpoint(output_dir, config.train.checkpoint_every_n_steps, is_main=ctx.is_main))
    if config.train.early_stop is not None:
        es = config.train.early_stop
        callbacks.append(EarlyStopping(monitor=es.monitor, mode=es.mode, patience=es.patience, min_delta=es.min_delta))
    if config.wandb.enabled:
        callbacks.append(WandbLogger(config, corpus_ref, split_ref, output_dir, is_main=ctx.is_main))

    trainer = Trainer(model, task, config.train, callbacks=callbacks, batch_adapter=adapter, dist_ctx=ctx)
    metrics = trainer.fit(train_loader, val_loader, resume_from=resume_from)

    if ctx.is_main:
        (output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2))
        # Write the pretrained model in HF format so a later run (a supervised
        # fine-tune, or generation for a causal LM) can point `model.backbone` here.
        if isinstance(model, (MaskedLMModel, CausalLMModel)):
            model.save_pretrained(output_dir / "backbone")

    return metrics
