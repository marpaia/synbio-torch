"""Evaluate the Kosuri composability sweep on each run's held-out test split.

Six runs make up the sweep: three encoders (flat sequence, structure-aware,
graph) crossed with two splits (``random`` and ``partout``, the held-out-parts
split). For each run this loads the resolved config and best checkpoint, rebuilds
the model and encoder, and scores it on that run's test partition. The reported
score is the coefficient of determination R^2 = 1 - SS_res/SS_tot on the log10
protein-level target, with a percentile 95% CI from resampling the test set.

Results are archived to ``examples/kosuri_test_metrics.json`` (the artifact the
paper's Table 1 composability block reads) and per-run predictions to
``examples/kosuri_predictions/``.

Run after the sweep finishes:

    env -u VIRTUAL_ENV uv run python examples/eval_kosuri.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from synbiotorch.config import RunConfig
from synbiotorch.datasets.dataset import Collator, EncodedDataset
from synbiotorch.engine import select_device
from synbiotorch.engine.batch import GraphBatchAdapter, TensorBatchAdapter
from synbiotorch.pipeline import prepare_data

ENCODERS = ["seq", "structure", "graph"]
SPLITS = ["random", "partout"]
ENCODER_LABEL = {"seq": "flat sequence", "structure": "structure-aware", "graph": "graph transformer"}


def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1.0 - ss_res / ss_tot).item()


def _bootstrap_r2_ci(y_true: torch.Tensor, y_pred: torch.Tensor, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    gen = torch.Generator().manual_seed(seed)
    size = y_true.numel()
    stats = torch.empty(n)
    for i in range(n):
        idx = torch.randint(0, size, (size,), generator=gen)
        stats[i] = _r2(y_true[idx], y_pred[idx])
    stats, _ = stats.sort()
    return stats[int(0.025 * n)].item(), stats[int(0.975 * n)].item()


def _build_eval(config: RunConfig, test_objects: list) -> tuple:
    """Return (model, loader, adapter) for the run's encoder kind."""
    if config.encoder.kind == "graph":
        from torch_geometric.loader import DataLoader as GeoLoader

        from synbiotorch.encoders.graph import GraphEncoder
        from synbiotorch.encoders.structure import DEFAULT_ROLES
        from synbiotorch.models.graph import build_graph_model

        encoder = GraphEncoder(roles=config.encoder.roles or DEFAULT_ROLES)
        model = build_graph_model(config.model, config.task, encoder.spec)
        dataset = EncodedDataset(test_objects, encoder)
        loader = GeoLoader(dataset, batch_size=256, shuffle=False)
        return model, loader, GraphBatchAdapter()

    from synbiotorch.encoders.base import build_encoder
    from synbiotorch.models import build_model
    from synbiotorch.tokenize.base import build_tokenizer

    tokenizer = build_tokenizer(config.tokenizer)
    encoder = build_encoder(config.encoder, tokenizer)
    spec = encoder.output_spec
    model = build_model(config.model, config.task, vocab_size=spec.vocab_size, pad_token_id=spec.pad_token_id)
    dataset = EncodedDataset(test_objects, encoder)
    collate = Collator(tokenizer.pad_token_id, with_labels=True, label_dtype=torch.float32)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, collate_fn=collate)
    return model, loader, TensorBatchAdapter()


@torch.no_grad()
def evaluate(run_dir: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    config = RunConfig.from_yaml(str(run_dir / "config.resolved.yaml"))
    data = prepare_data(config)
    test_objects = [data.objects[i] for i in data.split.test]

    model, loader, adapter = _build_eval(config, test_objects)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    preds: list[torch.Tensor] = []
    trues: list[torch.Tensor] = []
    for batch in loader:
        batch = adapter.to_device(batch, device)
        trues.append(adapter.labels(batch).float().cpu())
        out = adapter.forward(model, batch).squeeze(-1).float().detach().cpu()
        preds.append(out)
    return torch.cat(trues), torch.cat(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-out", default="examples/kosuri_test_metrics.json")
    args = parser.parse_args()

    metrics_out = Path(args.metrics_out)
    preds_dir = metrics_out.parent / "kosuri_predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"device: {device}\n")
    print(f"{'encoder':<20}{'split':<10}{'R^2':>9}{'MAE':>9}{'R^2 95% CI':>22}")
    print("-" * 70)

    summary = []
    for enc in ENCODERS:
        for split in SPLITS:
            run_dir = Path(f"runs/kosuri_{enc}_{split}")
            if not (run_dir / "best.pt").exists():
                print(f"{ENCODER_LABEL[enc]:<20}{split:<10}{'(no run)':>18}")
                continue
            y_true, y_pred = evaluate(run_dir, device)
            r2 = _r2(y_true, y_pred)
            mae = (y_true - y_pred).abs().mean().item()
            lo, hi = _bootstrap_r2_ci(y_true, y_pred)
            np.savez_compressed(preds_dir / f"{run_dir.name}.npz", y_true=y_true.numpy(), y_pred=y_pred.numpy())
            summary.append(
                {
                    "encoder": ENCODER_LABEL[enc],
                    "encoder_key": enc,
                    "split": split,
                    "n_test": int(y_true.numel()),
                    "test_r2": r2,
                    "test_mae": mae,
                    "test_r2_ci95": [lo, hi],
                    "run": run_dir.name,
                }
            )
            print(f"{ENCODER_LABEL[enc]:<20}{split:<10}{r2:>9.4f}{mae:>9.4f}{f'[{lo:.4f}, {hi:.4f}]':>22}")

    metrics_out.write_text(
        json.dumps({"device": str(device), "target": "log10(prot)", "runs": summary}, indent=2) + "\n"
    )
    print(f"\nwrote {metrics_out}")


if __name__ == "__main__":
    main()
