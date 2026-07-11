"""Evaluate trained Höllerer RBS runs on the fixed held-out test split.

Loads each run's resolved config and best checkpoint, rebuilds the model,
and scores it on the exact SAPIENs test partition (split == "test") so the
coefficient of determination and mean absolute error are directly comparable
to the published numbers. Prints a per-run table, archives the held-out test
metrics and predictions, and, for the strongest run, writes a measured-vs-predicted
scatter used as the paper's figure.

The combined test summary (``examples/hollerer_test_metrics.json``) is the
artifact the paper's Table 1 is read from; per-run predictions land in
``examples/hollerer_predictions/``. These held-out test metrics are distinct
from the ``final_metrics.json`` each run carries, which holds validation metrics.

Run after the training runs finish (not concurrently, to avoid re-syncing the
shared environment under a live run):

    env -u VIRTUAL_ENV uv run --with matplotlib python examples/eval_hollerer.py \
        --out ../research/synbio-torch/figures/rbs_scatter.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from synbiotorch.config import RunConfig
from synbiotorch.datasets.dataset import Collator, EncodedDataset
from synbiotorch.encoders.base import build_encoder
from synbiotorch.engine import select_device
from synbiotorch.models import build_model
from synbiotorch.pipeline import prepare_data
from synbiotorch.tokenize.base import build_tokenizer

RUNS = {
    "char (from scratch)": "runs/hollerer_scratch_char",
    "k-mer (from scratch)": "runs/hollerer_scratch_kmer",
    "DNABERT-2 (fine-tuned)": "runs/hollerer_finetune_dnabert2",
}


def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1.0 - ss_res / ss_tot).item()


def _metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, float]:
    return _r2(y_true, y_pred), (y_true - y_pred).abs().mean().item()


def _bootstrap_r2_ci(y_true: torch.Tensor, y_pred: torch.Tensor, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% CI for the test R^2 by resampling the test set with replacement."""
    gen = torch.Generator().manual_seed(seed)
    size = y_true.numel()
    stats = torch.empty(n)
    for i in range(n):
        idx = torch.randint(0, size, (size,), generator=gen)
        stats[i] = _r2(y_true[idx], y_pred[idx])
    stats, _ = stats.sort()
    return stats[int(0.025 * n)].item(), stats[int(0.975 * n)].item()


@torch.no_grad()
def evaluate(run_dir: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    config = RunConfig.from_yaml(str(run_dir / "config.resolved.yaml"))
    data = prepare_data(config)
    test = [data.objects[i] for i in data.split.test]

    tokenizer = build_tokenizer(config.tokenizer)
    encoder = build_encoder(config.encoder, tokenizer)
    spec = encoder.output_spec
    model = build_model(config.model, config.task, vocab_size=spec.vocab_size, pad_token_id=spec.pad_token_id)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    dataset = EncodedDataset(test, encoder)
    collate = Collator(tokenizer.pad_token_id, with_labels=True, label_dtype=torch.float32)
    preds: list[torch.Tensor] = []
    trues: list[torch.Tensor] = []
    batch_size = 512
    for start in range(0, len(dataset), batch_size):
        batch = collate([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels" and torch.is_tensor(v)}
        out = model(**inputs).squeeze(-1).float().cpu()
        preds.append(out)
        trues.append(batch["labels"].float())
    return torch.cat(trues), torch.cat(preds)


def _archive(
    run_dir: Path,
    preds_dir: Path,
    label: str,
    r2: float,
    mae: float,
    ci: tuple[float, float],
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> dict:
    """Archive a run's held-out test metrics and predictions.

    Predictions land in ``preds_dir`` (a tracked path) keyed by the run name; the
    ``runs/`` tree itself holds the large checkpoints and is not committed.
    """
    preds_path = preds_dir / f"{run_dir.name}.npz"
    record = {
        "configuration": label,
        "split": "test",
        "n_test": int(y_true.numel()),
        "test_r2": r2,
        "test_mae": mae,
        "test_r2_ci95": [ci[0], ci[1]],
        "run": run_dir.name,
        "checkpoint": "best.pt",
        "config": "config.resolved.yaml",
        "predictions": str(preds_path),
    }
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(preds_path, y_true=y_true.numpy(), y_pred=y_pred.numpy())
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="rbs_scatter.pdf", help="scatter figure output path")
    parser.add_argument(
        "--metrics-out",
        default="examples/hollerer_test_metrics.json",
        help="combined held-out test summary the paper's Table 1 is read from",
    )
    args = parser.parse_args()

    metrics_out = Path(args.metrics_out)
    preds_dir = metrics_out.parent / "hollerer_predictions"

    device = select_device()
    print(f"device: {device}\n")
    print(f"{'configuration':<26}{'R^2':>10}{'MAE':>10}{'R^2 95% CI':>22}")
    print("-" * 68)

    results = {}
    summary = []
    for label, path in RUNS.items():
        run_dir = Path(path)
        if not (run_dir / "best.pt").exists():
            print(f"{label:<26}{'(no run)':>20}")
            continue
        y_true, y_pred = evaluate(run_dir, device)
        r2, mae = _metrics(y_true, y_pred)
        lo, hi = _bootstrap_r2_ci(y_true, y_pred)
        results[label] = (r2, mae, y_true, y_pred)
        summary.append(_archive(run_dir, preds_dir, label, r2, mae, (lo, hi), y_true, y_pred))
        print(f"{label:<26}{r2:>10.4f}{mae:>10.4f}{f'[{lo:.4f}, {hi:.4f}]':>22}")

    if not results:
        return

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps({"device": str(device), "runs": summary}, indent=2) + "\n")
    print(f"\nwrote {metrics_out}")

    best = max(results, key=lambda k: results[k][0])
    r2, mae, y_true, y_pred = results[best]
    _scatter(y_true, y_pred, best, r2, Path(args.out))


def _scatter(y_true: torch.Tensor, y_pred: torch.Tensor, label: str, r2: float, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available; skipping figure. Re-run with `uv run --with matplotlib`.")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    ax.hexbin(y_true.numpy(), y_pred.numpy(), gridsize=60, bins="log", cmap="Blues", mincnt=1)
    ax.plot([0, 1], [0, 1], color="0.3", lw=1, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("measured translation rate")
    ax.set_ylabel("predicted translation rate")
    ax.set_title(f"{label}\n$R^2 = {r2:.3f}$", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
