"""Evaluate trained Höllerer RBS runs on the fixed held-out test split.

Loads each run's resolved config and best checkpoint, rebuilds the model,
and scores it on the exact SAPIENs test partition (split == "test") so the
coefficient of determination and mean absolute error are directly comparable
to the published numbers. Prints a per-run table and, for the strongest run,
writes a measured-vs-predicted scatter used as the paper's figure.

Run after the training runs finish (not concurrently, to avoid re-syncing the
shared environment under a live run):

    uv run --with matplotlib python examples/eval_hollerer.py \
        --out ../research/synbio-torch/figures/rbs_scatter.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from synbiotorch.config import RunConfig
from synbiotorch.datasets.dataset import Collator, EncodedDataset
from synbiotorch.encoders.base import build_encoder
from synbiotorch.engine.trainer import select_device
from synbiotorch.models import build_model
from synbiotorch.pipeline import prepare_data
from synbiotorch.tokenize.base import build_tokenizer

RUNS = {
    "char (from scratch)": "runs/hollerer_scratch_char",
    "k-mer (from scratch)": "runs/hollerer_scratch_kmer",
    "DNABERT-2 (fine-tuned)": "runs/hollerer_finetune_dnabert2",
}


def _metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[float, float]:
    mae = (y_true - y_pred).abs().mean().item()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot).item()
    return r2, mae


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="rbs_scatter.pdf", help="scatter figure output path")
    args = parser.parse_args()

    device = select_device()
    print(f"device: {device}\n")
    print(f"{'configuration':<26}{'R^2':>10}{'MAE':>10}")
    print("-" * 46)

    results = {}
    for label, path in RUNS.items():
        run_dir = Path(path)
        if not (run_dir / "best.pt").exists():
            print(f"{label:<26}{'(no run)':>20}")
            continue
        y_true, y_pred = evaluate(run_dir, device)
        r2, mae = _metrics(y_true, y_pred)
        results[label] = (r2, mae, y_true, y_pred)
        print(f"{label:<26}{r2:>10.4f}{mae:>10.4f}")

    if not results:
        return
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
