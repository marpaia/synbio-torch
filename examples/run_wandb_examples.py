"""Run the synthetic-data example configs with Weights & Biases tracking on.

Loads ``WANDB_API_KEY`` (and anything else) from the repo-root ``.env``, then
trains the two configs that run offline-for-data. wandb prints each run's URL as
it goes; this also prints the project workspace URL at the end.

    python examples/run_wandb_examples.py
"""

from __future__ import annotations

import os
from pathlib import Path

import wandb

import synbiotorch as st

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ["train_graph", "finetune_structure_aware"]
PROJECT = "synbio-torch-examples"


def load_env(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file."""
    if not path.exists():
        raise SystemExit(f"error: {path} not found; add WANDB_API_KEY to it first")
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> None:
    load_env(REPO_ROOT / ".env")
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("error: WANDB_API_KEY is not set (check .env)")

    for name in CONFIGS:
        config = st.RunConfig.from_yaml(REPO_ROOT / "examples" / "configs" / f"{name}.yaml")
        print(f"=== training {name} ===")
        metrics = st.run_training(config)
        print(f"{name} final metrics:", metrics)

    print(f"\nWorkspace: https://wandb.ai/{wandb.Api().default_entity}/{PROJECT}")


if __name__ == "__main__":
    main()
