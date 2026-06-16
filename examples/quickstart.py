"""Quickstart: train a graph transformer on synthetic SBOL data.

Runs fully offline (the synthetic corpus source generates SBOL components in
memory) and finishes in seconds on CPU. Demonstrates the Python API:

    python examples/quickstart.py
"""

from __future__ import annotations

import sboltorch as st


def main() -> None:
    config = st.RunConfig.model_validate(
        {
            "seed": 0,
            "output_dir": "runs/quickstart",
            "corpus": {"source": "synthetic", "n": 200, "label_key": "strength"},
            "encoder": {"kind": "graph"},
            "model": {"hidden_size": 32, "arch": {"num_hidden_layers": 2, "num_attention_heads": 4}},
            "task": {"kind": "supervised", "objective": "regression"},
            "train": {"batch_size": 16, "epochs": 40, "lr": 1.0e-3, "amp": False},
        }
    )
    metrics = st.run_training(config)
    print("final metrics:", metrics)


if __name__ == "__main__":
    main()
