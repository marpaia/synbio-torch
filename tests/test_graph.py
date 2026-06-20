"""Graph encoder + PyG graph transformer: encoding, model forward, end-to-end."""

from __future__ import annotations

import numpy as np
from torch_geometric.data import Data

from synbiotorch.config import RunConfig
from synbiotorch.encoders.graph import GraphEncoder
from synbiotorch.pipeline import run_training
from synbiotorch.sources.synthetic import generate_components


def test_encode_produces_pyg_data():
    enc = GraphEncoder()
    comp = generate_components(1, seed=1)[0]
    data = enc.encode(comp)
    assert isinstance(data, Data)
    # One Component + 4 SubComponents + 4 part Components + 1 Sequence = 10 nodes.
    assert data.x.shape[0] == len(comp.neighbors.nodes)
    assert data.x.shape[1] == 3  # (class id, role id, name-hash id)
    # Edges are bidirectional, so twice the directed composition edges.
    assert data.edge_index.shape[1] == 2 * len(comp.neighbors.edges)
    assert data.edge_type.shape[0] == data.edge_index.shape[1]
    assert abs(float(data.y.item()) - comp.label) < 1e-4  # float32 storage


def test_subcomponent_nodes_carry_roles():
    enc = GraphEncoder()
    comp = generate_components(1, seed=2)[0]
    data = enc.encode(comp)
    # At least the four sub-component nodes have a non-zero role id.
    assert int((data.x[:, 1] > 0).sum()) >= 4


def test_graph_model_forward():
    from torch_geometric.loader import DataLoader

    from synbiotorch.config import ModelConfig, TaskConfig
    from synbiotorch.models.graph import build_graph_model

    enc = GraphEncoder()
    batch = next(iter(DataLoader([enc.encode(c) for c in generate_components(4, seed=3)], batch_size=4)))
    model = build_graph_model(
        ModelConfig(hidden_size=32, arch={"num_hidden_layers": 2, "num_attention_heads": 4}),
        TaskConfig(kind="supervised", objective="regression"),
        enc.spec,
    )
    out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
    assert out.shape == (4,)


def test_graph_end_to_end(tmp_path):
    config = RunConfig.model_validate(
        {
            "seed": 1,
            "output_dir": str(tmp_path / "run"),
            "corpus": {"source": "synthetic", "n": 48, "label_key": "strength", "cache_dir": str(tmp_path / "cache")},
            "encoder": {"kind": "graph"},
            "model": {
                "hidden_size": 32,
                "arch": {"num_hidden_layers": 2, "num_attention_heads": 4},
            },
            "task": {"kind": "supervised", "objective": "regression"},
            "splits": {"strategy": "random", "ratios": [0.7, 0.15, 0.15]},
            "train": {"batch_size": 8, "epochs": 3, "lr": 5.0e-3, "amp": False},
        }
    )
    metrics = run_training(config)
    assert "val_mae" in metrics and np.isfinite(metrics["val_mae"])
