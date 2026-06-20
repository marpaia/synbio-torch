"""sboltorch — a PyTorch library for synthetic biology and biodesign automation.

Installed as ``sbol-torch``; imported as ``sboltorch``, commonly::

    import sboltorch as st

    config = st.RunConfig.from_yaml("run.yaml")
    metrics = st.run_training(config)
"""

from __future__ import annotations

from sboltorch.config import RunConfig
from sboltorch.data.corpus import build_corpus
from sboltorch.data.materialize import materialize
from sboltorch.generate import generate, generate_sequence
from sboltorch.pipeline import prepare_data, run_training
from sboltorch.types import Alphabet, Feature, SbolObject, SbolSequence

__version__ = "0.1.0"

__all__ = [
    "RunConfig",
    "SbolObject",
    "SbolSequence",
    "Feature",
    "Alphabet",
    "run_training",
    "prepare_data",
    "build_corpus",
    "materialize",
    "generate",
    "generate_sequence",
    "__version__",
]
