"""synbiotorch — a PyTorch library for synthetic biology and biodesign automation.

Installed as ``synbio-torch``; imported as ``synbiotorch``, commonly::

    import synbiotorch as st

    config = st.RunConfig.from_yaml("run.yaml")
    metrics = st.run_training(config)
"""

from __future__ import annotations

from synbiotorch.config import RunConfig
from synbiotorch.data.corpus import build_corpus
from synbiotorch.data.materialize import materialize
from synbiotorch.generate import generate, generate_sequence
from synbiotorch.pipeline import prepare_data, run_training
from synbiotorch.types import Alphabet, Design, Feature, Sequence

__version__ = "0.1.0"

__all__ = [
    "RunConfig",
    "Design",
    "Sequence",
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
