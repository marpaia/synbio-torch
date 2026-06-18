"""Normalize non-SBOL inputs to SBOL3, then ingest them as a local corpus.

Demonstrates the full bridge from a GenBank file to a materialized,
structure-aware corpus:

  1. Load SBOL_BIN (the `sbol` CLI path) from the repo-root .env.
  2. Run scripts/normalize_sbol.sh over examples/data/ -> examples/data/normalized/
     (GenBank imported, SBOL2 upgraded, SBOL3 re-serialized to Turtle).
  3. Materialize the normalized SBOL3 as a `local` corpus and report the
     sequences, features, and composition graph the parser recovered.

    python examples/normalize_and_ingest.py

The `sbol` binary is located via SBOL_BIN in the environment or .env; see
scripts/normalize_sbol.sh. FASTA is intentionally left for the native FASTA path
(its `key=value` labels would not survive GenBank/SBOL conversion).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from sboltorch.config import CorpusConfig
from sboltorch.data.corpus import build_corpus
from sboltorch.data.materialize import materialize

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "examples" / "data"
NORMALIZED_DIR = RAW_DIR / "normalized"
NAMESPACE = "https://sbol-torch.example/demo"


def load_env(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> None:
    load_env(REPO_ROOT / ".env")

    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== normalizing {RAW_DIR} -> {NORMALIZED_DIR} ===")
    subprocess.run(
        [str(REPO_ROOT / "scripts" / "normalize_sbol.sh"), str(RAW_DIR), str(NORMALIZED_DIR)],
        env={**os.environ, "NAMESPACE": NAMESPACE},
        check=True,
    )

    config = CorpusConfig(source="local", path=str(NORMALIZED_DIR), fmt="sbol")
    corpus = build_corpus(config)

    print("\n=== parsed records ===")
    objects = list(corpus)
    for obj in objects:
        if not obj.sbol_class.endswith("Component"):
            continue
        spans = [(f.locations[0].start, f.locations[0].end) for f in obj.features if f.locations]
        seq_len = len(obj.sequence.elements) if obj.sequence else 0
        graph_nodes = len(obj.neighbors.nodes) if obj.neighbors else 0
        print(f"  {obj.display_id}: {seq_len} bp, {len(obj.features)} features {spans}, {graph_nodes} graph nodes")

    result = materialize(corpus, config.cache_dir)
    print(f"\nmaterialized {result.count} objects -> {result.path}")


if __name__ == "__main__":
    main()
