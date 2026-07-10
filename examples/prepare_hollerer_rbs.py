"""Prepare the Höllerer et al. 2020 RBS dataset for synbio-torch.

Höllerer et al., "Large-scale DNA-based phenotypic recording and deep learning
enable highly accurate sequence-function mapping," Nat. Commun. 11:3551 (2020),
doi:10.1038/s41467-020-17222-4. Each record is a 17-nt E. coli RBS variant whose
label is the integrated flipping profile (IFP), a translation-initiation-rate
proxy in [0, 1].

The ML-ready arrays are published in the SAPIENs repository under CC BY-NC-ND 4.0
(BorgwardtLab/SAPIENs, data/LICENSE.md). They are downloaded on demand and are
NOT redistributed with synbio-torch. This script only reshapes them locally into
one CSV the ``table`` source reads.

The SAPIENs test arrays define the fixed held-out set (27,654 variants) on which
the paper reports R^2 = 0.9265 (10-model ensemble) / 0.9148 (single ResNet), so a
``column`` split over this CSV yields a metric comparable to those numbers. The
train/validation boundary is not pinned upstream, so it is drawn here with a
fixed seed; it affects only early-stopping, not the test metric.

Usage::

    python examples/prepare_hollerer_rbs.py
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "hollerer_rbs"
OUTPUT_CSV = DATA_DIR / "hollerer_rbs.csv"
# Pinned to a commit rather than a moving branch so the corpus is reproducible.
SAPIENS_COMMIT = "316a562e85379b03ba3bea8f5149391f59c581ed"
BASE_URL = f"https://raw.githubusercontent.com/BorgwardtLab/SAPIENs/{SAPIENS_COMMIT}/data"
# Expected array lengths (train/validation pool, held-out test); 303,503 total.
EXPECTED_TRAIN_VAL = 275_849
EXPECTED_TEST = 27_654
ARRAYS = (
    "sequences_train_validation.npy",
    "targets_train_validation.npy",
    "sequences_test.npy",
    "targets_test.npy",
)
VAL_FRACTION = 0.10
SEED = 42


def _ensure_arrays() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        dest = DATA_DIR / name
        if dest.exists():
            continue
        url = f"{BASE_URL}/{name}"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)


def _load(name: str) -> np.ndarray:
    return np.load(DATA_DIR / name, allow_pickle=False)


def main() -> None:
    _ensure_arrays()

    seqs_tv = _load("sequences_train_validation.npy")
    tgts_tv = _load("targets_train_validation.npy")
    seqs_test = _load("sequences_test.npy")
    tgts_test = _load("targets_test.npy")

    # Guard against a silently-changed upstream: the pinned arrays must match the
    # published partition sizes exactly.
    if len(seqs_tv) != EXPECTED_TRAIN_VAL or len(seqs_test) != EXPECTED_TEST:
        raise ValueError(
            f"unexpected array sizes: train/val={len(seqs_tv)} (want {EXPECTED_TRAIN_VAL}), "
            f"test={len(seqs_test)} (want {EXPECTED_TEST})"
        )

    # Draw a fixed validation partition out of the combined train/validation pool.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(seqs_tv))
    n_val = int(round(VAL_FRACTION * len(seqs_tv)))
    val_idx = set(order[:n_val].tolist())

    counts = {"train": 0, "val": 0, "test": 0}
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence", "label", "split"])
        for i, (seq, tgt) in enumerate(zip(seqs_tv, tgts_tv)):
            split = "val" if i in val_idx else "train"
            writer.writerow([f"rbs_tv_{i}", str(seq), float(tgt), split])
            counts[split] += 1
        for i, (seq, tgt) in enumerate(zip(seqs_test, tgts_test)):
            writer.writerow([f"rbs_test_{i}", str(seq), float(tgt), "test"])
            counts["test"] += 1

    total = sum(counts.values())
    print(f"wrote {total} rows to {OUTPUT_CSV}")
    print(f"  train={counts['train']}  val={counts['val']}  test={counts['test']}")


if __name__ == "__main__":
    main()
