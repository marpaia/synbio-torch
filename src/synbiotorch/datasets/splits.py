"""Deterministic, seeded train/val/test splitting.

Splitting is a pure function of (n, ratios, seed, optional labels) so the same
inputs always produce the same partition. This is enforced at the library
level — callers never hand-roll an ad-hoc shuffle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

_SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Split:
    """Index partitions over a dataset of size ``n = len(train) + len(val) + len(test)``."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]


def _partition(indices: np.ndarray, ratios: tuple[float, float, float]) -> tuple[list[int], list[int], list[int]]:
    n = len(indices)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    # Test gets the remainder so the three partitions always cover all n.
    train = indices[:n_train]
    val = indices[n_train : n_train + n_val]
    test = indices[n_train + n_val :]
    return list(train), list(val), list(test)


def make_split(
    n: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    labels: list[float | int] | None = None,
    strategy: str = "random",
) -> Split:
    """Partition ``range(n)`` into train/val/test.

    With ``strategy='stratified'`` and labels provided, the split is balanced
    across label bins (label value for classification, quantile bins for
    continuous targets) so each partition has a comparable target distribution.
    """
    rng = np.random.default_rng(seed)

    if strategy == "stratified" and labels is not None:
        bins = _stratify_bins(labels, rng)
        train: list[int] = []
        val: list[int] = []
        test: list[int] = []
        for _, members in sorted(bins.items()):
            arr = np.array(members)
            rng.shuffle(arr)
            t, v, te = _partition(arr, ratios)
            train += t
            val += v
            test += te
        train.sort()
        val.sort()
        test.sort()
        return Split(tuple(train), tuple(val), tuple(test))

    indices = np.arange(n)
    rng.shuffle(indices)
    t, v, te = _partition(indices, ratios)
    return Split(tuple(sorted(t)), tuple(sorted(v)), tuple(sorted(te)))


def split_of(key: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42) -> str:
    """Assign a stable key to ``'train'``/``'val'``/``'test'`` by hashing.

    The partition depends only on the key, the ratios, and the seed — no global
    index, no shuffle. It is identical across processes and workers and stable as
    the corpus grows (adding records never moves existing ones), which is what
    makes the streaming split reproducible. Uses md5 (not Python's salted
    ``hash``) so the mapping is deterministic across runs.
    """
    digest = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    frac = int(digest[:8], 16) / 0x1_0000_0000  # first 32 bits -> [0, 1)
    if frac < ratios[0]:
        return "train"
    if frac < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _stratify_bins(labels: list[float | int], rng: np.random.Generator, n_bins: int = 10) -> dict[int, list[int]]:
    arr = np.asarray(labels, dtype=float)
    distinct = np.unique(arr)
    if len(distinct) <= n_bins:
        # Treat as categorical: one bin per distinct label value.
        mapping = {v: i for i, v in enumerate(distinct)}
        keyed = [mapping[v] for v in arr]
    else:
        # Continuous: quantile bins.
        quantiles = np.quantile(arr, np.linspace(0, 1, n_bins + 1)[1:-1])
        keyed = list(np.digitize(arr, quantiles))
    bins: dict[int, list[int]] = {}
    for idx, key in enumerate(keyed):
        bins.setdefault(int(key), []).append(idx)
    return bins
