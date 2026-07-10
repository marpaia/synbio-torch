from __future__ import annotations

import pytest

from synbiotorch.datasets.splits import make_split, split_from_assignments
from synbiotorch.exceptions import ConfigError


def test_split_is_deterministic():
    a = make_split(100, seed=42)
    b = make_split(100, seed=42)
    assert a == b


def test_split_covers_all_indices_without_overlap():
    split = make_split(100, ratios=(0.8, 0.1, 0.1), seed=1)
    all_indices = set(split.train) | set(split.val) | set(split.test)
    assert all_indices == set(range(100))
    assert len(split.train) + len(split.val) + len(split.test) == 100
    assert not (set(split.train) & set(split.val))
    assert not (set(split.train) & set(split.test))


def test_different_seed_changes_partition():
    a = make_split(100, seed=1)
    b = make_split(100, seed=2)
    assert a != b


def test_stratified_split_balances_classes():
    labels = [0] * 50 + [1] * 50
    split = make_split(100, ratios=(0.8, 0.1, 0.1), seed=3, labels=labels, strategy="stratified")
    train_labels = [labels[i] for i in split.train]
    # Each class is ~80% represented in train, so neither dominates.
    assert 30 <= sum(1 for x in train_labels if x == 0) <= 50
    assert 30 <= sum(1 for x in train_labels if x == 1) <= 50


def test_column_split_honors_explicit_partition():
    split = split_from_assignments(["train", "test", "val", "train", "test"])
    assert split.train == (0, 3)
    assert split.val == (2,)
    assert split.test == (1, 4)


def test_column_split_accepts_aliases_and_is_case_insensitive():
    split = split_from_assignments(["TRAIN", "Validation", "valid", "Testing"])
    assert split.train == (0,)
    assert split.val == (1, 2)
    assert split.test == (3,)


def test_column_split_rejects_unknown_value():
    with pytest.raises(ConfigError):
        split_from_assignments(["train", "holdout"])


def test_column_split_rejects_missing_value():
    with pytest.raises(ConfigError):
        split_from_assignments(["train", None])
