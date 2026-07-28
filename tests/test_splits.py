"""The splitting logic carries the project's central methodological claim.

If these tests fail, every metric in results.md is suspect — so they assert the
claim directly rather than trusting that `stratify=y` was passed correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import DATA_PATH, TARGET, load_raw, make_splits

FRAUD_RATE = 0.00172


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """Same schema and imbalance as the real dataset, small enough to be fast."""
    rng = np.random.default_rng(0)
    n = 40_000
    n_fraud = int(round(n * FRAUD_RATE))

    data = {"Time": np.sort(rng.uniform(0, 172_800, n))}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n)
    data["Amount"] = rng.gamma(2.0, 40.0, n)

    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_fraud, replace=False)] = 1
    data[TARGET] = y
    return pd.DataFrame(data)


def test_split_proportions_are_70_15_15(synthetic_df):
    s = make_splits(synthetic_df)
    total = len(synthetic_df)
    assert len(s.X_train) / total == pytest.approx(0.70, abs=0.01)
    assert len(s.X_val) / total == pytest.approx(0.15, abs=0.01)
    assert len(s.X_test) / total == pytest.approx(0.15, abs=0.01)


def test_fraud_rate_preserved_across_splits(synthetic_df):
    """The reason stratification is non-negotiable at a 0.17% positive rate."""
    s = make_splits(synthetic_df)
    overall = synthetic_df[TARGET].mean()
    for y in (s.y_train, s.y_val, s.y_test):
        assert y.mean() == pytest.approx(overall, rel=0.10)
        assert y.sum() > 0, "a split with zero fraud cannot be evaluated"


def test_splits_are_disjoint_and_exhaustive(synthetic_df):
    s = make_splits(synthetic_df)
    train, val, test = set(s.X_train.index), set(s.X_val.index), set(s.X_test.index)

    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert train | val | test == set(synthetic_df.index)


def test_splits_are_deterministic(synthetic_df):
    a = make_splits(synthetic_df)
    b = make_splits(synthetic_df)
    assert list(a.X_train.index) == list(b.X_train.index)
    assert list(a.X_test.index) == list(b.X_test.index)


def test_different_seed_gives_a_different_partition(synthetic_df):
    a = make_splits(synthetic_df, random_state=42)
    b = make_splits(synthetic_df, random_state=7)
    assert list(a.X_test.index) != list(b.X_test.index)


def test_target_is_not_a_feature(synthetic_df):
    s = make_splits(synthetic_df)
    assert TARGET not in s.X_train.columns
    assert TARGET not in s.X_val.columns
    assert TARGET not in s.X_test.columns


@pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not downloaded")
def test_real_dataset_splits_contain_enough_fraud_to_evaluate():
    s = make_splits(load_raw())
    assert int(s.y_val.sum()) >= 50
    assert int(s.y_test.sum()) >= 50
    assert len(s.X_train) + len(s.X_val) + len(s.X_test) == 284_807
