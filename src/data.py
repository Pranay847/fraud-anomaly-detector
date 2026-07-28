"""Loading and splitting.

The split is the part of this project most worth getting right. With a 0.172%
positive class, an unstratified random split can hand you a validation set with
a wildly different fraud rate than the training set — or, at small enough sizes,
none at all. Every split here is stratified on the label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"

RANDOM_STATE = 42
TARGET = "Class"

# 0.15 of everything is test; then 0.176 of the remaining 0.85 is validation,
# which lands at 0.15 of the original. Result: 70 / 15 / 15.
TEST_SIZE = 0.15
VAL_SIZE_OF_TRAINVAL = 0.176


@dataclass(frozen=True)
class Splits:
    """Train / validation / test, produced once and shared by every model."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, X, y in [
            ("train", self.X_train, self.y_train),
            ("val", self.X_val, self.y_val),
            ("test", self.X_test, self.y_test),
        ]:
            rows.append(
                {
                    "split": name,
                    "rows": len(X),
                    "fraud": int(y.sum()),
                    "fraud_rate": float(y.mean()),
                }
            )
        return pd.DataFrame(rows)


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.download_data` first."
        )
    return pd.read_csv(path)


def make_splits(df: pd.DataFrame | None = None, *, random_state: int = RANDOM_STATE) -> Splits:
    """Stratified 70/15/15 split. Deterministic for a given random_state."""
    if df is None:
        df = load_raw()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE_OF_TRAINVAL,
        stratify=y_trainval,
        random_state=random_state,
    )

    return Splits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def get_cv(n_splits: int = 5, *, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Stratified K-fold for cross-validated scoring on the training set."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Seed every source of randomness we use, so runs are reproducible."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch is only needed for the autoencoder
        pass
