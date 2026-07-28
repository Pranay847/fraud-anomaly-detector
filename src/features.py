"""Feature engineering.

V1-V28 are PCA components, so there is no domain feature engineering to do on
them — no "merchant category", no "distance from home". What is left:

  * `Amount` is heavily right-skewed, so log1p it.
  * `Time` is seconds elapsed since the first transaction, spanning two days.
    As a raw feature that is nearly a row index, and a tree can happily split on
    "second day" — which generalises to nothing. We convert it to hour-of-day
    and drop the raw column.
  * Products of the V components that separate the classes best. PCA components
    are linearly uncorrelated by construction, but that says nothing about
    interactions, so this is one of the few places left where a hand-built
    feature can help.

Which V components are "best" is learned in `fit` from the training labels only.
Choosing them by looking at the whole dataset would leak the test set into the
feature definition.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
SECONDS_PER_DAY = 86_400


class FraudFeatures(BaseEstimator, TransformerMixin):
    """Adds log-amount, hour-of-day and top-V interaction terms.

    Parameters
    ----------
    n_top_v:
        How many V components to draw interaction terms from. All pairs among
        them are used, so n_top_v=4 adds 6 product features.
    add_interactions:
        Set False to get only the log-amount / hour-of-day transforms.
    """

    def __init__(self, n_top_v: int = 4, add_interactions: bool = True):
        self.n_top_v = n_top_v
        self.add_interactions = add_interactions

    def fit(self, X: pd.DataFrame, y=None):
        v_cols = [c for c in X.columns if c.startswith("V")]

        if self.add_interactions and y is not None and len(v_cols) >= 2:
            # Point-biserial correlation of each component with the label,
            # computed on this fold's training rows only.
            corr = X[v_cols].apply(lambda col: np.abs(np.corrcoef(col, y)[0, 1]))
            self.top_v_ = list(corr.sort_values(ascending=False).head(self.n_top_v).index)
        else:
            self.top_v_ = []

        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()

        if "Amount" in out.columns:
            out["log_amount"] = np.log1p(out["Amount"])
            out = out.drop(columns=["Amount"])

        if "Time" in out.columns:
            out["hour_of_day"] = (out["Time"] % SECONDS_PER_DAY) / 3600.0
            out = out.drop(columns=["Time"])

        for a, b in itertools.combinations(getattr(self, "top_v_", []), 2):
            out[f"{a}_x_{b}"] = X[a] * X[b]

        return out

    def get_feature_names_out(self, input_features=None):
        names = [c for c in self.feature_names_in_ if c not in ("Amount", "Time")]
        if "Amount" in self.feature_names_in_:
            names.append("log_amount")
        if "Time" in self.feature_names_in_:
            names.append("hour_of_day")
        names += [f"{a}_x_{b}" for a, b in itertools.combinations(self.top_v_, 2)]
        return np.asarray(names, dtype=object)


def save_top_v(top_v: list[str], path: Path | None = None) -> None:
    """Record which components the interactions were built from, for the writeup."""
    path = path or REPORTS_DIR / "top_v_features.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"top_v_features": top_v}, indent=2))
