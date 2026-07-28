"""Baseline: class-weighted logistic regression.

This is meant to be a real baseline, not a strawman. `class_weight='balanced'`
reweights the loss by inverse class frequency, which is the cheapest legitimate
answer to the imbalance — if a gradient-boosted model cannot beat it, that is
worth knowing before anyone spends a week tuning one.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import RANDOM_STATE
from .features import FraudFeatures

DISPLAY_NAME = "Logistic Regression (class-weighted)"


def build(random_state: int = RANDOM_STATE) -> Pipeline:
    """Feature engineering → scaling → logistic regression, as one estimator.

    Everything lives inside the Pipeline so that under cross-validation the
    scaler and the interaction-term selection are fit on the training fold only.
    """
    return Pipeline(
        steps=[
            ("features", FraudFeatures(n_top_v=4)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
