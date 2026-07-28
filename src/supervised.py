"""XGBoost, three ways of handling the imbalance, and a tuned final model.

The three variants exist to answer a question the project plan poses directly:
does synthetic oversampling actually beat simply telling the model that the
positive class is rare? SMOTE interpolates between minority neighbours, which
on a 492-example class can manufacture patterns that do not exist. Random
undersampling throws away ~199,000 legitimate transactions. Class weighting
keeps every row and just reweights the gradient.

Every resampler is a step *inside* an imblearn Pipeline. That is what makes the
"resampling never touches validation data" claim true rather than aspirational:
imblearn's Pipeline applies `fit_resample` during `fit` and skips it entirely
during `predict`, so under cross-validation each fold's held-out rows are
scored against a model that never saw a synthetic neighbour of them.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from .data import RANDOM_STATE, get_cv
from .features import FraudFeatures

VARIANTS = ("weighted", "smote", "under")

DISPLAY_NAMES = {
    "weighted": "XGBoost + class weighting",
    "smote": "XGBoost + SMOTE",
    "under": "XGBoost + random undersampling",
    "tuned": "XGBoost (tuned)",
}

DEFAULT_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="aucpr",
    tree_method="hist",
)


def scale_pos_weight(y) -> float:
    """n_negative / n_positive — XGBoost's built-in imbalance knob."""
    y = np.asarray(y)
    pos = float(y.sum())
    return float((len(y) - pos) / pos) if pos else 1.0


def build(
    variant: str = "weighted",
    *,
    y_train=None,
    random_state: int = RANDOM_STATE,
    n_jobs: int = 4,
    **overrides,
) -> ImbPipeline:
    """Build the feature → (resample) → XGBoost pipeline for one variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

    params = {**DEFAULT_PARAMS, **overrides}
    params["random_state"] = random_state
    params["n_jobs"] = n_jobs

    if variant == "weighted":
        if y_train is None:
            raise ValueError("the 'weighted' variant needs y_train to compute scale_pos_weight")
        params["scale_pos_weight"] = scale_pos_weight(y_train)

    steps = [("features", FraudFeatures(n_top_v=4))]
    if variant == "smote":
        steps.append(("resample", SMOTE(random_state=random_state, k_neighbors=5)))
    elif variant == "under":
        steps.append(("resample", RandomUnderSampler(random_state=random_state)))
    steps.append(("clf", XGBClassifier(**params)))

    return ImbPipeline(steps=steps)


SEARCH_SPACE = {
    "clf__max_depth": randint(3, 10),
    "clf__learning_rate": loguniform(0.01, 0.3),
    "clf__n_estimators": randint(150, 700),
    "clf__min_child_weight": randint(1, 10),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.6, 0.4),
}


def build_search(
    variant: str,
    y_train,
    *,
    n_iter: int = 25,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    search_jobs: int = 5,
    model_jobs: int = 2,
) -> RandomizedSearchCV:
    """Randomized search scored by average precision, not accuracy.

    `scoring='average_precision'` is the whole point — optimising accuracy here
    would select whichever configuration predicts "legitimate" most confidently.

    Thread budget: `search_jobs * model_jobs` should stay under the core count,
    otherwise the folds fight each other for CPU and the search gets slower.
    """
    return RandomizedSearchCV(
        estimator=build(variant, y_train=y_train, random_state=random_state, n_jobs=model_jobs),
        param_distributions=SEARCH_SPACE,
        n_iter=n_iter,
        scoring="average_precision",
        cv=get_cv(n_splits, random_state=random_state),
        n_jobs=search_jobs,
        random_state=random_state,
        refit=True,
        verbose=1,
    )
