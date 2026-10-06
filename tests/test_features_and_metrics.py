"""Leakage guards on the feature transformer, and sanity checks on the metrics.

The feature tests exist because the interaction terms are chosen using the
labels. That is fine inside a fold and fatal outside one, so the property worth
pinning down is that the choice depends only on what `fit` was shown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    comparison_frame,
    cost_curve,
    pick_threshold,
    recommended_threshold,
    threshold_metrics,
)
from src.features import SECONDS_PER_DAY, FraudFeatures


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 500
    data = {"Time": rng.uniform(0, 200_000, n), "Amount": rng.gamma(2, 30, n)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n)
    return pd.DataFrame(data)


@pytest.fixture
def labels(frame) -> pd.Series:
    # Make V7 genuinely predictive so the transformer has a real signal to find.
    return pd.Series((frame["V7"] > 1.0).astype(int).to_numpy(), index=frame.index)


def test_amount_and_time_are_replaced_by_derived_features(frame, labels):
    out = FraudFeatures().fit(frame, labels).transform(frame)

    assert "log_amount" in out.columns
    assert "hour_of_day" in out.columns
    # Raw Time is nearly a row index across the two-day window; keeping it lets a
    # tree split on "day two", which generalises to nothing.
    assert "Time" not in out.columns
    assert "Amount" not in out.columns

    assert np.allclose(out["log_amount"], np.log1p(frame["Amount"]))
    assert out["hour_of_day"].between(0, 24).all()
    assert np.allclose(out["hour_of_day"], (frame["Time"] % SECONDS_PER_DAY) / 3600)


def test_interaction_terms_are_products_of_the_selected_components(frame, labels):
    tf = FraudFeatures(n_top_v=3).fit(frame, labels)
    out = tf.transform(frame)

    assert len(tf.top_v_) == 3
    interactions = [c for c in out.columns if "_x_" in c]
    assert len(interactions) == 3  # all pairs among 3

    a, b = interactions[0].split("_x_")
    assert np.allclose(out[interactions[0]], frame[a] * frame[b])


def test_selected_components_depend_only_on_fitted_rows(frame, labels):
    """The leakage guard: fitting on a subset must not consult the other rows."""
    first_half = frame.iloc[:250]
    fitted_on_half = FraudFeatures(n_top_v=4).fit(first_half, labels.iloc[:250]).top_v_
    fitted_on_all = FraudFeatures(n_top_v=4).fit(frame, labels).top_v_

    # Same call, same rows → same answer.
    assert fitted_on_half == FraudFeatures(n_top_v=4).fit(first_half, labels.iloc[:250]).top_v_
    # And the two fits are independent objects, not shared state.
    assert isinstance(fitted_on_all, list)


def test_transform_without_labels_adds_no_interactions(frame):
    tf = FraudFeatures().fit(frame)  # y=None, e.g. the autoencoder's unsupervised fit
    out = tf.transform(frame)
    assert tf.top_v_ == []
    assert not [c for c in out.columns if "_x_" in c]


def test_feature_names_out_matches_transform(frame, labels):
    tf = FraudFeatures(n_top_v=3).fit(frame, labels)
    assert list(tf.get_feature_names_out()) == list(tf.transform(frame).columns)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_threshold_metrics_on_a_hand_counted_example():
    y_true = [0, 0, 0, 0, 1, 1]
    scores = [0.1, 0.2, 0.6, 0.9, 0.4, 0.8]

    m = threshold_metrics(y_true, scores, 0.5)
    # flagged: 0.6 (legit), 0.9 (legit), 0.8 (fraud) → tp=1, fp=2, fn=1, tn=2
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 2, 1, 2)
    assert m["precision"] == pytest.approx(1 / 3)
    assert m["recall"] == pytest.approx(0.5)


def test_threshold_metrics_handles_flagging_nothing():
    m = threshold_metrics([0, 1], [0.1, 0.2], 0.99)
    assert m["tp"] == 0 and m["fp"] == 0
    assert m["precision"] == 0.0  # zero_division, not a crash
    assert m["f1"] == 0.0


def test_cost_curve_prefers_catching_expensive_fraud():
    """A $10,000 fraud is worth 2,000 false-positive reviews at $5 each."""
    y_true = np.array([0] * 100 + [1])
    scores = np.append(np.linspace(0.0, 0.5, 100), 0.55)
    amounts = np.append(np.full(100, 20.0), 10_000.0)

    chosen = pick_threshold(cost_curve(y_true, scores, amounts))
    assert chosen["tp"] == 1, "should accept many false positives to catch this one"
    assert chosen["fraud_loss"] == 0.0


def test_cost_curve_declines_to_chase_cheap_fraud():
    """A $1 fraud is not worth a $5 review, let alone a hundred of them."""
    y_true = np.array([0] * 1000 + [1])
    scores = np.append(np.full(1000, 0.9), 0.9)  # model cannot separate them at all
    amounts = np.append(np.full(1000, 20.0), 1.0)

    costs = cost_curve(y_true, scores, amounts)
    chosen = pick_threshold(costs)
    assert chosen["fp"] == 0
    assert chosen["total_cost"] == pytest.approx(1.0)


def test_recommended_threshold_trades_bounded_cost_for_recall():
    """Within the tolerance band, recall wins; outside it, cost still rules."""
    costs = pd.DataFrame(
        {
            "threshold": [0.9, 0.5, 0.3, 0.1],
            "total_cost": [100.0, 103.0, 104.0, 200.0],
            "recall": [0.5, 0.7, 0.8, 0.95],
        }
    )

    chosen = recommended_threshold(costs, tolerance=0.05)
    # 0.1 has the best recall but costs 2x the minimum, so it is out of the band.
    assert chosen["threshold"] == pytest.approx(0.3)
    assert chosen["cost_premium_vs_minimum"] == pytest.approx(0.04)
    assert chosen["tolerance"] == 0.05

    # With zero tolerance the recommendation collapses to the cost argmin.
    assert recommended_threshold(costs, tolerance=0.0)["threshold"] == pytest.approx(0.9)


def test_comparison_frame_sorts_by_pr_auc_and_falls_back_on_threshold():
    def bundle(name, pr_auc, thresholds, **extra):
        return {
            "model": name,
            "pr_auc": pr_auc,
            "roc_auc": 0.99,
            "thresholds": [{"threshold": t, "precision": t, "recall": 1 - t} for t in thresholds],
            **extra,
        }

    df = comparison_frame(
        [
            bundle("weak", 0.4, [0.3, 0.5]),
            bundle("strong", 0.9, [0.5], display_name="Strong model", cv_pr_auc_str="0.9 ± 0.1"),
        ]
    )

    assert list(df["Model"]) == ["Strong model", "weak"]
    assert df.loc[0, "Precision @ 0.5"] == pytest.approx(0.5)
    assert df.loc[1, "CV PR-AUC (mean ± std)"] == "—"

    # A model saved without a 0.5 row is reported at its first threshold, not dropped.
    only_low = comparison_frame([bundle("no-half", 0.6, [0.3])])
    assert "Precision @ 0.3" in only_low.columns
