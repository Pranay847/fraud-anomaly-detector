"""Hand-computed checks on the cost-sensitivity grid and the metric bundle.

Scenario: four legit transactions (scores 0.1, 0.2, 0.3, 0.7) and two frauds,
a cheap one scored 0.5 ($2) and an expensive one scored 0.9 ($100). Catching the
cheap fraud means also flagging the legit 0.7, so the choice is one review
(fp_cost) against one missed fraud ($2 + surcharge).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import DEFAULT_THRESHOLDS, cost_sensitivity, evaluate_scores

Y = np.array([0, 0, 0, 0, 1, 1])
SCORES = np.array([0.1, 0.2, 0.3, 0.7, 0.5, 0.9])
AMOUNTS = np.array([10.0, 10.0, 10.0, 10.0, 2.0, 100.0])


@pytest.fixture
def grid():
    return cost_sensitivity(Y, SCORES, AMOUNTS, fp_costs=(1.0, 5.0, 50.0), fn_surcharges=(0.0, 50.0))


def test_grid_covers_every_cost_pair_in_order(grid):
    pairs = list(zip(grid["fp_cost"], grid["fn_surcharge"]))
    assert pairs == [(1.0, 0.0), (1.0, 50.0), (5.0, 0.0), (5.0, 50.0), (50.0, 0.0), (50.0, 50.0)]


@pytest.mark.parametrize(
    "fp_cost, surcharge, caught, fps, cost",
    [
        (1.0, 0.0, 2, 1, 1.0),  # a $1 review beats a $2 loss
        (5.0, 0.0, 1, 0, 2.0),  # a $5 review does not
        (5.0, 50.0, 2, 1, 5.0),  # surcharge makes the cheap fraud worth $52
        (50.0, 0.0, 1, 0, 2.0),
        (50.0, 50.0, 2, 1, 50.0),
    ],
)
def test_optimum_flips_with_review_cost_and_surcharge(grid, fp_cost, surcharge, caught, fps, cost):
    row = grid[(grid["fp_cost"] == fp_cost) & (grid["fn_surcharge"] == surcharge)].iloc[0]
    assert row["frauds_caught"] == caught
    assert row["frauds_missed"] == 2 - caught
    assert row["false_positives"] == fps
    assert row["total_cost"] == pytest.approx(cost)


def test_surcharge_does_not_mutate_caller_amounts():
    amounts = AMOUNTS.copy()
    cost_sensitivity(Y, SCORES, amounts, fp_costs=(5.0,), fn_surcharges=(50.0,))
    np.testing.assert_array_equal(amounts, AMOUNTS)


def test_evaluate_scores_bundle_on_unbounded_scores():
    # Reconstruction errors are not probabilities; ranking is all that matters.
    errors = SCORES * 40
    result = evaluate_scores(Y, errors, model_name="ae", split="test", thresholds=(20.0,), extra={"epochs": 3})
    assert result["model"] == "ae" and result["split"] == "test"
    assert result["n"] == 6 and result["n_fraud"] == 2
    assert result["roc_auc"] == pytest.approx(7 / 8)  # legit 0.7 outranks fraud 0.5
    assert [t["threshold"] for t in result["thresholds"]] == [20.0]
    assert result["thresholds"][0]["tp"] == 2 and result["thresholds"][0]["fp"] == 1
    assert result["epochs"] == 3


def test_evaluate_scores_defaults_to_the_standard_thresholds():
    result = evaluate_scores(Y, SCORES, model_name="m")
    assert result["split"] == "val"
    assert [t["threshold"] for t in result["thresholds"]] == list(DEFAULT_THRESHOLDS)
