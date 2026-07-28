# Results

All numbers below are produced by `python -m src.train --model all` and read back from `reports/metrics/*.json`. Nothing here is typed in by hand.

## Data split

Stratified, so the fraud rate is preserved everywhere.

| Split | Rows | Fraud | Fraud rate |
|---|---:|---:|---:|
| train | 199,478 | 344 | 0.172% |
| val | 42,607 | 74 | 0.174% |
| test | 42,722 | 74 | 0.173% |

## Model comparison (validation set)

| Model | PR-AUC | ROC-AUC | Precision | Recall | CV PR-AUC (5-fold, train) | Over-prediction |
|---|---:|---:|---:|---:|---|---:|
| XGBoost (tuned) — weighted | 0.8760 | 0.9836 | 0.8824 | 0.8108 | 0.8571 (search best) | 1.0x |
| XGBoost + SMOTE | 0.8577 | 0.9899 | 0.4507 | 0.8649 | 0.8422 ± 0.0562 | 5.0x |
| XGBoost + class weighting | 0.8552 | 0.9907 | 0.8333 | 0.8108 | 0.8420 ± 0.0474 | 1.5x |
| XGBoost + random undersampling | 0.8073 | 0.9897 | 0.0604 | 0.8919 | 0.7074 ± 0.0871 | 25.9x |
| Logistic Regression (class-weighted) | 0.7636 | 0.9852 | 0.0540 | 0.9189 | 0.7838 ± 0.0585 | 39.6x |
| Autoencoder (reconstruction error) | 0.5008 | 0.9411 | 0.2710 | 0.7838 | — | n/a (not a probability) |

Precision and recall are quoted at threshold 0.5 for the probability models. The autoencoder's score is a reconstruction error, not a probability, so its row is quoted at the 99.5th percentile of its own score distribution — the comparable operating point, since 0.5 is meaningless on that scale.

**Over-prediction** is the mean predicted probability divided by the observed fraud rate — 1.0x is perfectly calibrated in the large. PR-AUC and ROC-AUC are rank-based and cannot see this at all, but it is why precision at a fixed threshold varies so wildly down the table: the more aggressively a model reweights or resamples, the more its probabilities inflate, and the less "0.5" means the same thing from one row to the next.

## Threshold choice

Cost model: reviewing a false positive costs **$5.00** of analyst time; missing a fraud costs the **full transaction amount**. Sweeping the threshold on the validation set and minimising total cost gives:

- Cost-minimising threshold: **0.3100**
- Catches 61 of 74 frauds (82.4% recall) at 87.1% precision
- Flags 9 legitimate transactions → $45.00 in review time
- Lets $1,601.80 of fraud through
- Total modelled cost: **$1,646.80** over 42,607 validation transactions

See `reports/cost_vs_threshold.png` for the full curve and `reports/cost_curve.csv` for the underlying numbers.

### The recommendation is not the argmin

The cost surface is nearly flat across a wide band of thresholds, so the minimum is a knife-edge — it wins by a fraction of a percent on a modelled quantity while giving up real recall. Reporting it as *the* answer would be false precision about an invented $5 assumption.

**Rule applied:** among thresholds whose total cost is within 5% of the minimum, take the one that catches the most fraud.

- **Recommended threshold: 0.0300**
- Catches 65 of 74 frauds (87.8% recall) at 74.7% precision
- Flags 22 legitimate transactions (0.052% of traffic) → $110.00 in review time
- Costs 3.8% more than the minimum, and buys 4 extra frauds caught

That premium is worth paying because the cost model cannot see chargeback fees, support handling, or the customer who closes their account — all of which push the true cost of a miss above the transaction value.

### Sensitivity to the cost assumptions

The $5 review cost is invented, so the optimum derived from it is worth exactly as much as the guess behind it. What transfers is the shape of the response. `fn surcharge` adds a fixed per-fraud cost on top of the transaction value — chargeback fees, support handling, the customer who leaves — because on this dataset the median fraud is under $10, and transaction value alone badly undercounts what missing one costs.

| FP cost | FN surcharge | Threshold | Recall | Precision | Caught | Missed | False alarms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| $1 | $0 | 0.3100 | 82.4% | 87.1% | 61 | 13 | 9 |
| $1 | $50 | 0.0121 | 89.2% | 61.1% | 66 | 8 | 42 |
| $5 | $0 | 0.3100 | 82.4% | 87.1% | 61 | 13 | 9 |
| $5 | $50 | 0.0371 | 87.8% | 75.6% | 65 | 9 | 21 |
| $20 | $0 | 0.9900 | 79.7% | 96.7% | 59 | 15 | 2 |
| $20 | $50 | 0.1100 | 86.5% | 85.3% | 64 | 10 | 11 |
| $50 | $0 | 0.9900 | 79.7% | 96.7% | 59 | 15 | 2 |
| $50 | $50 | 0.9900 | 79.7% | 96.7% | 59 | 15 | 2 |

## Final test set result

The test set was held out for the entire project and evaluated **exactly once**, on `XGBoost (tuned) — weighted` — the single best model by validation PR-AUC.

| Metric | Value |
|---|---:|
| PR-AUC | 0.8362 |
| ROC-AUC | 0.9676 |
| Threshold | 0.0300 |
| Precision @ threshold | 0.7011 |
| Recall @ threshold | 0.8243 |
| Frauds caught | 61 of 74 |
| False positives | 26 |

Validation PR-AUC was 0.8760; the test set came in +0.0398 lower at 0.8362.
A gap of this size on 74 positives is within sampling noise. The validation figure was the optimistic end of the range, which is why the test set is only looked at once.
