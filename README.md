# Fraud Detection on Imbalanced Transaction Data

Detects fraudulent credit card transactions in a dataset where only 0.172% of records are fraud
(492 out of 284,807). Compares six approaches across supervised and unsupervised methods, and
selects a deployment threshold using an explicit cost model.

Built on the [ULB Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Overview

Severe class imbalance breaks the metrics most classifiers are evaluated with. A model that labels
every transaction "legitimate" achieves 99.827% accuracy on this dataset and catches zero fraud, so
accuracy is not reported anywhere in this project. PR-AUC (average precision) is the primary metric
throughout, with ROC-AUC reported as secondary.

The project covers:

- Stratified 70/15/15 splitting with 5-fold cross-validation on the training set
- A class-weighted logistic regression baseline
- XGBoost under three imbalance strategies (class weighting, SMOTE, random undersampling)
- Hyperparameter search optimizing average precision
- A PyTorch autoencoder trained only on legitimate transactions, using reconstruction error as an
  anomaly score
- Threshold selection driven by a stated cost model, with sensitivity analysis
- Error analysis on the misclassified transactions

## Results

Validation set (42,607 transactions, 74 fraud):

| Model | PR-AUC | ROC-AUC | Precision @ 0.5 | Recall @ 0.5 | Over-prediction |
|---|---:|---:|---:|---:|---:|
| **XGBoost (tuned, class-weighted)** | **0.8760** | 0.9836 | 0.8824 | 0.8108 | 1.0x |
| XGBoost + SMOTE | 0.8577 | 0.9899 | 0.4507 | 0.8649 | 5.0x |
| XGBoost + class weighting | 0.8552 | 0.9907 | 0.8333 | 0.8108 | 1.5x |
| XGBoost + random undersampling | 0.8073 | 0.9897 | 0.0604 | 0.8919 | 25.9x |
| Logistic Regression (class-weighted) | 0.7636 | 0.9852 | 0.0540 | 0.9189 | 39.6x |
| Autoencoder (reconstruction error) | 0.5008 | 0.9411 | 0.2710 | 0.7838 | n/a |

Test set, evaluated once on the model selected by validation PR-AUC:

| Metric | Value |
|---|---:|
| PR-AUC | 0.8362 |
| Recall @ threshold 0.03 | 0.8243 (61 of 74 frauds) |
| Precision @ threshold 0.03 | 0.7011 (26 false positives) |

Full numbers, including the cost sensitivity table, are in [results.md](results.md).

### Notes on the table

**ROC-AUC is not informative here.** Every model scores above 0.98 while PR-AUC ranges from 0.50 to
0.88. The autoencoder is the clearest case: 0.9411 ROC-AUC against 0.5008 PR-AUC. With 284,315 true
negatives, the false positive rate stays low regardless of how many legitimate transactions are
flagged, which compresses genuinely different models into a narrow band.

**SMOTE did not outperform class weighting.** The gap is 0.0025 PR-AUC against a cross-validated
fold spread of ±0.05. The one-standard-error rule selects the simplest variant within one standard
error of the best, which is class weighting. Random undersampling was clearly worse at 0.8073, and
had the widest fold spread at ±0.087.

**Over-prediction** is mean predicted probability divided by observed fraud rate; 1.0x is calibrated.
Resampling and reweighting inflate predicted probabilities, up to 26x for undersampling and 40x for
balanced logistic regression. Rank-based metrics cannot detect this, but it is why precision at a
fixed 0.5 threshold ranges from 0.88 to 0.05 across the table.

## Requirements

- Python 3.12
- ~2 GB disk space (dataset is 144 MB, model artifacts ~2 MB)
- CPU only; no GPU required

## Setup

```bash
git clone https://github.com/Pranay847/fraud-anomaly-detector.git
cd fraud-anomaly-detector
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate`
otherwise), then:

```bash
pip install -r requirements.txt
```

For a byte-exact environment, use `requirements-lock.txt` instead.

## Getting the data

```bash
python -m src.download_data
```

The dataset is not committed. This downloads `creditcard.csv` to `data/` from a Hugging Face mirror
pinned to a commit SHA, and verifies the file size, SHA-256, shape, column names and fraud count
before accepting it. The download is skipped if a verified copy already exists.

Kaggle is the canonical source but requires an account. The mirror is byte-identical
(150,828,752 bytes, `sha256 76274b69...12de5`). To use a manual download instead, place
`creditcard.csv` in `data/` and the verification step will validate it.

## Usage

Train a single model:

```bash
python -m src.train --model baseline
```

Valid values are `baseline`, `xgb-weighted`, `xgb-smote`, `xgb-under`, `xgb-tuned`, `autoencoder`,
and `all`. Each run writes metrics to `reports/metrics/<name>.json` and an artifact to `models/`.

Train everything and regenerate all plots and `results.md`:

```bash
python -m src.train --model all
```

Rebuild plots and `results.md` from existing metrics without retraining:

```bash
python -m src.train --report
```

Score the test set:

```bash
python -m src.train --final-test
```

This runs the best validation model against the test set once and then refuses to run again, since
repeated test-set evaluation turns a held-out set into a second validation set. Pass `--force` to
override.

Run the tests:

```bash
pytest tests/ -q
```

### Runtimes

Measured on 14 CPU cores. `--model all` takes about 26 minutes total.

| Model | Time |
|---|---:|
| `baseline` | 22s |
| `xgb-weighted` | 31s |
| `xgb-smote` | 64s |
| `xgb-under` | 18s |
| `xgb-tuned` | 10m 18s |
| `autoencoder` | 13m 24s |

Useful flags for faster runs: `--skip-cv` skips cross-validation, `--n-iter` sets the number of
hyperparameter search iterations (default 25), and `--ae-epochs` caps autoencoder training
(default 100). `--search-jobs` and `--model-jobs` control parallelism; their product should stay
at or below the core count.

## Project structure

```
src/
  download_data.py   dataset fetch and integrity verification
  data.py            loading, stratified splits, cross-validation folds
  features.py        log-amount, hour-of-day, V-component interactions
  baseline.py        class-weighted logistic regression
  supervised.py      XGBoost variants and hyperparameter search
  autoencoder.py     PyTorch autoencoder trained on legitimate transactions
  evaluate.py        metrics, plots, cost analysis
  train.py           CLI entry point
notebooks/
  01_eda.ipynb              exploratory analysis
  02_error_analysis.ipynb   analysis of misclassified transactions
tests/               split integrity, leakage guards, metric correctness
reports/             generated plots, metrics JSON, cost curves
data/                dataset (gitignored)
models/              trained artifacts (gitignored)
```

## Method

### Splitting

Stratified 70/15/15 in two stages, seeded at 42:

| Split | Rows | Fraud | Fraud rate |
|---|---:|---:|---:|
| train | 199,478 | 344 | 0.172% |
| val | 42,607 | 74 | 0.174% |
| test | 42,722 | 74 | 0.173% |

Stratification matters at this positive rate: splitting 492 positives three ways without it makes
the validation fraud count vary substantially, and every metric moves with it. Because 74 positives
is a thin basis for comparison, the baseline and the three XGBoost variants are also scored by
5-fold stratified cross-validation, reported as mean ± std.

### Resampling

Every resampler is a step inside an `imblearn.pipeline.Pipeline`:

```python
ImbPipeline([("features", FraudFeatures()),
             ("resample", SMOTE(random_state=42)),
             ("clf", XGBClassifier(...))])
```

imblearn's pipeline applies `fit_resample` during `fit` and skips it during `predict`, so under
cross-validation no fold is scored by a model that saw synthetic neighbors derived from its own
held-out rows. Resampling before splitting is a common source of inflated results on this dataset.

The same applies to feature engineering: interaction terms are built from the components most
correlated with the label, and that correlation is computed inside `fit()` from training rows only.

### Features

`V1`–`V28` are PCA components, so no domain feature engineering is possible on them. The two raw
columns are used as follows:

- `Amount` is log-transformed (it is heavily right-skewed)
- `Time` becomes hour-of-day, and the raw column is dropped. Across a two-day window raw `Time` is
  close to a row index, and a tree can split on "day two", which does not generalize
- Interaction terms are products of the four components most correlated with the label
  (V17, V14, V12, V10)

### Autoencoder

Trained only on legitimate training transactions, so it models normal behavior; per-row
reconstruction error becomes the anomaly score. The feature transformer and scaler are fit on the
same legitimate subset, so no fraud information reaches the model.

Early stopping monitors reconstruction loss on the legitimate portion of validation rather than all
of it. Increasing fraud reconstruction error is the desired behavior, so a combined loss would
penalize the model for working correctly.

### Threshold selection

The cost model assumes a false positive costs $5 in analyst review time and a missed fraud costs the
transaction amount. Under those assumptions the cost-minimizing threshold is 0.31, but the cost
surface is nearly flat between 0.03 and 0.99, so the minimum is not meaningfully better than nearby
points.

The threshold shipped is 0.03, chosen by a stated rule: among thresholds within 5% of minimum cost,
take the one catching the most fraud. It costs 3.8% more than the minimum and catches 4 more frauds,
flagging 22 of 42,607 validation transactions (0.052%). The rationale is that transaction value
undercounts the true cost of a miss, which also includes chargeback fees and support handling, and
the median fraud here is only $9.25.

`results.md` includes a sensitivity table showing how the recommended threshold responds to
different cost assumptions; it ranges from 0.01 to 0.99, with recall from 79.7% to 89.2%.

## Notebooks

Both are committed with outputs, so they render on GitHub without running anything.

- [01_eda.ipynb](notebooks/01_eda.ipynb) covers class imbalance, amount and time distributions by
  class, and component correlations. Findings: fraud skews smaller at the median ($9.25 against
  $22.00), the 2am fraud rate is 1.71% against a 0.173% baseline, and V17/V14/V12/V10 separate the
  classes most strongly.
- [02_error_analysis.ipynb](notebooks/02_error_analysis.ipynb) examines the misclassified
  transactions. On all eight of the most discriminative components, missed frauds sit closer to the
  legitimate mean than to the caught-fraud mean, meaning they are not a distinguishable fraud
  subtype in this feature space. Of the 22 false positives, 6 score above 0.90.

## Limitations

**PCA features.** The original transaction details were PCA-transformed before release for privacy.
This rules out domain feature engineering and means model explanations cannot be interpreted by a
fraud analyst. The error analysis indicates this is the binding constraint on performance rather
than the choice of model.

**Duplicate rows.** The dataset contains 1,081 exact duplicates, kept here for comparability with
published results on this dataset. As a result 208 of 42,607 validation rows (0.488%) have an exact
duplicate in training, including 2 of the 74 validation frauds, which inflates the reported metrics
slightly.

**No temporal validation.** The data covers two days from one institution in 2013. There is not
enough time span to test for drift, and fraud patterns change as attackers adapt.

**Small positive counts.** Each held-out split contains 74 frauds, so single-split estimates are
noisy. Cross-validated results are reported alongside point estimates for this reason.

**Cost assumptions are illustrative.** The $5 review cost is a stated assumption, not a measured
figure, which is what the sensitivity analysis is for.

## Data source

Machine Learning Group, Université Libre de Bruxelles. Transactions made by European cardholders in
September 2013, released with `V1`–`V28` PCA-transformed for confidentiality. Available on
[Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
