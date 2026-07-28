# Fraud Detection on Imbalanced Transaction Data

Comparing a class-weighted logistic regression, XGBoost under three different
imbalance strategies, and a PyTorch autoencoder trained only on legitimate
transactions — on 284,807 credit card transactions of which 492 (0.172%) are
fraud.

The point of the project is not to produce the highest number. It is to decide
which model to deploy, at which threshold, and to be able to say why.

---

## Why accuracy is the wrong metric

99.827% of these transactions are legitimate. A model that returns "not fraud"
for every input scores **99.827% accuracy** and catches **zero** of the 492
frauds. Accuracy cannot tell that model apart from a useful one, so it is not
reported anywhere in this project.

The primary metric throughout is **PR-AUC** (average precision), which is
computed only from how the model ranks the positive class and is unmoved by the
enormous pile of true negatives. ROC-AUC is reported as a secondary — every
model here scores above 0.98 on it, which is precisely the problem: with 284,315
true negatives the false-positive rate stays tiny no matter how many legitimate
transactions get flagged, so ROC-AUC compresses genuinely different models into
a narrow band that looks uniformly excellent.

## Results

Validation set (42,607 transactions, 74 fraud). Full detail in [results.md](results.md).

| Model | PR-AUC | ROC-AUC | Precision @ 0.5 | Recall @ 0.5 | Over-prediction |
|---|---:|---:|---:|---:|---:|
| **XGBoost (tuned, class-weighted)** | **0.8760** | 0.9836 | 0.8824 | 0.8108 | 1.0x |
| XGBoost + SMOTE | 0.8577 | 0.9899 | 0.4507 | 0.8649 | 5.0x |
| XGBoost + class weighting | 0.8552 | 0.9907 | 0.8333 | 0.8108 | 1.5x |
| XGBoost + random undersampling | 0.8073 | 0.9897 | 0.0604 | 0.8919 | 25.9x |
| Logistic Regression (class-weighted) | 0.7636 | 0.9852 | 0.0540 | 0.9189 | 39.6x |
| Autoencoder (reconstruction error) | 0.5008 | 0.9411 | 0.2710 | 0.7838 | n/a |

**Final test set — evaluated exactly once, on the single model selected by validation PR-AUC:**

| Metric | Value |
|---|---:|
| PR-AUC | **0.8362** |
| Recall @ threshold 0.03 | 0.8243 (61 of 74 frauds) |
| Precision @ threshold 0.03 | 0.7011 (26 false positives) |

Validation said 0.8760, the test set said 0.8362. The 0.04 gap is what sampling
noise on 74 positives looks like — the cross-validated fold spread measured on
this same pipeline was ±0.047 — but the test number is the honest one, and the
validation figure was the optimistic end of the range. That is precisely why
the test set is opened once.

### Four things the table says

**Look at the ROC-AUC column, then look at the autoencoder row.** It scores
**0.9411 ROC-AUC** — which sounds like a strong model — and **0.5008 PR-AUC**.
Every model here clears 0.98 ROC-AUC while their PR-AUCs range from 0.50 to
0.88. If you selected on ROC-AUC you would call these models roughly
equivalent. They are not. This one row is the entire argument for the metric
choice, made concrete.

**SMOTE did not beat class weighting.** 0.8577 against 0.8552, with a
cross-validated fold spread of ±0.05 — the gap is twenty times smaller than the
noise. Treating that as a win would be reading the validation set. The
one-standard-error rule was applied instead: among variants within one standard
error of the best, take the simplest, which is class weighting (it invents no
data and adds no hyperparameter). Undersampling was the clear loser at 0.8073 —
discarding 199,000 legitimate transactions costs real information, and its fold
spread (±0.087) is the widest in the table.

**Resampling wrecks calibration, and rank metrics cannot see it.** The
over-prediction column is the mean predicted probability divided by the actual
fraud rate. The tuned model is calibrated almost exactly (1.0x); undersampling
over-predicts by **26x** and balanced logistic regression by **40x**. PR-AUC and
ROC-AUC are rank-based and are blind to this — but it is why precision at a
fixed 0.5 threshold swings from 0.88 to 0.05 down the table. A threshold of
"0.5" does not mean the same thing in any two of these rows, which matters the
moment you deploy one.

**Tuning moved PR-AUC up and ROC-AUC down** — 0.8552 → 0.8760 on PR-AUC, 0.9907
→ 0.9836 on ROC-AUC. The search optimised `average_precision`, and it traded
away the metric it was not asked about. Small, but a clean demonstration that
the scoring choice inside the search is a real decision.



---

## Data

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
(ULB) — two days of European cardholder transactions from September 2013.

`V1`–`V28` are principal components. The original features were transaction
details a bank cannot publish, so they were PCA-transformed before release.
This is worth stating plainly because it is a **real production constraint, not
a defect of the exercise**: fraud systems routinely operate on features that are
hashed, embedded, or otherwise stripped of meaning for privacy reasons. Two
things follow. There is no domain feature engineering to do — no "distance from
home", no "merchant category" — and no model explanation here will ever be
readable by a fraud analyst. `Time` and `Amount` are the only raw columns.

### What the EDA changed

Full detail in [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).

**Fraud is not the large-value tail.** Median fraud is **$9.25** against **$22.00**
for legitimate transactions — fraud skews *smaller* at the median and only leads
on the mean ($122 vs $88). The intuition that fraud means big-ticket purchases
is wrong here, and it has a direct consequence: since a missed fraud costs
roughly the transaction amount, and the median fraud is under ten dollars, the
economics of chasing every case are much weaker than they first appear. This is
what makes the threshold analysis the most interesting part of the project.

**Hour-of-day carries real signal.** The 2am fraud rate is **1.71%** against a
0.173% baseline — roughly ten times over. Volume collapses overnight while fraud
does not. `Time` is therefore converted to hour-of-day and the raw column
dropped: across a two-day window raw `Time` is nearly a row index, and a tree
will happily split on "day two", which generalises to nothing.

**V17, V14, V12 and V10 separate the classes most strongly** (correlations of
-0.33 to -0.22 with the label). The interaction terms are built from these four.

**There are 1,081 exact duplicate rows.** They are kept rather than dropped, for
comparability with the published results this dataset is usually benchmarked
against — but keeping them has a measurable cost, quantified under Limitations
below.

---

## Method

### Splitting

Stratified 70/15/15, in two stages, seeded:

```python
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.176, stratify=y_trainval, random_state=42)
```

| Split | Rows | Fraud | Fraud rate |
|---|---:|---:|---:|
| train | 199,478 | 344 | 0.172% |
| val | 42,607 | 74 | 0.174% |
| test | 42,722 | 74 | 0.173% |

`stratify` is the load-bearing argument. Split 492 positives three ways without
it and the validation fraud count becomes a lottery — and since every metric
here is computed from those 74 positives, the metric moves with the draw.

Because 74 positives is a thin basis for a decision, the baseline and all three
XGBoost variants are additionally scored by **5-fold stratified cross-validation
on the training set**, reported as mean ± std. The fold-to-fold spread turns out
to be ±0.05 PR-AUC, which is larger than most of the gaps between models — and
that fact drives the model selection below.

### Resampling never touches held-out data

Every resampler is a step *inside* an `imblearn.pipeline.Pipeline`:

```python
ImbPipeline([("features", FraudFeatures()),
             ("resample", SMOTE(random_state=42)),
             ("clf", XGBClassifier(...))])
```

This is the structural guarantee, not a promise: imblearn's pipeline calls
`fit_resample` during `fit` and skips it entirely at `predict` time. Under
cross-validation, no fold is ever scored by a model that saw a synthetic
neighbour interpolated from its own held-out rows. Resampling before splitting
is the classic way to inflate results on this dataset, and the pipeline is what
makes avoiding it verifiable rather than aspirational.

The same applies to feature engineering. The interaction terms are built from
the components most correlated with the label, and that correlation is computed
inside `fit()` — so the choice of features is made per-fold, from training rows
only.

### The autoencoder

Trained on **legitimate training transactions only**, so it learns to
reconstruct normal behaviour and nothing else. Fraud, being off that manifold,
should reconstruct poorly, and per-row reconstruction error becomes an anomaly
score produced without a single fraud label.

One deviation from the standard recipe: early stopping monitors reconstruction
loss on the **legitimate portion** of validation, not all of it. Including fraud
makes the stopping signal ambiguous — rising fraud reconstruction error is the
outcome we want, so a combined loss would penalise the model for succeeding.

---

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt
python -m src.download_data
pytest tests/ -q
python -m src.train --model all
```

`src/download_data.py` fetches the CSV from a Hugging Face mirror pinned to a
commit SHA, and refuses it unless the size, SHA-256, shape, column names and
fraud count all match the published dataset. Kaggle is the canonical source but
requires an account; the mirror is byte-identical
(`sha256 76274b69…12de5`, 150,828,752 bytes).

```
src/
  download_data.py   fetch + verify the dataset
  data.py            loading, stratified splits, CV
  features.py        log-amount, hour-of-day, V-interactions
  baseline.py        class-weighted logistic regression
  supervised.py      XGBoost + three resampling variants + search
  autoencoder.py     PyTorch autoencoder, trained on legitimate rows only
  evaluate.py        metrics, plots, cost analysis
  train.py           CLI: --model, --report, --final-test
```

---

## Which model I would deploy, and at what threshold

**Deploy the tuned, class-weighted XGBoost at a threshold of 0.03.**

On validation that catches **65 of 74 frauds (87.8% recall)** at 74.7%
precision, flagging **22 transactions out of 42,607** — a 0.052% flag rate, or
about one review per 1,900 transactions. On the held-out test set the same
configuration caught 61 of 74 (82.4%) with 26 false positives.

### Why 0.03 and not the cost-minimising threshold

The stated cost model — $5 of analyst time per false positive, the transaction
amount per missed fraud — has its minimum at threshold 0.31. I am not
recommending it, for a reason visible in
[reports/cost_vs_threshold.png](reports/cost_vs_threshold.png): **the cost
surface is almost flat**. Thresholds from 0.03 to 0.99 all sit within a few
percent of the minimum. Taking the argmin of a curve that flat means reporting
false precision about an invented $5 assumption.

The rule applied instead is stated up front: *among thresholds within 5% of the
minimum cost, take the one that catches the most fraud.* That is threshold 0.03
— it costs 3.8% more than the minimum and buys 4 more frauds caught. The
premium is worth paying because the cost model cannot see chargeback fees,
support handling, or the customer who closes their account. Transaction value
systematically undercounts what a miss costs, and on this dataset the median
fraud is $9.25, so that omission dominates.

The [sensitivity table](results.md#sensitivity-to-the-cost-assumptions) shows
how far this moves: across plausible cost assumptions the recommended threshold
ranges from 0.01 to 0.99 and recall from 79.7% to 89.2%. **The shape of that
response is the transferable result, not the specific number** — anyone
deploying this would substitute their own review cost and chargeback fee and
re-read the table.

## What the errors say

Full analysis in
[notebooks/02_error_analysis.ipynb](notebooks/02_error_analysis.ipynb).

**The missed fraud is not a subtype the model overlooks — it is fraud that
looks legitimate.** On all eight of the components that separate the classes
most strongly, the missed frauds sit closer to the *legitimate* mean than to
the caught-fraud mean. V14 is the clearest: legitimate averages 0.016, caught
fraud averages **-7.82**, missed fraud averages **-0.93**. In the feature space
available to the model, those nine transactions are ordinary spending.

That decides the next step. **More trees, deeper trees, and another week of
hyperparameter search cannot recover them** — no amount of tuning separates
points that overlap. The next real gain is different features, not a better
model: velocity, deviation from the cardholder's own history, merchant risk,
device reuse. None of them exist here, and the PCA transform means they cannot
be reconstructed either.

Two smaller notes. The residual is **not** just the cheap tail — missed fraud
averages $177.68 against $149.87 for caught fraud, so the misses skew slightly
*more* expensive and are a genuine blind spot rather than a deliberate economy.
And 6 of the 22 false positives score above 0.90: the model is confidently
wrong about those, and with a queue this small they are worth pulling by hand,
because confident errors are where label noise hides.

## Limitations

**The PCA features cap what is achievable and remove all interpretability.**
No domain feature engineering is possible, and no explanation this model
produces can be handed to a fraud analyst. The error analysis suggests this is
the binding constraint on performance, not the model class.

**1,081 exact duplicate rows leak across the split.** They are kept for
comparability with published results on this dataset, but the cost is
measurable: **208 of 42,607 validation rows (0.488%) have an exact duplicate in
training, including 2 of the 74 validation frauds.** That inflates the reported
metrics slightly. Deduplicating first would be the more defensible choice for a
production model and would change the numbers here by a small but non-zero
amount.

**Two days of transactions from 2013, one geography, one institution.** There
is no temporal drift testing, because there is not enough time span to do it.
Fraud patterns move continuously as attackers adapt, and a model trained on
September 2013 would degrade against November 2013 in ways nothing here can
measure. A production version needs scheduled retraining and drift monitoring
on the score distribution.

**74 positives in each held-out split** is a thin basis for any single
estimate. This is why cross-validation is reported alongside the point
estimates, and why the ±0.05 fold spread is quoted whenever two models are
compared.

**The cost model is invented.** $5 per review is a plausible guess, not a
measured figure, which is why the sensitivity table exists.

## With more time or data

- **Deduplicate and re-run**, and report the honest delta rather than the
  flattering number.
- **Velocity and history features** — the single highest-value change, and the
  one the error analysis points at directly. Requires transaction data this
  dataset does not contain.
- **Calibrate the probabilities explicitly** (isotonic or Platt scaling on a
  held-out fold) so the threshold means the same thing after every retrain.
- **Temporal validation** — train on early transactions, test on later ones,
  which measures the thing a random split cannot.
- **Cost-sensitive training** rather than post-hoc thresholding: weight each
  example by its transaction amount so the model optimises money rather than
  counts.
- **An ensemble of the tuned XGBoost and the autoencoder.** The autoencoder is
  much worse alone, but it is wrong in different ways, and it needs no labels —
  which is exactly the property you want against fraud patterns that have not
  been labelled yet.

## Note on the autoencoder

It lost, decisively: 0.5008 PR-AUC against 0.8760. That is the honest result and
it is not a surprise — a fully labelled dataset plays to the strengths of
supervised learning, and asking a model to find fraud without ever showing it
one is a handicap.

The comparison is still worth making, for two reasons. Its PR-AUC of 0.50 is
roughly **290 times better than the 0.0017 a random classifier achieves**, and
it got there without a single fraud label. And the labels this project relies
on are a luxury: in production they arrive weeks late, incomplete, and never at
all for the attack that started this morning. A detector that needs no positive
labels degrades differently from one that does, which is an argument for
running it *alongside* the supervised model rather than instead of it.

---

## Resume bullets

Measured numbers only.

- Built an imbalanced classification pipeline on 284K credit card transactions
  (0.172% fraud rate), using stratified train/validation/test splits and PR-AUC
  as the primary metric; demonstrated the failure mode concretely by showing a
  model with 0.94 ROC-AUC scoring 0.50 PR-AUC on the same data.
- Compared six approaches — class-weighted logistic regression, XGBoost with
  class weighting / SMOTE / undersampling, a tuned XGBoost, and a PyTorch
  autoencoder trained only on legitimate transactions — improving PR-AUC from
  0.764 (baseline) to 0.876 (validation), 0.836 on a test set opened once.
- Applied the one-standard-error rule to reject a 0.0025 PR-AUC "win" for SMOTE
  that sat inside a ±0.05 cross-validation spread, and identified that
  aggressive resampling inflated predicted probabilities by up to 26x — making
  a fixed decision threshold incomparable across models.
- Translated model output into a deployment recommendation via an explicit cost
  model, choosing a threshold catching 87.8% of fraud at a 0.05% flag rate, and
  showed through sensitivity analysis that the recommended threshold ranges from
  0.01 to 0.99 across plausible cost assumptions.

