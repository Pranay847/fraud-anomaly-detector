"""Shared metrics, plots and the cost analysis.

Accuracy is deliberately absent from every table this module produces. On a
dataset that is 99.83% negative, predicting "never fraud" scores 99.83%
accuracy while catching nothing — the metric cannot distinguish a useless model
from a good one. PR-AUC (average precision) is the primary metric throughout;
ROC-AUC is reported as a secondary, because the enormous true-negative count
inflates it into looking good almost regardless of the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # scripts must not try to open windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"

DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7)

# Cost assumptions for the deployment analysis. These are stated, not measured —
# the point is to make the tradeoff explicit rather than to be precise.
FP_COST = 5.00  # analyst time to review one flagged legitimate transaction


def threshold_metrics(y_true, scores, threshold: float) -> dict:
    """Precision / recall / confusion counts at one decision threshold."""
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_scores(
    y_true,
    scores,
    *,
    model_name: str,
    split: str = "val",
    thresholds=DEFAULT_THRESHOLDS,
    extra: dict | None = None,
) -> dict:
    """Full metric bundle for a set of anomaly / probability scores.

    `scores` need only be monotonically related to fraud likelihood, which is
    what lets the autoencoder's reconstruction error be compared against
    calibrated probabilities on the same PR-AUC footing.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    result = {
        "model": model_name,
        "split": split,
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "thresholds": [threshold_metrics(y_true, scores, t) for t in thresholds],
    }
    if extra:
        result.update(extra)
    return result


def save_metrics(metrics: dict, name: str) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{name}.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


def load_metrics(name: str) -> dict:
    return json.loads((METRICS_DIR / f"{name}.json").read_text())


def load_all_metrics() -> list[dict]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    return [json.loads(p.read_text()) for p in sorted(METRICS_DIR.glob("*.json"))]


# --------------------------------------------------------------------------
# Cost analysis
# --------------------------------------------------------------------------


def cost_curve(y_true, scores, amounts, *, fp_cost: float = FP_COST) -> pd.DataFrame:
    """Expected cost across the threshold range.

    A false positive costs a fixed amount of analyst time. A false negative
    costs the transaction that got through — so missing a $2,000 fraud is worth
    400 reviews, and the optimal threshold depends on the amount distribution,
    not just on the counts.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    amounts = np.asarray(amounts)

    candidates = np.quantile(scores, np.linspace(0.90, 0.99999, 400))
    candidates = np.unique(np.concatenate([candidates, np.linspace(0, 1, 101)]))
    candidates = candidates[(candidates >= scores.min()) & (candidates <= scores.max())]
    # A threshold above every score means "flag nothing". Without it the sweep
    # cannot express declining to act, which is the right answer when a model
    # separates nothing and the fraud it would catch is worth less than the
    # reviews it would cost.
    candidates = np.append(candidates, np.nextafter(scores.max(), np.inf))

    rows = []
    for t in candidates:
        flagged = scores >= t
        fp = int(np.sum(flagged & (y_true == 0)))
        missed = (~flagged) & (y_true == 1)
        fn = int(np.sum(missed))
        tp = int(np.sum(flagged & (y_true == 1)))
        fraud_loss = float(amounts[missed].sum())
        rows.append(
            {
                "threshold": float(t),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "review_cost": fp * fp_cost,
                "fraud_loss": fraud_loss,
                "total_cost": fp * fp_cost + fraud_loss,
                "recall": tp / max(tp + fn, 1),
                "precision": tp / max(tp + fp, 1),
            }
        )
    return pd.DataFrame(rows)


def pick_threshold(costs: pd.DataFrame) -> dict:
    """The cost-minimising threshold from a cost curve."""
    best = costs.loc[costs["total_cost"].idxmin()]
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in best.items()}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_pr_curves(curves: dict[str, tuple], path: Path | None = None) -> Path:
    """Overlaid precision-recall curves. `curves` maps label -> (y_true, scores)."""
    path = path or REPORTS_DIR / "pr_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    baseline_rate = None
    for label, (y_true, scores) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, label=f"{label} (PR-AUC = {ap:.3f})", linewidth=1.8)
        baseline_rate = np.mean(y_true)

    if baseline_rate is not None:
        ax.axhline(
            baseline_rate,
            linestyle="--",
            color="grey",
            linewidth=1,
            label=f"random classifier ({baseline_rate:.4f})",
        )

    ax.set_xlabel("Recall (fraction of fraud caught)")
    ax.set_ylabel("Precision (fraction of flags that are fraud)")
    ax.set_title("Precision-Recall curves — validation set")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrices(
    y_true, scores, thresholds, *, model_name: str, path: Path | None = None
) -> Path:
    path = path or REPORTS_DIR / "confusion_matrices.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(thresholds), figsize=(4.2 * len(thresholds), 4))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, thresholds):
        m = threshold_metrics(y_true, scores, t)
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(cm.max(), 2)))
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=11)
        ax.set_xticks([0, 1], ["pred legit", "pred fraud"])
        ax.set_yticks([0, 1], ["true legit", "true fraud"])
        ax.set_title(
            f"threshold {t:g}\nprecision {m['precision']:.3f} · recall {m['recall']:.3f}",
            fontsize=10,
        )
    fig.suptitle(f"{model_name} — confusion matrices (validation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cost_curve(costs: pd.DataFrame, chosen: dict, path: Path | None = None) -> Path:
    path = path or REPORTS_DIR / "cost_vs_threshold.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(costs["threshold"], costs["total_cost"], label="total cost", linewidth=2)
    ax.plot(costs["threshold"], costs["review_cost"], label="analyst review cost", linestyle="--")
    ax.plot(costs["threshold"], costs["fraud_loss"], label="undetected fraud loss", linestyle=":")
    ax.axvline(
        chosen["threshold"],
        color="crimson",
        linewidth=1.2,
        label=f"chosen threshold = {chosen['threshold']:.4f}",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Cost on the validation set (USD)")
    ax.set_title(f"Expected cost vs. threshold (FP = ${FP_COST:.2f}, FN = transaction amount)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def comparison_frame(metrics: list[dict], threshold: float = 0.5) -> pd.DataFrame:
    """The results.md table, built from the saved metric JSONs."""
    rows = []
    for m in metrics:
        at = next(
            (t for t in m["thresholds"] if abs(t["threshold"] - threshold) < 1e-9),
            m["thresholds"][0],
        )
        rows.append(
            {
                "Model": m.get("display_name", m["model"]),
                "PR-AUC": m["pr_auc"],
                "ROC-AUC": m["roc_auc"],
                f"Precision @ {at['threshold']:g}": at["precision"],
                f"Recall @ {at['threshold']:g}": at["recall"],
                "CV PR-AUC (mean ± std)": m.get("cv_pr_auc_str", "—"),
            }
        )
    return pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)


def plot_comparison_table(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or REPORTS_DIR / "comparison_table.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    display = df.copy()
    for col in display.columns:
        if display[col].dtype.kind == "f":
            display[col] = display[col].map(lambda v: f"{v:.4f}")

    fig, ax = plt.subplots(figsize=(2.0 + 1.9 * len(display.columns), 1.0 + 0.45 * len(display)))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for j in range(len(display.columns)):
        table[0, j].set_facecolor("#dfe6ee")
        table[0, j].set_text_props(weight="bold")
    for i in range(1, len(display) + 1):
        if i == 1:
            for j in range(len(display.columns)):
                table[i, j].set_facecolor("#eaf6ea")
    ax.set_title("Model comparison — validation set (sorted by PR-AUC)", pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
