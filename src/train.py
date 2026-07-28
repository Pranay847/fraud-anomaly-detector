"""CLI entrypoint. Runs any model, writes metrics, produces the report artifacts.

    python -m src.train --model baseline
    python -m src.train --model all
    python -m src.train --report            # rebuild plots + results.md
    python -m src.train --final-test        # score the test set, exactly once

Every run writes `reports/metrics/<name>.json` and caches the validation scores
to `models/scores/`, so the comparison table and the plots are generated from
measured numbers rather than transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from . import baseline, supervised
from .data import RANDOM_STATE, get_cv, load_raw, make_splits, set_seeds
from .evaluate import (
    METRICS_DIR,
    REPORTS_DIR,
    comparison_frame,
    cost_curve,
    evaluate_scores,
    load_all_metrics,
    load_metrics,
    pick_threshold,
    plot_comparison_table,
    plot_confusion_matrices,
    plot_cost_curve,
    plot_pr_curves,
    save_metrics,
)
from .features import save_top_v

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SCORES_DIR = MODELS_DIR / "scores"

SUPERVISED_MODELS = ["baseline", "xgb-weighted", "xgb-smote", "xgb-under"]
ALL_MODELS = SUPERVISED_MODELS + ["xgb-tuned", "autoencoder"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _save_scores(name: str, split: str, y_true, scores) -> None:
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(SCORES_DIR / f"{name}_{split}.npz", y_true=np.asarray(y_true), scores=np.asarray(scores))


def load_scores(name: str, split: str = "val"):
    path = SCORES_DIR / f"{name}_{split}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data["y_true"], data["scores"]


def _cv_pr_auc(estimator, X, y, *, n_splits: int, n_jobs: int) -> dict:
    """Mean ± std PR-AUC across stratified folds.

    A single train/val split on 74 validation frauds is a noisy measurement.
    Cross-validating on the training set says whether a gap between two models
    is real or is one lucky partition.
    """
    scores = cross_val_score(
        estimator,
        X,
        y,
        cv=get_cv(n_splits),
        scoring="average_precision",
        n_jobs=n_jobs,
    )
    return {
        "cv_pr_auc_mean": float(scores.mean()),
        "cv_pr_auc_std": float(scores.std()),
        "cv_pr_auc_folds": [float(s) for s in scores],
        "cv_pr_auc_str": f"{scores.mean():.4f} ± {scores.std():.4f}",
    }


def _best_variant_from_metrics() -> str:
    """Pick which resampling strategy to tune, based on what actually won."""
    best, best_score = "weighted", -1.0
    for variant in supervised.VARIANTS:
        try:
            m = load_metrics(f"xgb-{variant}")
        except FileNotFoundError:
            continue
        if m["pr_auc"] > best_score:
            best, best_score = variant, m["pr_auc"]
    return best


# ---------------------------------------------------------------------------
# per-model training
# ---------------------------------------------------------------------------


def run_model(name: str, splits, args) -> dict:
    print(f"\n=== {name} ===")
    started = time.perf_counter()
    X_train, y_train = splits.X_train, splits.y_train
    X_val, y_val = splits.X_val, splits.y_val

    if name == "autoencoder":
        # Imported here so the supervised models don't drag torch in.
        from .autoencoder import AutoencoderDetector, plot_reconstruction_error

        detector = AutoencoderDetector(
            epochs=args.ae_epochs, patience=args.ae_patience, random_state=RANDOM_STATE
        )
        detector.fit(X_train, y_train, X_val, y_val)
        scores = detector.score(X_val)
        detector.save(MODELS_DIR / "autoencoder.pt")

        metrics = evaluate_scores(
            y_val,
            scores,
            model_name=name,
            thresholds=[float(np.quantile(scores, q)) for q in (0.99, 0.995, 0.999)],
            extra={
                "display_name": "Autoencoder (reconstruction error)",
                "epochs_trained": detector.epochs_trained_,
                "best_val_recon_loss": detector.best_val_loss_,
                "n_train_legit": int((np.asarray(y_train) == 0).sum()),
                "note": (
                    "Trained on legitimate training transactions only; thresholds are "
                    "score quantiles because reconstruction error is not a probability."
                ),
            },
        )
        plot_reconstruction_error(y_val, scores, REPORTS_DIR / "recon_error_dist.png")

    else:
        if name == "baseline":
            estimator = baseline.build()
            display = baseline.DISPLAY_NAME
        elif name == "xgb-tuned":
            variant = args.tune_variant or _best_variant_from_metrics()
            print(f"  tuning the '{variant}' variant (best PR-AUC among the resampling runs)")
            search = supervised.build_search(
                variant,
                y_train,
                n_iter=args.n_iter,
                n_splits=args.cv_folds,
                search_jobs=args.search_jobs,
                model_jobs=args.model_jobs,
            )
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
            display = f"{supervised.DISPLAY_NAMES['tuned']} — {variant}"
            print(f"  best CV PR-AUC {search.best_score_:.4f}")
            print(f"  best params    {search.best_params_}")
        else:
            variant = name.replace("xgb-", "")
            # Threads per model stay low here: cross-validation runs
            # `search_jobs` folds at once, and search_jobs * model_jobs threads
            # is the budget that has to fit inside the core count.
            estimator = supervised.build(variant, y_train=y_train, n_jobs=args.model_jobs)
            display = supervised.DISPLAY_NAMES[variant]

        cv_info = {}
        if name != "xgb-tuned" and not args.skip_cv:
            print(f"  cross-validating ({args.cv_folds}-fold, scoring=average_precision)...")
            cv_info = _cv_pr_auc(
                estimator, X_train, y_train, n_splits=args.cv_folds, n_jobs=args.search_jobs
            )
            print(f"  CV PR-AUC {cv_info['cv_pr_auc_str']}")

        if name != "xgb-tuned":  # the search already refit on the full training set
            # Nothing runs in parallel with the final fit, so give it the cores.
            if hasattr(estimator, "named_steps") and "clf" in estimator.named_steps:
                estimator.set_params(clf__n_jobs=args.model_jobs * args.search_jobs)
            estimator.fit(X_train, y_train)

        scores = estimator.predict_proba(X_val)[:, 1]
        joblib.dump(estimator, MODELS_DIR / f"{name}.joblib")

        extra = {"display_name": display, **cv_info}
        if name == "xgb-tuned":
            extra["best_params"] = {k: str(v) for k, v in search.best_params_.items()}
            extra["tuned_variant"] = variant
            extra["search_best_cv_pr_auc"] = float(search.best_score_)
            extra["cv_pr_auc_str"] = f"{search.best_score_:.4f} (search best)"
        if hasattr(estimator, "named_steps") and "features" in estimator.named_steps:
            top_v = getattr(estimator.named_steps["features"], "top_v_", [])
            if top_v:
                extra["top_v_features"] = top_v
                save_top_v(top_v)

        metrics = evaluate_scores(y_val, scores, model_name=name, extra=extra)

    metrics["train_seconds"] = round(time.perf_counter() - started, 1)
    save_metrics(metrics, name)
    _save_scores(name, "val", y_val, scores)

    print(f"  PR-AUC  {metrics['pr_auc']:.4f}   ROC-AUC {metrics['roc_auc']:.4f}")
    print(f"  ({metrics['train_seconds']}s)")
    return metrics


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def build_report(splits) -> dict:
    """Regenerate every plot and results.md from the saved metrics and scores."""
    metrics = load_all_metrics()
    metrics = [m for m in metrics if m.get("split") == "val"]
    if not metrics:
        raise RuntimeError("no validation metrics found — run some models first")

    curves = {}
    for m in metrics:
        cached = load_scores(m["model"], "val")
        if cached is not None:
            curves[m.get("display_name", m["model"])] = cached
    plot_pr_curves(curves)

    table = comparison_frame(metrics)
    plot_comparison_table(table)

    best = max(metrics, key=lambda m: m["pr_auc"])
    best_name = best["model"]
    y_true, scores = load_scores(best_name, "val")

    is_probability = best_name != "autoencoder"
    thresholds = (
        [0.3, 0.5, 0.7]
        if is_probability
        else [float(np.quantile(scores, q)) for q in (0.99, 0.995, 0.999)]
    )
    plot_confusion_matrices(
        y_true, scores, thresholds, model_name=best.get("display_name", best_name)
    )

    amounts = splits.X_val["Amount"].to_numpy()
    costs = cost_curve(y_true, scores, amounts)
    chosen = pick_threshold(costs)
    plot_cost_curve(costs, chosen)
    costs.to_csv(REPORTS_DIR / "cost_curve.csv", index=False)

    report = {
        "best_model": best_name,
        "best_display_name": best.get("display_name", best_name),
        "chosen_threshold": chosen,
        "comparison_table": table.to_dict(orient="records"),
    }
    (REPORTS_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    write_results_md(splits, metrics, table, best, chosen, costs)
    print(f"\nBest validation model: {report['best_display_name']} (PR-AUC {best['pr_auc']:.4f})")
    print(f"Cost-minimising threshold: {chosen['threshold']:.4f}")
    return report


def write_results_md(splits, metrics, table, best, chosen, costs) -> Path:
    summary = splits.summary()
    lines = ["# Results", ""]
    lines.append(
        "All numbers below are produced by `python -m src.train --model all` and read "
        "back from `reports/metrics/*.json`. Nothing here is typed in by hand."
    )
    lines += ["", "## Data split", "", "Stratified, so the fraud rate is preserved everywhere.", ""]
    lines.append("| Split | Rows | Fraud | Fraud rate |")
    lines.append("|---|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['split']} | {int(r['rows']):,} | {int(r['fraud'])} | {r['fraud_rate']:.3%} |"
        )

    lines += ["", "## Model comparison (validation set)", ""]
    header = "| Model | PR-AUC | ROC-AUC | Precision | Recall | CV PR-AUC (5-fold, train) |"
    lines += [header, "|---|---:|---:|---:|---:|---|"]
    for m in sorted(metrics, key=lambda x: -x["pr_auc"]):
        at = m["thresholds"][1] if len(m["thresholds"]) > 1 else m["thresholds"][0]
        lines.append(
            f"| {m.get('display_name', m['model'])} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} "
            f"| {at['precision']:.4f} | {at['recall']:.4f} | {m.get('cv_pr_auc_str', '—')} |"
        )
    lines += [
        "",
        "Precision and recall are quoted at threshold 0.5 for the probability models. "
        "The autoencoder's score is a reconstruction error, not a probability, so its "
        "row is quoted at the 99.5th percentile of its own score distribution — the "
        "comparable operating point, since 0.5 is meaningless on that scale.",
    ]

    lines += ["", "## Threshold choice", ""]
    lines.append(
        f"Cost model: reviewing a false positive costs **$5.00** of analyst time; missing "
        f"a fraud costs the **full transaction amount**. Sweeping the threshold on the "
        f"validation set and minimising total cost gives:"
    )
    lines += [
        "",
        f"- **Chosen threshold: {chosen['threshold']:.4f}**",
        f"- Catches {int(chosen['tp'])} of {int(chosen['tp'] + chosen['fn'])} frauds "
        f"({chosen['recall']:.1%} recall) at {chosen['precision']:.1%} precision",
        f"- Flags {int(chosen['fp'])} legitimate transactions → ${chosen['review_cost']:,.2f} in review time",
        f"- Lets ${chosen['fraud_loss']:,.2f} of fraud through",
        f"- **Total expected cost: ${chosen['total_cost']:,.2f}** over "
        f"{len(splits.y_val):,} validation transactions",
        "",
        "See `reports/cost_vs_threshold.png` for the full curve and "
        "`reports/cost_curve.csv` for the underlying numbers.",
    ]

    test_metrics = [
        json.loads(p.read_text())
        for p in sorted(METRICS_DIR.glob("*_test.json"))
    ]
    lines += ["", "## Final test set result", ""]
    if test_metrics:
        t = test_metrics[0]
        lines.append(
            f"The test set was held out for the entire project and evaluated **exactly once**, "
            f"on `{t.get('display_name', t['model'])}` — the single best model by validation PR-AUC."
        )
        lines += ["", "| Metric | Value |", "|---|---:|"]
        at = t["thresholds"][0]
        lines += [
            f"| PR-AUC | {t['pr_auc']:.4f} |",
            f"| ROC-AUC | {t['roc_auc']:.4f} |",
            f"| Threshold | {at['threshold']:.4f} |",
            f"| Precision @ threshold | {at['precision']:.4f} |",
            f"| Recall @ threshold | {at['recall']:.4f} |",
            f"| Frauds caught | {at['tp']} of {at['tp'] + at['fn']} |",
            f"| False positives | {at['fp']:,} |",
        ]
    else:
        lines.append(
            "_Not yet run. `python -m src.train --final-test` scores the test set once, "
            "on the best validation model._"
        )

    path = ROOT / "results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def run_final_test(splits, args) -> dict:
    """Score the test set. Once. On one model."""
    report_path = REPORTS_DIR / "report.json"
    if not report_path.exists():
        raise RuntimeError("run `--report` first so the best validation model is known")
    report = json.loads(report_path.read_text())
    name = args.test_model or report["best_model"]
    threshold = float(report["chosen_threshold"]["threshold"])

    existing = list(METRICS_DIR.glob("*_test.json"))
    if existing and not args.force:
        raise RuntimeError(
            f"the test set has already been evaluated ({existing[0].name}). "
            "That was the one shot. Pass --force only if you genuinely intend to "
            "burn it again, and say so in the writeup."
        )

    print(f"\n=== FINAL TEST: {name} @ threshold {threshold:.4f} ===")
    if name == "autoencoder":
        raise RuntimeError(
            "the autoencoder artifact needs refitting to score the test set; "
            "re-run `--model autoencoder` and extend this branch if it wins"
        )

    estimator = joblib.load(MODELS_DIR / f"{name}.joblib")
    scores = estimator.predict_proba(splits.X_test)[:, 1]
    metrics = evaluate_scores(
        splits.y_test,
        scores,
        model_name=name,
        split="test",
        thresholds=[threshold],
        extra={
            "display_name": report["best_display_name"],
            "note": "Test set evaluated exactly once, on the model selected by validation PR-AUC.",
        },
    )
    save_metrics(metrics, f"{name}_test")
    _save_scores(name, "test", splits.y_test, scores)

    at = metrics["thresholds"][0]
    print(f"  PR-AUC   {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC  {metrics['roc_auc']:.4f}")
    print(f"  recall   {at['recall']:.4f}  ({at['tp']} of {at['tp'] + at['fn']} frauds)")
    print(f"  precision {at['precision']:.4f}  ({at['fp']:,} false positives)")
    return metrics


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=ALL_MODELS + ["all"], help="which model to train")
    parser.add_argument("--report", action="store_true", help="rebuild plots and results.md")
    parser.add_argument("--final-test", action="store_true", help="score the test set (once)")
    parser.add_argument("--test-model", help="override which model the test set is scored with")
    parser.add_argument("--force", action="store_true", help="allow re-running the final test")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true", help="skip cross-validation (faster)")
    parser.add_argument("--n-iter", type=int, default=25, help="RandomizedSearchCV iterations")
    parser.add_argument("--tune-variant", choices=supervised.VARIANTS, help="force a variant to tune")
    parser.add_argument("--search-jobs", type=int, default=5, help="parallel CV folds")
    parser.add_argument("--model-jobs", type=int, default=2, help="threads per XGBoost model")
    parser.add_argument("--ae-epochs", type=int, default=100)
    parser.add_argument("--ae-patience", type=int, default=8)
    args = parser.parse_args()

    if not any([args.model, args.report, args.final_test]):
        parser.error("nothing to do - pass --model, --report or --final-test")

    # The default Windows console encoding mangles the arithmetic symbols this
    # script prints (± in particular).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    set_seeds()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("loading data…")
    df = load_raw()
    splits = make_splits(df)
    print(splits.summary().to_string(index=False))

    if args.model:
        names = ALL_MODELS if args.model == "all" else [args.model]
        for name in names:
            run_model(name, splits, args)

    if args.report or args.model == "all":
        build_report(splits)

    if args.final_test:
        run_final_test(splits, args)
        build_report(splits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
