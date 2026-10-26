import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import (
    CV_FOLDS, CV_RESULTS_PATH, FIGURES_DIR, MODEL_DIR, N_ITER_SEARCH,
    RANDOM_STATE, REPORTS_DIR, TUNED_MODEL_PATH, TUNED_PARAMS_PATH,
    TUNING_SCORING,
)
from src.model_utils import build_models, make_pipeline

# Every metric is computed on each CV fold, so the table can show mean +/- std
# rather than a single number from one lucky (or unlucky) split.
SCORERS = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "f1_macro": "f1_macro",
    "precision": "precision",
    "recall": "recall",
    "brier": "neg_brier_score",
}
REFIT_METRIC = {v: k for k, v in SCORERS.items()}[TUNING_SCORING]

# Search spaces are prefixed with "clf__" because they're applied to the
# classifier step inside the preprocess -> SMOTE -> clf pipeline.
SEARCH_SPACES = {
    "Logistic Regression": {
        "clf__C": loguniform(1e-3, 1e2),
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
    },
    "KNN": {
        "clf__n_neighbors": randint(3, 60),
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    },
    "Decision Tree": {
        "clf__max_depth": [3, 4, 5, 6, 8, 10, 15, None],
        "clf__min_samples_leaf": randint(1, 60),
        "clf__min_samples_split": randint(2, 40),
        "clf__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "clf__n_estimators": randint(200, 600),
        "clf__max_depth": [5, 8, 12, 16, None],
        "clf__min_samples_leaf": randint(1, 20),
        "clf__max_features": ["sqrt", "log2", 0.5],
    },
    "XGBoost": {
        "clf__n_estimators": randint(100, 600),
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__max_depth": randint(2, 8),
        "clf__min_child_weight": randint(1, 10),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__reg_lambda": loguniform(1e-2, 10),
    },
}


def _test_metrics(pipeline, X_test, y_test):
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "brier": brier_score_loss(y_test, proba),
    }


def tune_and_compare(X_train, y_train, X_test, y_test, preprocessor):
    """RandomizedSearchCV for every model on the training set only.

    The *model choice* is made on cross-validated PR-AUC, never on the test
    set; the test set is scored once at the end as an independent check. CV
    scores of the winning configuration are slightly optimistic (they were
    also used to pick it), which is exactly why the held-out test column is
    reported next to them.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows, tuned_params, searches = [], {}, {}
    for name, clf in build_models().items():
        # The search runs folds in parallel, so keep each model single-
        # threaded to avoid oversubscribing the CPU.
        if "n_jobs" in clf.get_params():
            clf.set_params(n_jobs=1)

        search = RandomizedSearchCV(
            make_pipeline(preprocessor, clf),
            param_distributions=SEARCH_SPACES[name],
            n_iter=N_ITER_SEARCH,
            scoring=SCORERS,
            refit=REFIT_METRIC,
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        print(f"Tuning {name} ({N_ITER_SEARCH} candidates x {CV_FOLDS} folds)...")
        search.fit(X_train, y_train)
        searches[name] = search

        best = search.best_index_
        cvr = search.cv_results_
        row = {"model": name}
        for metric in SCORERS:
            mean = cvr[f"mean_test_{metric}"][best]
            # Brier is reported by sklearn as a negative score; flip it back
            # so lower = better reads naturally.
            row[f"cv_{metric}_mean"] = -mean if metric == "brier" else mean
            row[f"cv_{metric}_std"] = cvr[f"std_test_{metric}"][best]
        for metric, value in _test_metrics(search.best_estimator_, X_test, y_test).items():
            row[f"test_{metric}"] = value
        rows.append(row)

        tuned_params[name] = {
            k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
            for k, v in search.best_params_.items()
        }
        print(f"  CV PR-AUC {row['cv_pr_auc_mean']:.3f} +/- {row['cv_pr_auc_std']:.3f}"
              f" | test PR-AUC {row['test_pr_auc']:.3f}")

    results = pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False)
    best_name = results.iloc[0]["model"]

    os.makedirs(REPORTS_DIR, exist_ok=True)
    results.round(4).to_csv(CV_RESULTS_PATH, index=False)
    with open(TUNED_PARAMS_PATH, "w") as f:
        json.dump({"selected_by": f"cv_{REFIT_METRIC}", "best_model": best_name,
                   "params": tuned_params}, f, indent=2)
    joblib.dump(searches[best_name].best_estimator_, TUNED_MODEL_PATH)
    _plot_comparison(results)

    print("\n" + _summary_table(results).to_string(index=False))
    print(f"\nSelected by CV {REFIT_METRIC}: {best_name}")
    print(f"Saved comparison to {CV_RESULTS_PATH}")
    print(f"Saved tuned {best_name} pipeline to {TUNED_MODEL_PATH}")
    return results, best_name


def _summary_table(results):
    """Compact human-readable view: mean +/- std for the headline metrics."""
    out = pd.DataFrame({"Model": results["model"]})
    for metric, label in [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"),
                          ("f1", "F1 (buyers)"), ("f1_macro", "Macro F1")]:
        out[f"CV {label}"] = [
            f"{m:.3f} ± {s:.3f}" for m, s in
            zip(results[f"cv_{metric}_mean"], results[f"cv_{metric}_std"])
        ]
        out[f"Test {label}"] = results[f"test_{metric}"].map("{:.3f}".format)
    return out


def _plot_comparison(results):
    ordered = results.sort_values("cv_pr_auc_mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = range(len(ordered))
    ax.barh(y, ordered["cv_pr_auc_mean"], xerr=ordered["cv_pr_auc_std"],
            color="#4C72B0", alpha=0.85, capsize=4, label=f"{CV_FOLDS}-fold CV (mean ± std)")
    ax.scatter(ordered["test_pr_auc"], y, color="#DD8452", zorder=3,
               marker="D", label="Held-out test")
    ax.set_yticks(list(y))
    ax.set_yticklabels(ordered["model"])
    ax.set_xlabel("PR-AUC (purchase class)")
    ax.set_xlim(0, 1)
    ax.set_title("Tuned models: cross-validated vs. held-out PR-AUC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "cv_model_comparison.png"), dpi=150)
    plt.close(fig)
