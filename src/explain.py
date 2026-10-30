import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.config import (
    CATEGORICAL_COLUMNS, EXPLAIN_RESULTS_PATH, FIGURES_DIR, RANDOM_STATE,
    REPORTS_DIR, TUNED_MODEL_PATH,
)


class _RawColumnModel:
    """Exposes the full pipeline as f(numeric array of ORIGINAL columns) ->
    log-odds of purchase, so SHAP treats each of the 17 original columns as
    one player.

    Why not TreeExplainer on the one-hot features? Two failure modes, both
    checked against the model's own predictions:
    - tree-path mode weights branches by training data that includes SMOTE's
      synthetic buyers (one-hot dummies interpolated to values like 0.4); it
      gave Region a near-constant +1.7 log-odds and ranked it #1, while
      shuffling Region barely moves PR-AUC (0.002).
    - interventional mode masks each dummy independently, creating
      impossible sessions ("November AND May"); it said November LOWERS
      P(purchase), but setting every test session to November gives the
      highest mean prediction of any month (0.27).
    Masking whole original columns avoids both.
    """

    def __init__(self, pipeline, X_reference):
        self.preprocess = pipeline.named_steps["preprocess"]
        self.clf = pipeline.named_steps["clf"]
        self.columns = list(X_reference.columns)
        # Categorical values -> integer codes (SHAP maskers need numbers).
        self.categories = {
            c: sorted(X_reference[c].unique(), key=str)
            for c in self.columns if c in CATEGORICAL_COLUMNS
        }

    def encode(self, X):
        X = X.copy()
        for c, cats in self.categories.items():
            X[c] = X[c].map({v: i for i, v in enumerate(cats)}).fillna(-1)
        return X.astype(float).to_numpy()

    def __call__(self, arr):
        X = pd.DataFrame(arr, columns=self.columns)
        for c, cats in self.categories.items():
            X[c] = [cats[int(i)] if i >= 0 else None for i in X[c]]
        return self.clf.predict(self.preprocess.transform(X), output_margin=True)


def run_explainability(X_train, X_test, y_test, top_n=15,
                       n_explain=500, background_size=100):
    """Two independent views of what drives the tuned model's predictions,
    both on the held-out test set:

    - SHAP: per-prediction attributions on the original columns, with a
      background of REAL (not SMOTE-resampled) training sessions. Shows
      direction as well as size. Run on a random sample of test sessions
      because the model-agnostic explainer is slower than TreeExplainer.
    - Permutation importance: how much test PR-AUC drops when a column is
      shuffled. Model-agnostic, all test sessions.

    If both agree on the top drivers, the finding is more trustworthy than
    either one alone.
    """
    if not os.path.exists(TUNED_MODEL_PATH):
        raise FileNotFoundError(
            f"{TUNED_MODEL_PATH} not found -- run `python main.py tune` first."
        )
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pipeline = joblib.load(TUNED_MODEL_PATH)
    model = _RawColumnModel(pipeline, X_train)

    background = model.encode(X_train.sample(background_size, random_state=RANDOM_STATE))
    sample = X_test.sample(min(n_explain, len(X_test)), random_state=RANDOM_STATE)
    print(f"Computing SHAP values on {len(sample)} test sessions "
          f"({background_size} background sessions)...")
    explainer = shap.PermutationExplainer(
        model, shap.maskers.Independent(background, max_samples=background_size),
        seed=RANDOM_STATE,
    )
    explanation = explainer(model.encode(sample))
    explanation.feature_names = model.columns

    shap_by_column = pd.Series(
        np.abs(explanation.values).mean(axis=0), index=model.columns,
    ).sort_values(ascending=False)

    # Beeswarm colour = feature value, which is only meaningful for numeric
    # columns (categorical codes are arbitrary), so plot numeric ones only.
    numeric = [i for i, c in enumerate(model.columns) if c not in model.categories]
    plt.figure()
    shap.plots.beeswarm(explanation[:, numeric], max_display=top_n, show=False)
    plt.title("SHAP values for numeric columns (log-odds of purchase)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Categorical columns: average signed SHAP per category instead.
    category_effects = {
        c: pd.Series(explanation.values[:, model.columns.index(c)])
              .groupby(sample[c].astype(str).values).mean()
              .sort_values(ascending=False).round(3).to_dict()
        for c in model.categories
    }

    print("Computing permutation importance (10 repeats, scored on PR-AUC)...")
    perm = permutation_importance(
        pipeline, X_test, y_test, scoring="average_precision",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    perm_by_column = pd.DataFrame({
        "mean": perm.importances_mean, "std": perm.importances_std,
    }, index=X_test.columns).sort_values("mean", ascending=False)

    _plot_importance(shap_by_column.head(top_n), perm_by_column.head(top_n))

    # Direction for numeric columns: correlation between a column's value and
    # its SHAP value (positive = higher value -> more likely to buy).
    direction = {
        model.columns[i]: float(np.corrcoef(explanation.data[:, i], explanation.values[:, i])[0, 1])
        for i in numeric if np.std(explanation.data[:, i]) > 0
    }

    out = {
        "model": type(model.clf).__name__,
        "shap_sessions_explained": len(sample),
        "shap_mean_abs_by_column": shap_by_column.round(4).to_dict(),
        "permutation_importance_pr_auc_drop": {
            k: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)}
            for k, v in perm_by_column.to_dict("index").items()
        },
        "shap_direction_numeric": {k: round(v, 3) for k, v in direction.items()},
        "shap_mean_by_category": category_effects,
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(EXPLAIN_RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("\nTop drivers:")
    print(f"  {'Column':<26}{'mean |SHAP|':>12}{'PR-AUC drop':>14}")
    for col in shap_by_column.index[:10]:
        p = perm_by_column.loc[col]
        print(f"  {col:<26}{shap_by_column[col]:>12.3f}{p['mean']:>9.3f} ± {p['std']:.3f}")
    print(f"\nMonth effect (mean SHAP): {category_effects['Month']}")
    print(f"Saved to {EXPLAIN_RESULTS_PATH}")
    return out


def _plot_importance(shap_by_column, perm):
    """Side by side so agreement (or disagreement) between the two methods
    is visible at a glance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    s = shap_by_column.iloc[::-1]
    ax1.barh(s.index, s.values, color="#4C72B0", alpha=0.85)
    ax1.set_xlabel("Mean |SHAP| (log-odds of purchase)")
    ax1.set_title("SHAP (original columns)")
    p = perm.iloc[::-1]
    ax2.barh(p.index, p["mean"], xerr=p["std"], color="#DD8452", alpha=0.85, capsize=3)
    ax2.axvline(0, color="grey", linewidth=0.8)
    ax2.set_xlabel("Drop in test PR-AUC when the column is shuffled")
    ax2.set_title("Permutation importance")
    fig.suptitle("What drives the tuned XGBoost model (held-out test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
