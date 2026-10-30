import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from src.config import (
    CALIBRATION_RESULTS_PATH, CV_FOLDS, FIGURES_DIR, PROFIT_RESULTS_PATH,
    RANDOM_STATE, REPORTS_DIR, TUNED_MODEL_PATH,
)
from src.profit import _at_threshold, breakeven_threshold


def _metrics(y, p):
    return {
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "pr_auc": average_precision_score(y, p),
        "mean_predicted_p_purchase": float(np.mean(p)),
        "actual_purchase_rate": float(np.mean(y)),
    }


def run_calibration(X_train, y_train, X_test, y_test):
    """Does P(purchase)=0.3 actually mean 30% of such sessions buy?

    SMOTE trains the model on a 50/50 class mix, so it overestimates
    P(purchase) on real (15% buyer) traffic. CalibratedClassifierCV learns a
    mapping from raw scores to observed frequencies on held-out folds of the
    training set; the test set is only used to evaluate.

    Isotonic is the pre-registered choice (enough data -- ~10k sessions --
    for its flexible fit); sigmoid/Platt is reported alongside for comparison.
    """
    if not os.path.exists(TUNED_MODEL_PATH):
        raise FileNotFoundError(
            f"{TUNED_MODEL_PATH} not found -- run `python main.py tune` first."
        )
    tuned = joblib.load(TUNED_MODEL_PATH)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    probas = {"Uncalibrated": tuned.predict_proba(X_test)[:, 1]}
    for method in ["isotonic", "sigmoid"]:
        base = clone(tuned)
        if "n_jobs" in base.named_steps["clf"].get_params():
            base.named_steps["clf"].set_params(n_jobs=1)
        print(f"Fitting {method} calibration ({CV_FOLDS} folds)...")
        calibrated = CalibratedClassifierCV(base, method=method, cv=cv, n_jobs=-1)
        calibrated.fit(X_train, y_train)
        probas[method.capitalize()] = calibrated.predict_proba(X_test)[:, 1]

    metrics = {name: _metrics(y_test, p) for name, p in probas.items()}

    # Payoff of calibration: with honest probabilities the threshold follows
    # straight from the cost assumptions -- no threshold search needed.
    t_star = breakeven_threshold()
    with open(PROFIT_RESULTS_PATH) as f:
        empirical = json.load(f)["test"]["profit_optimal_threshold"]
    profit = {
        "uncalibrated_at_breakeven": _at_threshold(y_test, probas["Uncalibrated"], t_star),
        "isotonic_at_breakeven": _at_threshold(y_test, probas["Isotonic"], t_star),
        "uncalibrated_at_empirical_threshold": empirical,
    }

    out = {"breakeven_threshold": t_star, "test_metrics": metrics, "profit_per_1000": profit}
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(CALIBRATION_RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    _plot(y_test, probas)

    print(f"\n{'':<14}{'Brier':>8}{'LogLoss':>9}{'PR-AUC':>8}{'Mean P':>8}  (actual rate {np.mean(y_test):.3f})")
    for name, m in metrics.items():
        print(f"{name:<14}{m['brier']:>8.4f}{m['log_loss']:>9.4f}{m['pr_auc']:>8.3f}"
              f"{m['mean_predicted_p_purchase']:>8.3f}")
    print(f"\nTest profit per 1,000 sessions:")
    for label, key in [(f"Uncalibrated @ break-even {t_star:.3f}", "uncalibrated_at_breakeven"),
                       (f"Isotonic     @ break-even {t_star:.3f}", "isotonic_at_breakeven"),
                       (f"Uncalibrated @ searched {empirical['threshold']:.2f}",
                        "uncalibrated_at_empirical_threshold")]:
        print(f"  {label:<38} ${profit[key]['profit_per_1000_sessions']:.0f}")
    print(f"\nSaved to {CALIBRATION_RESULTS_PATH}")
    return out


def _plot(y_test, probas):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Perfectly calibrated")
    colors = {"Uncalibrated": "#C44E52", "Isotonic": "#4C72B0", "Sigmoid": "#55A868"}
    for name, p in probas.items():
        # Quantile bins: most sessions have very low P(purchase), so uniform
        # bins would leave the upper range nearly empty.
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", color=colors[name],
                label=f"{name} (Brier {brier_score_loss(y_test, p):.3f})")
    ax.set_xlabel("Mean predicted P(purchase)")
    ax.set_ylabel("Observed share of sessions that purchased")
    ax.set_title("Reliability diagram (held-out test)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "calibration_curve.png"), dpi=150)
    plt.close(fig)
