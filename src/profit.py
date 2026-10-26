import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import (
    AVG_ORDER_VALUE, CONTACT_COST, CV_FOLDS, DISCOUNT_VALUE, FIGURES_DIR,
    GROSS_MARGIN, PROFIT_RESULTS_PATH, RANDOM_STATE, RECOVERY_RATE,
    REPORTS_DIR, TUNED_MODEL_PATH,
)

THRESHOLDS = np.round(np.arange(0.0, 1.0001, 0.01), 2)
PER_N = 1000  # report profit per 1,000 sessions so it doesn't depend on test-set size


def payoffs(recovery_rate=RECOVERY_RATE):
    """Incremental $ of sending one offer, vs. doing nothing.

    - Would-be abandoner (y=0) targeted: `recovery_rate` of them now buy,
      paying full margin minus the discount; every one costs CONTACT_COST.
    - Would-be buyer (y=1) targeted: they'd have bought anyway, so the
      discount they redeem is pure loss, plus CONTACT_COST.
    - Anyone not targeted: $0 (that's the baseline).
    """
    margin = AVG_ORDER_VALUE * GROSS_MARGIN
    gain_abandoner = recovery_rate * (margin - DISCOUNT_VALUE) - CONTACT_COST
    loss_buyer = -(DISCOUNT_VALUE + CONTACT_COST)
    return gain_abandoner, loss_buyer


def breakeven_threshold(recovery_rate=RECOVERY_RATE):
    """For *calibrated* probabilities, target iff the expected value is
    positive: (1 - p) * gain - p * loss > 0  ->  p < gain / (gain + loss)."""
    gain, loss = payoffs(recovery_rate)
    return gain / (gain - loss) if gain > 0 else 0.0


def profit_curve(y_true, p_purchase, recovery_rate=RECOVERY_RATE):
    """Profit per PER_N sessions for every threshold: target sessions with
    P(purchase) < t. t=0 targets nobody; t=1 targets everybody."""
    y_true = np.asarray(y_true)
    gain, loss = payoffs(recovery_rate)
    profits = []
    for t in THRESHOLDS:
        targeted = p_purchase < t if t < 1 else np.ones_like(y_true, dtype=bool)
        n_abandoners = np.sum(targeted & (y_true == 0))
        n_buyers = np.sum(targeted & (y_true == 1))
        profits.append((n_abandoners * gain + n_buyers * loss) / len(y_true) * PER_N)
    return np.array(profits)


def _at_threshold(y_true, p_purchase, t, recovery_rate=RECOVERY_RATE):
    y_true = np.asarray(y_true)
    targeted = p_purchase < t
    gain, loss = payoffs(recovery_rate)
    n_ab = int(np.sum(targeted & (y_true == 0)))
    n_buy = int(np.sum(targeted & (y_true == 1)))
    return {
        "threshold": float(t),
        "profit_per_1000_sessions": (n_ab * gain + n_buy * loss) / len(y_true) * PER_N,
        "share_of_sessions_targeted": float(targeted.mean()),
        "abandoners_reached_pct": n_ab / max(1, int(np.sum(y_true == 0))),
        "buyers_given_unneeded_discount_pct": n_buy / max(1, int(np.sum(y_true == 1))),
    }


def run_profit_analysis(X_train, y_train, X_test, y_test):
    """Pick the profit-maximising threshold on OUT-OF-FOLD training
    predictions, then report it once on the untouched test set.

    Choosing the threshold on the test set would be the same leak as tuning
    hyperparameters on it. The probabilities come from a SMOTE-trained model,
    so they're not calibrated (SMOTE inflates P(purchase)); the empirical
    sweep handles that, which is why the chosen threshold differs from the
    theoretical break-even printed for comparison.
    """
    if not os.path.exists(TUNED_MODEL_PATH):
        raise FileNotFoundError(
            f"{TUNED_MODEL_PATH} not found -- run `python main.py tune` first."
        )
    tuned = joblib.load(TUNED_MODEL_PATH)
    model_name = type(tuned.named_steps["clf"]).__name__

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"Getting out-of-fold probabilities for tuned {model_name}...")
    oof = cross_val_predict(clone(tuned), X_train, y_train, cv=cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    test_p = tuned.predict_proba(X_test)[:, 1]

    oof_curve = profit_curve(y_train, oof)
    test_curve = profit_curve(y_test, test_p)
    best_t = float(THRESHOLDS[int(np.argmax(oof_curve))])

    y_test_arr = np.asarray(y_test)
    gain, _ = payoffs()
    test_results = {
        "profit_optimal_threshold": _at_threshold(y_test, test_p, best_t),
        "default_0.5_threshold": _at_threshold(y_test, test_p, 0.5),
        "target_everyone": _at_threshold(y_test, test_p, 1.0 + 1e-9),
        "target_nobody": _at_threshold(y_test, test_p, 0.0),
        # Upper bound: a perfect model that targets every abandoner and no buyer.
        "perfect_model": {"profit_per_1000_sessions":
                          float(np.mean(y_test_arr == 0) * gain * PER_N)},
    }

    sensitivity = []
    for rr in [0.05, 0.10, 0.15, 0.20]:
        curve = profit_curve(y_train, oof, rr)
        t = float(THRESHOLDS[int(np.argmax(curve))])
        sensitivity.append({
            "recovery_rate": rr,
            "chosen_threshold": t,
            "test_profit_per_1000": _at_threshold(y_test, test_p, t, rr)["profit_per_1000_sessions"],
            "test_profit_per_1000_at_0.5": _at_threshold(y_test, test_p, 0.5, rr)["profit_per_1000_sessions"],
        })

    out = {
        "model": model_name,
        "assumptions": {
            "avg_order_value": AVG_ORDER_VALUE, "gross_margin": GROSS_MARGIN,
            "discount_value": DISCOUNT_VALUE, "contact_cost": CONTACT_COST,
            "recovery_rate": RECOVERY_RATE,
        },
        "theoretical_breakeven_threshold_if_calibrated": breakeven_threshold(),
        "threshold_chosen_on": "out-of-fold training predictions",
        "test": test_results,
        "sensitivity_to_recovery_rate": sensitivity,
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(PROFIT_RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    _plot(oof_curve, test_curve, best_t, model_name)
    _print_summary(out)
    return out


def _print_summary(out):
    t = out["test"]
    print(f"\nThreshold chosen on out-of-fold train predictions: "
          f"{t['profit_optimal_threshold']['threshold']:.2f} "
          f"(break-even if probabilities were calibrated: "
          f"{out['theoretical_breakeven_threshold_if_calibrated']:.3f})")
    print("\nTest-set profit per 1,000 sessions:")
    for key, label in [("profit_optimal_threshold", "Profit-optimal threshold"),
                       ("default_0.5_threshold", "Default 0.5 threshold"),
                       ("target_everyone", "Offer to everyone"),
                       ("target_nobody", "Offer to nobody"),
                       ("perfect_model", "Perfect model (upper bound)")]:
        r = t[key]
        extra = ""
        if "share_of_sessions_targeted" in r:
            extra = (f"  | targets {r['share_of_sessions_targeted']:.0%} of sessions, "
                     f"reaches {r['abandoners_reached_pct']:.0%} of abandoners, "
                     f"discounts {r['buyers_given_unneeded_discount_pct']:.0%} of buyers")
        print(f"  {label:<30} ${r['profit_per_1000_sessions']:>8.0f}{extra}")
    print("\nSensitivity to recovery rate (threshold re-chosen each time):")
    for s in out["sensitivity_to_recovery_rate"]:
        print(f"  recovery {s['recovery_rate']:.0%}: threshold {s['chosen_threshold']:.2f} -> "
              f"${s['test_profit_per_1000']:.0f} vs ${s['test_profit_per_1000_at_0.5']:.0f} at 0.5")
    print(f"\nSaved to {PROFIT_RESULTS_PATH}")


def _plot(oof_curve, test_curve, best_t, model_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(THRESHOLDS, oof_curve, label="Out-of-fold (train) — used to choose", color="#4C72B0")
    ax.plot(THRESHOLDS, test_curve, label="Held-out test", color="#DD8452", linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(best_t, color="#55A868", linestyle=":", label=f"Chosen threshold = {best_t:.2f}")
    ax.axvline(0.5, color="#C44E52", linestyle=":", label="Default 0.5")
    ax.set_xlabel("Send offer if P(purchase) < threshold")
    ax.set_ylabel(f"Incremental profit per {PER_N:,} sessions ($)")
    ax.set_title(f"Retention-offer profit vs. decision threshold ({model_name})")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "profit_vs_threshold.png"), dpi=150)
    plt.close(fig)
