import argparse

from src.preprocessing import load_and_prepare_data
from src.model_utils import train_and_evaluate_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step", nargs="?", default="baseline",
        choices=["baseline", "tune", "profit", "explain", "calibrate", "all"],
        help="baseline: default hyperparameters, one split (original behaviour); "
             "tune: RandomizedSearchCV + 5-fold CV comparison; "
             "profit: cost-based threshold analysis on the tuned model; "
             "explain: SHAP + permutation importance on the tuned model; "
             "calibrate: probability calibration (needs profit first); "
             "all: tune, profit, explain, calibrate",
    )
    step = parser.parse_args().step

    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare_data()
    if step == "baseline":
        train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor)
    if step in ("tune", "all"):
        from src.tuning import tune_and_compare
        tune_and_compare(X_train, y_train, X_test, y_test, preprocessor)
    if step in ("profit", "all"):
        from src.profit import run_profit_analysis
        run_profit_analysis(X_train, y_train, X_test, y_test)
    if step in ("explain", "all"):
        from src.explain import run_explainability
        run_explainability(X_train, X_test, y_test)
    if step in ("calibrate", "all"):
        from src.calibration import run_calibration
        run_calibration(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
