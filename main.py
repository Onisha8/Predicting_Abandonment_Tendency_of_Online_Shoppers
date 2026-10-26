import argparse

from src.preprocessing import load_and_prepare_data
from src.model_utils import train_and_evaluate_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step", nargs="?", default="baseline", choices=["baseline", "tune", "profit", "all"],
        help="baseline: default hyperparameters, one split (original behaviour); "
             "tune: RandomizedSearchCV + 5-fold CV comparison; "
             "profit: cost-based threshold analysis on the tuned model; "
             "all: tune then profit",
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


if __name__ == "__main__":
    main()
