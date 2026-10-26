import json
import os

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import BEST_MODEL_PATH, METRICS_PATH, MODEL_DIR, RANDOM_STATE


def build_models():
    """random_state is set on every stochastic model so runs are
    reproducible (the RandomForest previously had none)."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        # eval_metric moved here; use_label_encoder was removed from recent
        # XGBoost releases and raises an error if passed.
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
    }


def make_pipeline(preprocessor, clf):
    """preprocessor -> SMOTE -> classifier, as one estimator. SMOTE is applied
    *inside* the pipeline so it only ever resamples the training fold -- it
    never touches X_test or held-out CV folds."""
    return ImbPipeline(steps=[
        ("preprocess", clone(preprocessor)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", clf),
    ])


def train_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor):
    """Baseline run: every model with default hyperparameters, one
    train/test split. See src/tuning.py for the tuned, cross-validated
    comparison."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    results = {}
    best_name, best_pipeline, best_score = None, None, -1

    for name, clf in build_models().items():
        pipeline = make_pipeline(preprocessor, clf)

        print(f"\n{'=' * 60}\nTraining: {name}\n{'=' * 60}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        print("Classification report (both classes -- do not headline only "
              "the majority class's numbers):")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

        auc = None
        if hasattr(pipeline, "predict_proba"):
            auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
            print(f"AUC-ROC: {auc:.4f}")

        print(f"Macro F1 (both classes averaged equally): {macro_f1:.4f}")

        results[name] = {
            "classification_report": report,
            "macro_f1": macro_f1,
            "auc_roc": auc,
        }

        if macro_f1 > best_score:
            best_name, best_pipeline, best_score = name, pipeline, macro_f1

    print(f"\nBest model by macro F1: {best_name} ({best_score:.4f})")

    import joblib
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2, default=str)

    print(f"Saved best model to {BEST_MODEL_PATH}")
    print(f"Saved full metrics to {METRICS_PATH}")

    return results, best_name
