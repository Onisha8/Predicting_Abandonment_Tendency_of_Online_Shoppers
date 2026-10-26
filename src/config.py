import os

# Paths are relative to the project root, resolved from this file's own
# location, so the code works regardless of the current working directory
# (fixes the hardcoded Windows path that was in the original notebook).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "online_shoppers_intention.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

TUNED_MODEL_PATH = os.path.join(MODEL_DIR, "tuned_best_model.joblib")

# Small result tables and figures ARE committed (unlike models/), so the
# README can show them.
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
CV_RESULTS_PATH = os.path.join(REPORTS_DIR, "cv_model_comparison.csv")
TUNED_PARAMS_PATH = os.path.join(REPORTS_DIR, "tuned_params.json")
PROFIT_RESULTS_PATH = os.path.join(REPORTS_DIR, "profit_threshold.json")

TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = "Revenue"

# Cross-validated tuning. PR-AUC (average precision) is the selection metric:
# it's threshold-free (the threshold is chosen separately, by profit) and,
# unlike ROC-AUC, it isn't inflated by the large number of easy negatives.
CV_FOLDS = 5
N_ITER_SEARCH = 25
TUNING_SCORING = "average_precision"

# ---------------------------------------------------------------------------
# Business assumptions for the profit/threshold analysis. These are
# ILLUSTRATIVE numbers, not figures from the dataset (which has no prices).
# They're kept here so anyone can plug in their own and re-run.
#
# Scenario: a session the model thinks will NOT buy (P(purchase) < threshold)
# gets a retention offer (e.g. a discount pop-up / email).
# ---------------------------------------------------------------------------
AVG_ORDER_VALUE = 100.0      # $ per completed order
GROSS_MARGIN = 0.30          # fraction of order value kept as profit
DISCOUNT_VALUE = 10.0        # $ given away whenever an offer is redeemed
CONTACT_COST = 0.50          # $ to serve one offer (email/pop-up/ops)
RECOVERY_RATE = 0.10         # share of targeted would-be abandoners who then buy
# Would-be buyers who get the offer still buy, but now redeem the discount,
# so it's pure cost on a sale that was going to happen anyway.

# Categorical columns, encoded explicitly rather than relying on pandas
# dtype inference (Weekend/OperatingSystems/etc. are numeric-looking but
# are actually categorical codes, not ordinal quantities).
CATEGORICAL_COLUMNS = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend",
]
