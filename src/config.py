import os

# Paths are relative to the project root, resolved from this file's own
# location, so the code works regardless of the current working directory
# (fixes the hardcoded Windows path that was in the original notebook).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "online_shoppers_intention.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = "Revenue"

# Categorical columns, encoded explicitly rather than relying on pandas
# dtype inference (Weekend/OperatingSystems/etc. are numeric-looking but
# are actually categorical codes, not ordinal quantities).
CATEGORICAL_COLUMNS = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend",
]
