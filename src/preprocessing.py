import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import (
    DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, CATEGORICAL_COLUMNS,
)


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV. No encoding/scaling here -- that must only ever
    happen after the train/test split, otherwise the test set's statistics
    leak into training (this was a bug in the previous version, where
    StandardScaler was fit on the full dataset before splitting)."""
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates()
    if before != len(df):
        print(f"Dropped {before - len(df)} exact duplicate rows.")
    missing = df.isnull().sum().sum()
    if missing:
        raise ValueError(
            f"Found {missing} missing values -- decide on an imputation "
            f"strategy before proceeding rather than silently dropping rows."
        )
    return df


def split_raw_data(df: pd.DataFrame):
    """Split BEFORE any encoding/scaling/resampling touches the data."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build (but do not fit) a ColumnTransformer. Fitting happens only
    inside the modeling Pipeline, on the training fold, so scaling/encoding
    parameters never see test (or validation-fold) data."""
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in X_train.columns]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
        ]
    )


def load_and_prepare_data():
    """Convenience wrapper: returns raw (unencoded) train/test splits plus
    a not-yet-fitted preprocessor, ready to be dropped into a Pipeline."""
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_raw_data(df)
    preprocessor = build_preprocessor(X_train)
    return X_train, X_test, y_train, y_test, preprocessor
