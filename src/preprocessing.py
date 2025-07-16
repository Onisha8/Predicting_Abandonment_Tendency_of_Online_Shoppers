import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.config import DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    # Drop NA or check for missing
    df = df.dropna()

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Separate features and target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN].astype(int)

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Handle imbalance
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test
