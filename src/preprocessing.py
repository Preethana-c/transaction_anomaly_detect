"""
Shared preprocessing for the Transaction Anomaly Detector.

This module is imported by both the training notebooks (Phase 1-3) and the
FastAPI streaming backend (Phase 4), so that transactions scored live go
through EXACTLY the same transform as transactions used for training. If
these ever drift apart, the live anomaly scores become meaningless.

IMPORTANT: `Class` is the fraud label. It is read here only so callers can
split it off; it never participates in fitting the scaler and is never
passed to a model as a feature.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Project paths, resolved relative to this file so scripts work regardless
# of the current working directory they're launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "creditcard_processed.csv"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"

LABEL_COLUMN = "Class"
# V1-V28 arrive already PCA-transformed and scaled (per the Kaggle dataset
# description) -- we leave them untouched. Only Time and Amount are raw
# (and Amount gets a log1p transform before scaling -- see `_log_amount`).
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
# The exact, ordered feature set every model trains on and every live
# transaction is scored on. Class is deliberately excluded.
FEATURE_COLUMNS = PCA_COLUMNS + ["Time_scaled", "Amount_scaled"]


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from Kaggle first -- see README.md 'Getting the data'."
        )
    return pd.read_csv(path)


def _log_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 2-col frame [Time, Amount] with Amount log1p-transformed.

    Amount is heavily right-skewed (most transactions are small, a long
    tail runs up to five figures -- see the Phase 1 histogram). Fitting
    StandardScaler on raw Amount lets that tail dominate: a handful of
    large but perfectly legitimate purchases end up with huge Amount
    z-scores, which in turn dominate a reconstruction-error-based anomaly
    score and drown out the actual fraud signal in the other 29 features.
    log1p compresses the tail before scaling, so "large but normal" and
    "small but structurally weird" don't collapse into the same score
    range for the wrong reason. This is a transform of Amount's own
    distribution -- it never looks at Class.
    """
    out = df[["Time", "Amount"]].copy()
    out["Amount"] = np.log1p(out["Amount"])
    return out


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on Time and log1p(Amount). No labels involved."""
    scaler = StandardScaler()
    scaler.fit(_log_amount(df))
    return scaler


def save_scaler(scaler: StandardScaler, path: Path = SCALER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path = SCALER_PATH) -> StandardScaler:
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {path}. Run notebooks/01_eda_preprocessing.py first."
        )
    return joblib.load(path)


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Return a copy of df with Time_scaled / Amount_scaled columns added.

    Amount_scaled is StandardScaler applied to log1p(Amount), not raw
    Amount -- see `_log_amount`. Original Time/Amount columns are kept
    (useful for display in the demo); models only ever read
    FEATURE_COLUMNS.
    """
    out = df.copy()
    scaled = scaler.transform(_log_amount(out))
    out["Time_scaled"] = scaled[:, 0]
    out["Amount_scaled"] = scaled[:, 1]
    return out


def split_features_and_label(df: pd.DataFrame):
    """Split a processed dataframe into (X, y).

    y is returned ONLY for evaluation purposes downstream -- training code
    must never look at it. Keeping the split here (rather than in every
    notebook cell) makes that boundary explicit and hard to violate by accident.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].copy() if LABEL_COLUMN in df.columns else None
    return X, y


def load_processed(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run notebooks/01_eda_preprocessing.py first."
        )
    return pd.read_csv(path)


def train_test_split_df(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Deterministic, unstratified train/test split.

    Deliberately NOT stratified on Class: stratifying would mean using the
    label to shape the split, which is a small but real crack in the
    "blind to labels" framing. A plain random split means fraud lands in
    both train and test at roughly its natural base rate -- and fraud
    rows are NOT removed from the training set, because without labels we
    have no way to know which rows to remove. That's realistic: a
    production model trains on whatever traffic arrives, fraud included.

    random_state is fixed so every phase (training, evaluation, the
    streaming replay in Phase 4) reconstructs the exact same split from
    the same processed CSV without needing to persist duplicate files.
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
