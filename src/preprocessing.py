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
# description) -- we leave them untouched. Only Time and Amount are raw.
COLUMNS_TO_SCALE = ["Time", "Amount"]
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


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on Time and Amount only. No labels involved."""
    scaler = StandardScaler()
    scaler.fit(df[COLUMNS_TO_SCALE])
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

    Original Time/Amount columns are kept (useful for display in the demo);
    models only ever read FEATURE_COLUMNS.
    """
    out = df.copy()
    scaled = scaler.transform(out[COLUMNS_TO_SCALE])
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
