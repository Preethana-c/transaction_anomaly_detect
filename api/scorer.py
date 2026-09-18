"""
Live anomaly scoring for the streaming backend.

Loads the same artifacts notebooks/02 and notebooks/03 saved --
`models/isolation_forest.joblib` and `models/isolation_forest_threshold.json`
-- once at import time, and reuses `src/preprocessing.py` for feature
construction so a transaction scored here goes through the identical
pipeline as training. That's the whole point of centralizing
preprocessing in `src/`: train/serve skew (subtly different scaling
between training and inference) is one of the most common ways a model
that tested well quietly stops working once deployed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import preprocessing as prep  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "isolation_forest.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "isolation_forest_threshold.json"


class Scorer:
    def __init__(self):
        if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
            raise FileNotFoundError(
                "Deployed model artifacts missing -- run notebooks/02_train_models.py "
                "and notebooks/03_evaluation.py first."
            )
        self.model = joblib.load(MODEL_PATH)
        self.scaler = prep.load_scaler()
        with open(THRESHOLD_PATH) as f:
            threshold_info = json.load(f)
        self.threshold = float(threshold_info["threshold"])

    def score_row(self, raw_row: dict) -> tuple[float, bool]:
        """Score a single transaction dict (Time, Amount, V1..V28) live.

        Returns (anomaly_score, flagged). Sign convention matches
        Phase 2/3: higher score = more anomalous. `raw_row` may include
        extra keys (e.g. Class, is_synthetic) -- they're ignored here,
        same as `preprocessing.split_features_and_label` ignores Class.
        """
        df = pd.DataFrame([raw_row])
        df_scaled = prep.apply_scaler(df, self.scaler)
        X, _ = prep.split_features_and_label(df_scaled)
        score = float(-self.model.score_samples(X)[0])
        flagged = score >= self.threshold
        return score, flagged


_scorer: Scorer | None = None


def get_scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        _scorer = Scorer()
    return _scorer
