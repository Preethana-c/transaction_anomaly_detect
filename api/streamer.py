"""
The simulated live transaction feed.

We don't replay all ~57K test-set rows verbatim -- at any demo-watchable
delay that's hours long, and 98 fraud rows scattered across 57K would
mean long, boring stretches with nothing to see. Instead the demo stream
is: every fraud row in the test set (all 98 -- we don't want to miss any)
plus a random sample of legitimate test-set rows, shuffled together with
a fixed seed. It's still exclusively real, held-out test-set transactions
(save for the explicitly-requested synthetic injections below) -- just a
curated mix sized for a demo that finishes in a few minutes instead of
hours.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import config  # noqa: E402
from api.scorer import get_scorer  # noqa: E402
from api.storage import get_storage  # noqa: E402
from api.synthetic import make_synthetic_transaction  # noqa: E402
from src import preprocessing as prep  # noqa: E402


def build_demo_stream() -> pd.DataFrame:
    df = prep.load_processed()
    _, test_df = prep.train_test_split_df(df, test_size=0.2, random_state=42)

    fraud_rows = test_df[test_df["Class"] == 1]
    legit_rows = test_df[test_df["Class"] == 0].sample(
        n=min(config.DEMO_N_LEGIT, len(test_df)), random_state=config.DEMO_SEED
    )
    demo_df = pd.concat([fraud_rows, legit_rows]).sample(frac=1, random_state=config.DEMO_SEED)
    return demo_df.reset_index(drop=True)


class StreamState:
    """Holds the single background streaming task and its live stats.

    One demo stream at a time is enough for this project -- `start()` is
    a no-op (returns current status) if a stream is already running.
    """

    def __init__(self):
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self.running = False
        self.total_processed = 0
        self.total_flagged = 0
        self.started_at: float | None = None

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "total_processed": self.total_processed,
            "total_flagged": self.total_flagged,
            "started_at": self.started_at,
            "elapsed_seconds": (time.time() - self.started_at) if self.started_at else None,
        }

    async def start(self, delay_seconds: float):
        if self.running:
            return self.status()

        self._stop_event.clear()
        self.total_processed = 0
        self.total_flagged = 0
        self.started_at = time.time()
        self.running = True
        self._task = asyncio.create_task(self._run(delay_seconds))
        return self.status()

    async def stop(self):
        if not self.running:
            return self.status()
        self._stop_event.set()
        if self._task:
            await self._task
        return self.status()

    async def _run(self, delay_seconds: float):
        demo_df = build_demo_stream()
        scorer = get_scorer()
        storage = get_storage()
        rng = np.random.RandomState(config.DEMO_SEED)

        for idx, row in demo_df.iterrows():
            if self._stop_event.is_set():
                break

            is_synthetic = idx > 0 and idx % config.SYNTHETIC_ANOMALY_EVERY_N == 0
            true_class = None
            if is_synthetic:
                base_row = demo_df.iloc[rng.randint(len(demo_df))]
                raw_row = make_synthetic_transaction(base_row, synthetic_time=float(row["Time"]))
            else:
                raw_row = row[["Time", "Amount"] + prep.PCA_COLUMNS].to_dict()
                true_class = int(row["Class"])

            score, flagged = scorer.score_row(raw_row)

            storage.insert({
                "original_time": raw_row["Time"],
                "amount": raw_row["Amount"],
                "anomaly_score": score,
                "threshold": scorer.threshold,
                "flagged": bool(flagged),
                "is_synthetic": bool(is_synthetic),
                "true_class": true_class,
                "model_name": "isolation_forest",
            })

            self.total_processed += 1
            if flagged:
                self.total_flagged += 1

            await asyncio.sleep(delay_seconds)

        self.running = False


stream_state = StreamState()
