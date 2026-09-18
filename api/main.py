"""
FastAPI backend: simulates a live transaction feed, scores each
transaction with the deployed Isolation Forest model, and logs the
result to Supabase (or an in-memory store if Supabase isn't configured
-- see api/storage.py).

Run locally from the repo root:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import config  # noqa: E402
from api.schemas import StreamStartRequest, StreamStatus, Transaction  # noqa: E402
from api.scorer import get_scorer  # noqa: E402
from api.storage import get_storage  # noqa: E402
from api.streamer import stream_state  # noqa: E402

app = FastAPI(title="Transaction Anomaly Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "supabase_configured": config.SUPABASE_CONFIGURED,
        "storage_backend": "supabase" if config.SUPABASE_CONFIGURED else "in-memory",
    }


@app.post("/stream/start", response_model=StreamStatus)
async def start_stream(req: StreamStartRequest):
    try:
        get_scorer()  # fail fast with a clear error if model artifacts are missing
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return await stream_state.start(req.delay_seconds)


@app.post("/stream/stop", response_model=StreamStatus)
async def stop_stream():
    return await stream_state.stop()


@app.get("/stream/status", response_model=StreamStatus)
def stream_status():
    return stream_state.status()


@app.get("/transactions/recent", response_model=list[Transaction])
def recent_transactions(limit: int = 50, flagged_only: bool = False):
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    storage = get_storage()
    return storage.recent(limit=limit, flagged_only=flagged_only)


@app.get("/models/comparison")
def model_comparison():
    """Phase 3's evaluation table, served for the Phase 5 dashboard.

    A static artifact (already computed at training time) -- this just
    exposes it over HTTP so the dashboard doesn't need filesystem access
    to a notebook output when it's deployed separately on Streamlit Cloud.
    """
    path = PROJECT_ROOT / "data" / "processed" / "model_comparison.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="model_comparison.csv not found -- run notebooks/03_evaluation.py")
    return pd.read_csv(path).to_dict(orient="records")
