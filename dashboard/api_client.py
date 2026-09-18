"""
Thin wrapper around the FastAPI backend's HTTP endpoints.

Live/status calls are never cached (the whole point is to see fresh
state). The two /models/* endpoints return static Phase 3 artifacts that
don't change between backend restarts, so they're cached for the session
to avoid re-fetching a ~57K-row scores table on every rerun.
"""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from dashboard.config import API_BASE_URL

TIMEOUT = 5
SCORES_TIMEOUT = 20


def get_health() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"status": "unreachable", "error": str(e)}


def get_stream_status() -> dict:
    r = requests.get(f"{API_BASE_URL}/stream/status", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def start_stream(delay_seconds: float) -> dict:
    r = requests.post(f"{API_BASE_URL}/stream/start", json={"delay_seconds": delay_seconds}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def stop_stream() -> dict:
    r = requests.post(f"{API_BASE_URL}/stream/stop", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_recent_transactions(limit: int = 100, flagged_only: bool = False) -> pd.DataFrame:
    r = requests.get(
        f"{API_BASE_URL}/transactions/recent",
        params={"limit": limit, "flagged_only": flagged_only},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=3600, show_spinner="Loading model comparison...")
def get_model_comparison() -> pd.DataFrame:
    r = requests.get(f"{API_BASE_URL}/models/comparison", timeout=TIMEOUT)
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=3600, show_spinner="Loading test-set scores...")
def get_model_scores() -> pd.DataFrame:
    r = requests.get(f"{API_BASE_URL}/models/scores", timeout=SCORES_TIMEOUT)
    r.raise_for_status()
    return pd.DataFrame(r.json())
