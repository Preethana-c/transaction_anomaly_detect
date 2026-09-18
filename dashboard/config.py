"""
Dashboard configuration.

Tries Streamlit secrets first (how you'd set API_BASE_URL when deployed
on Streamlit Community Cloud, in .streamlit/secrets.toml), then falls
back to a plain environment variable / root .env (for local dev), then a
localhost default.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _get(key: str, default: str) -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


API_BASE_URL = _get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
