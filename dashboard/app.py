"""
Streamlit dashboard for the Transaction Anomaly Detector.

Run locally (with the FastAPI backend already running):
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import compare_view, how_view, live_view  # noqa: E402

st.set_page_config(page_title="Transaction Anomaly Detector", page_icon="💳", layout="wide")

st.title("💳 Transaction Anomaly Detector")
st.caption("Real-time, unsupervised fraud detection -- trained without ever seeing a fraud label.")

tab_live, tab_compare, tab_how = st.tabs(["🔴 Live Feed", "📊 Model Comparison", "❓ How This Works"])

with tab_live:
    live_view.render()

with tab_compare:
    compare_view.render()

with tab_how:
    how_view.render()
