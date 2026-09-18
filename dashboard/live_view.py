from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dashboard import api_client as api


def _status_badge(running: bool) -> str:
    return "🟢 Running" if running else "⚪ Stopped"


def render():
    st.subheader("Live transaction feed")
    st.caption(
        "Replays a curated mix of real, held-out test-set transactions (every fraud "
        "row + a random sample of legitimate ones) plus periodic synthetic "
        "obvious-anomaly transactions, scoring each live with the deployed "
        "Isolation Forest model."
    )

    health = api.get_health()
    if health.get("status") != "ok":
        st.error(
            f"Backend unreachable at the configured API URL. "
            f"Start it locally with `uvicorn api.main:app --reload`, or check "
            f"the API_BASE_URL setting.\n\nDetail: {health.get('error', 'unknown error')}"
        )
        return

    storage_note = "Supabase" if health.get("supabase_configured") else "in-memory (not persisted -- see README Supabase setup)"
    st.caption(f"Storage backend: **{storage_note}**")

    status = api.get_stream_status()

    col_start, col_stop, col_delay, col_refresh = st.columns([1, 1, 2, 1])
    with col_delay:
        delay = st.slider("Delay between transactions (s)", 0.05, 2.0, 0.3, 0.05, disabled=status["running"])
    with col_start:
        st.write("")
        if st.button("▶ Start stream", disabled=status["running"], width='stretch'):
            api.start_stream(delay)
            st.rerun()
    with col_stop:
        st.write("")
        if st.button("⏹ Stop stream", disabled=not status["running"], width='stretch'):
            api.stop_stream()
            st.rerun()
    with col_refresh:
        st.write("")
        auto_refresh = st.checkbox("Auto-refresh", value=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", _status_badge(status["running"]))
    m2.metric("Processed", status["total_processed"])
    m3.metric("Flagged", status["total_flagged"])
    flag_rate = (status["total_flagged"] / status["total_processed"] * 100) if status["total_processed"] else 0
    m4.metric("Flag rate", f"{flag_rate:.1f}%")

    df = api.get_recent_transactions(limit=200)

    if df.empty:
        st.info("No transactions yet -- click **Start stream** above.")
    else:
        df = df.copy()
        df["alert"] = df["flagged"].map({True: "🚨 Flagged", False: "✅ normal"})
        df["kind"] = df["is_synthetic"].map({True: "synthetic", False: "real"})
        df["true_label"] = df["true_class"].map({1: "fraud", 0: "legitimate", None: "n/a (synthetic)"})

        fig = px.scatter(
            df.iloc[::-1],
            x="ingested_at",
            y="anomaly_score",
            color="alert",
            symbol="kind",
            hover_data=["amount", "true_label"],
            color_discrete_map={"🚨 Flagged": "#d62728", "✅ normal": "#1f77b4"},
            title="Anomaly score over time (most recent 200)",
        )
        fig.add_hline(
            y=df["threshold"].iloc[0],
            line_dash="dash",
            line_color="gray",
            annotation_text="deployment threshold",
        )
        st.plotly_chart(fig, width='stretch')

        display_cols = ["ingested_at", "alert", "amount", "anomaly_score", "kind", "true_label"]
        st.dataframe(
            df[display_cols].rename(columns={
                "ingested_at": "time",
                "amount": "amount ($)",
                "anomaly_score": "score",
            }),
            width='stretch',
            height=400,
        )

    if auto_refresh:
        st_autorefresh(interval=2000, key="live_feed_autorefresh")
