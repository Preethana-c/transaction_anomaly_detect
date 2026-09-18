from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

from dashboard import api_client as api

MODEL_SCORE_COLUMNS = {
    "Isolation Forest": "iso_forest_score",
    "One-Class SVM": "ocsvm_score",
    "Autoencoder": "autoencoder_score",
}


def _threshold_table(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1[:-1]))

    def _row(label, t):
        y_pred = (scores >= t).astype(int)
        return {
            "strategy": label,
            "threshold": round(float(t), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "n_flagged": int(y_pred.sum()),
        }

    rows = [_row("F1-optimal (deployed default)", thresholds[best_idx])]
    for target in (0.9, 0.6):
        candidates = [(t, r) for t, r in zip(thresholds, recall[:-1]) if r >= target]
        t = max(candidates, key=lambda tr: tr[0])[0] if candidates else thresholds.min()
        label = f"~{int(target*100)}% recall"
        rows.append(_row(label, t))
    return rows, thresholds[best_idx]


def render():
    st.subheader("Model comparison")
    st.caption(
        "All three models were trained with zero access to the fraud label. "
        "This comparison is the ONLY place labels are used -- for evaluation, after the fact."
    )

    try:
        comparison_df = api.get_model_comparison()
        scores_df = api.get_model_scores()
    except Exception as e:
        st.error(f"Couldn't load evaluation data from the backend: {e}")
        return

    st.markdown("#### Test-set metrics (K = number of true fraud cases)")
    st.dataframe(
        comparison_df.style.format({"pr_auc": "{:.3f}", "precision@k": "{:.3f}", "recall@k": "{:.3f}", "f1@k": "{:.3f}"}),
        width='stretch',
    )

    st.info(
        "**Isolation Forest was deployed, not the originally-planned autoencoder.** "
        "A real preprocessing issue was found during evaluation (large legitimate "
        "purchases were distorting the autoencoder's error) and fixed, but even after "
        "the fix, Isolation Forest measurably outperformed it here -- PR-AUC 0.23 vs "
        "0.02. The autoencoder stays trained above as the comparison baseline."
    )

    y_true = scores_df["Class"].values

    st.markdown("#### Precision-Recall curves")
    fig = go.Figure()
    for name, col in MODEL_SCORE_COLUMNS.items():
        precision, recall, _ = precision_recall_curve(y_true, scores_df[col].values)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name))
    baseline = float(y_true.mean())
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray", annotation_text=f"random baseline ({baseline:.4f})")
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=450)
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Deployment threshold tradeoff (Isolation Forest)")
    st.caption(
        "The threshold is a business tradeoff -- not a fixed correct number. "
        "Moving it left catches more fraud at the cost of more false alarms; "
        "moving it right does the opposite."
    )
    rows, _ = _threshold_table(y_true, scores_df["iso_forest_score"].values)
    st.dataframe(rows, width='stretch')

    with st.expander("Autoencoder reconstruction error by true class (comparison baseline)"):
        fig_ae = go.Figure()
        for cls, label, color in [(0, "Legitimate", "#1f77b4"), (1, "Fraud", "#d62728")]:
            fig_ae.add_trace(
                go.Histogram(
                    x=scores_df[scores_df["Class"] == cls]["autoencoder_score"],
                    name=label,
                    opacity=0.6,
                    histnorm="probability density",
                    marker_color=color,
                )
            )
        fig_ae.update_layout(barmode="overlay", xaxis_title="reconstruction error (MSE)", height=400, xaxis_range=[0, 5])
        st.plotly_chart(fig_ae, width='stretch')
        st.caption("Fraud skews right, but overlaps heavily with legitimate transactions -- a big part of why it ranks worse than Isolation Forest.")
