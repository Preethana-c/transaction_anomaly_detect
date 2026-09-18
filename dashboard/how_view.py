from __future__ import annotations

import numpy as np
import streamlit as st
from sklearn.metrics import f1_score, precision_score, recall_score

from dashboard import api_client as api


def render():
    st.subheader("How this works")

    st.markdown(
        """
**No one told this model what fraud looks like.**

Out of every 1,000 transactions in this dataset, fewer than 2 are fraud.
Waiting for confirmed fraud labels to train on is slow, expensive, and by
the time you have enough of them the fraud patterns have often already
moved on. So instead of teaching a model "here's what fraud looks like,"
this system teaches it "here's what a *normal* transaction looks like" --
using nothing but the transactions themselves, no fraud/not-fraud
answer key at all -- and flags whatever doesn't fit that pattern.

### The model: Isolation Forest

Imagine repeatedly slicing the data in half with random cuts -- pick a
random feature (say, transaction amount), pick a random split point, and
divide the data into two groups. Keep doing that, again and again.

A typical, unremarkable transaction looks like a lot of other
transactions, so it takes many random cuts before it ends up alone in
its own little group. A transaction that's unusual -- an amount nobody
else has, a combination of features nobody else has -- tends to get
isolated in just a handful of cuts, because there's nothing else nearby
to keep grouping it with.

**The anomaly score is basically: how few cuts did it take to isolate
this transaction?** Fewer cuts -> higher anomaly score -> more likely to
get flagged.

### Why not just flag every big purchase?

An earlier version of this demo's synthetic "obviously weird" test
transactions only changed the dollar amount -- and the model didn't
flag a single one. It turns out a large purchase alone isn't that
unusual across 30 features; the model only reacts when *many* features
look strange together, not just one. That's a feature, not a bug: a
model that flagged every large legitimate purchase would be useless in
practice, buried in false alarms.

### The threshold is a business decision, not a fact

The model outputs a *score*, not a yes/no answer. Turning that score
into "flag this" or "don't" means picking a cutoff -- and where you put
that cutoff is a tradeoff, not something the data can decide for you:
        """
    )

    try:
        scores_df = api.get_model_scores()
    except Exception as e:
        st.warning(f"Couldn't load scores for the interactive demo below: {e}")
        return

    y_true = scores_df["Class"].values
    scores = scores_df["iso_forest_score"].values

    lo, hi = float(scores.min()), float(np.percentile(scores, 99.9))
    default = float(np.median(scores) + (hi - float(np.median(scores))) * 0.3)
    t = st.slider(
        "Try moving the threshold yourself:",
        min_value=lo, max_value=hi, value=default, step=(hi - lo) / 200,
        format="%.3f",
    )

    y_pred = (scores >= t).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    n_flagged = int(y_pred.sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions flagged", n_flagged)
    c2.metric("Precision", f"{precision*100:.1f}%", help="Of everything flagged, what % is actually fraud")
    c3.metric("Recall", f"{recall*100:.1f}%", help="Of all real fraud, what % did we catch")
    c4.metric("F1", f"{f1:.3f}")

    if precision < 0.3 and n_flagged > 20:
        st.caption("⬅️ Low threshold: catches more fraud, but a review team would drown in false alarms.")
    elif recall < 0.15:
        st.caption("➡️ High threshold: very few false alarms, but most fraud slips through uncaught.")
    else:
        st.caption("A middle ground -- roughly where the deployed default sits.")

    st.markdown(
        """
### What was tried and set aside

An autoencoder (a neural network trained to reconstruct normal
transactions, flagging whatever it reconstructs poorly) was also built
and evaluated -- see the **Model Comparison** tab. It's a legitimate,
commonly-used approach, but it measurably underperformed Isolation
Forest on this specific dataset, so it wasn't the one deployed here.
Being able to say "I tried X, measured it honestly, and it lost to Y" is
a more defensible engineering story than only ever showing the model
that won.
        """
    )
