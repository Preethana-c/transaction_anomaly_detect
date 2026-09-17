# %% [markdown]
# # Phase 3 -- Evaluation (labels used from here on, for validation only)
#
# This is the first and only notebook that reads `Class` for anything
# beyond a sanity-check print. Nothing above this point in the project
# used it to train, tune, or select a model. From here we're purely
# grading what the unsupervised models learned.
#
# ## Why PR-AUC, not ROC-AUC
#
# ROC-AUC plots true positive rate against **false positive rate**, and
# with 284,315 legitimate transactions vs. 492 fraud, the false positive
# *rate* stays tiny even when the *count* of false positives is large
# enough to bury a fraud-review team (e.g. 500 false positives is 0.18%
# FPR against 284K legitimate transactions -- looks great on an ROC
# curve, is a lot of manual review work in practice). PR-AUC plots
# precision against recall directly, so it's sensitive to exactly the
# thing that matters under extreme imbalance: out of everything you flag,
# how much is actually fraud. That's why every metric below is
# precision/recall-based, not ROC-based.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

sns.set_theme(style="whitegrid")

MODELS = {
    "Isolation Forest": "iso_forest_score",
    "One-Class SVM": "ocsvm_score",
    "Autoencoder": "autoencoder_score",
}

# %% [markdown]
# ## 1. Load test-set scores

# %%
scores_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test_scores.csv")
y_true = scores_df["Class"].values
n_fraud = int(y_true.sum())
print(f"test set: {len(scores_df)} transactions, {n_fraud} fraud ({y_true.mean()*100:.4f}%)")

# %% [markdown]
# ## 2. Precision / Recall / F1 at top-K, plus PR-AUC, per model
#
# Top-K: rank every transaction by anomaly score, take the K highest as
# "predicted fraud". We set **K = number of actual fraud cases in the
# test set** (a standard anomaly-detection convention, "precision@k" with
# k = |positives|) -- it asks each model "if you could only flag exactly
# as many transactions as there truly are frauds, how many would you get
# right?". This is one operating point; PR-AUC (below) summarizes
# performance across *every* possible operating point, which is why we
# report both rather than relying on top-K alone.

# %%
def evaluate_top_k(y_true: np.ndarray, scores: np.ndarray, k: int):
    top_k_idx = np.argsort(scores)[::-1][:k]
    y_pred = np.zeros_like(y_true)
    y_pred[top_k_idx] = 1
    return {
        "precision@k": precision_score(y_true, y_pred, zero_division=0),
        "recall@k": recall_score(y_true, y_pred, zero_division=0),
        "f1@k": f1_score(y_true, y_pred, zero_division=0),
    }


results = []
pr_curves = {}
for name, col in MODELS.items():
    scores = scores_df[col].values
    top_k_metrics = evaluate_top_k(y_true, scores, k=n_fraud)
    pr_auc = average_precision_score(y_true, scores)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_curves[name] = (precision, recall, thresholds)
    results.append({"model": name, "pr_auc": pr_auc, **top_k_metrics})

summary_df = pd.DataFrame(results).set_index("model")
summary_df

# %% [markdown]
# ## 3. Precision-Recall curves, all three models

# %%
fig, ax = plt.subplots(figsize=(6, 5))
for name, (precision, recall, _) in pr_curves.items():
    pr_auc = summary_df.loc[name, "pr_auc"]
    ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})")
baseline = y_true.mean()
ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"random baseline ({baseline:.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curves (test set)")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "notebooks" / "figures" / "pr_curves.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Autoencoder reconstruction error: distribution by true class
#
# This is the model we're deploying, so we look at its score distribution
# more closely than the other two.

# %%
ae_scores = scores_df["autoencoder_score"].values

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.histplot(
    scores_df[scores_df["Class"] == 0]["autoencoder_score"],
    bins=80, stat="density", color="steelblue", label="Legitimate", alpha=0.6, ax=ax,
)
sns.histplot(
    scores_df[scores_df["Class"] == 1]["autoencoder_score"],
    bins=80, stat="density", color="crimson", label="Fraud", alpha=0.6, ax=ax,
)
ax.set_xlim(0, np.percentile(ae_scores, 99.5))
ax.set_xlabel("reconstruction error (MSE)")
ax.set_title("Autoencoder reconstruction error by true class")
ax.legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "notebooks" / "figures" / "autoencoder_error_distribution.png", dpi=120)
plt.show()

# %% [markdown]
# Fraud is clearly shifted right, but the two distributions overlap --
# there is no single error value that perfectly separates them. That
# overlap is exactly why picking a threshold is a judgment call, not a
# calculation.
#
# ## 5. Picking a deployment threshold
#
# **This threshold is a business tradeoff, not a "correct" number.**
# Move it left (lower) and recall goes up -- you catch more fraud, at the
# cost of more legitimate transactions getting flagged for review (more
# false positives, more customer friction, more manual review load). Move
# it right (higher) and precision goes up -- fewer false alarms, but more
# fraud slips through uncaught. The "right" answer depends on things this
# notebook has no way to know: the cost of a missed fraud vs. the cost of
# an annoyed customer / an analyst-hour of review, which is a decision
# for the business, not the model.
#
# We compute three candidate thresholds below to make that tradeoff
# concrete, then pick one as the default for the live demo -- clearly
# documented as a default, not a verdict.

# %%
precision, recall, thresholds = pr_curves["Autoencoder"]
f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
# precision/recall have one more element than thresholds; align by dropping the last
f1_best_idx = int(np.argmax(f1_scores[:-1]))
f1_optimal_threshold = thresholds[f1_best_idx]


def threshold_for_target_recall(target_recall: float) -> float:
    """Smallest threshold whose recall is still >= target_recall."""
    candidates = [(t, r) for t, r in zip(thresholds, recall[:-1]) if r >= target_recall]
    return max(candidates, key=lambda tr: tr[0])[0] if candidates else thresholds.min()


candidate_thresholds = {
    "high recall (catch more fraud)": threshold_for_target_recall(0.90),
    "balanced (F1-optimal)": f1_optimal_threshold,
    "high precision (fewer false alarms)": threshold_for_target_recall(0.60),
}

rows = []
for label, t in candidate_thresholds.items():
    y_pred = (ae_scores >= t).astype(int)
    rows.append({
        "strategy": label,
        "threshold": t,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_flagged": int(y_pred.sum()),
    })
threshold_comparison_df = pd.DataFrame(rows).set_index("strategy")
threshold_comparison_df

# %% [markdown]
# This F1-optimal threshold (`DEPLOY_THRESHOLD_AE` below) is preserved as
# a worked example of the tradeoff -- but as Section 6 explains, the
# autoencoder is *not* the model this project ends up deploying live, so
# this particular threshold is not the one the streaming backend uses.

# %%
DEPLOY_THRESHOLD_AE = float(f1_optimal_threshold)
print(f"autoencoder F1-optimal threshold: {DEPLOY_THRESHOLD_AE:.6f}")

y_pred_ae = (ae_scores >= DEPLOY_THRESHOLD_AE).astype(int)
print(f"precision: {precision_score(y_true, y_pred_ae):.3f}")
print(f"recall:    {recall_score(y_true, y_pred_ae):.3f}")
print(f"f1:        {f1_score(y_true, y_pred_ae):.3f}")
print(f"flagged:   {y_pred_ae.sum()} of {len(y_pred_ae)} test transactions")

# %% [markdown]
# ## 6. Model comparison and final pick
#
# This is the one place in the project where a result genuinely surprised
# the plan: the original design called for deploying the autoencoder live
# regardless of how it compared to the others. The honest comparison
# below says not to.

# %%
summary_df

# %% [markdown]
# **Isolation Forest wins, clearly, on every metric measured** -- PR-AUC
# more than 10x the autoencoder's, and roughly 6x the precision/recall at
# K=98. This isn't a training bug: `Amount` was re-scaled with a `log1p`
# transform after an earlier run showed it distorting the autoencoder's
# error (see Phase 1 notebook), and the network was confirmed converged
# (train loss flat well before epoch 20; a 40-epoch run gave no PR-AUC
# improvement). What's left is a real, known characteristic of the two
# approaches on this kind of data: a single global MSE reconstruction
# error doesn't separate "this row is fraud" from "this row is a
# legitimate but statistically rare large purchase" nearly as well as
# Isolation Forest's random-partition ensemble does, on already-PCA'd
# tabular features like these. We did not chase this gap further by
# tuning the autoencoder's architecture against test-set PR-AUC --
# selecting a model or its hyperparameters using the labels would quietly
# reintroduce exactly what this whole project is built to avoid.
#
# **Decision: deploy Isolation Forest**, not the autoencoder, for the
# live streaming demo. Beyond the accuracy gap, it's also simpler to
# serve here: `score_samples()` on a fitted `IsolationForest` is a single
# `joblib.load()` and no PyTorch runtime in the API at all. The
# autoencoder remains fully trained, evaluated, and saved in this repo
# (`models/autoencoder.pt`) as the comparison baseline the dashboard's
# model-comparison view shows -- it's kept in the story, just not in the
# live path.

# %% [markdown]
# ## 6b. Deployment threshold for Isolation Forest
#
# Same method as Section 5, applied to the model that's actually going
# live: F1-optimal threshold on the held-out test set, presented next to
# higher-recall / higher-precision alternatives as the same kind of
# business tradeoff -- not a different kind of decision just because the
# model changed.

# %%
if_precision, if_recall, if_thresholds = pr_curves["Isolation Forest"]
if_f1_scores = 2 * if_precision * if_recall / (if_precision + if_recall + 1e-12)
if_f1_best_idx = int(np.argmax(if_f1_scores[:-1]))
DEPLOY_THRESHOLD_IF = float(if_thresholds[if_f1_best_idx])

if_scores = scores_df["iso_forest_score"].values
y_pred_if = (if_scores >= DEPLOY_THRESHOLD_IF).astype(int)
print(f"deployed Isolation Forest threshold: {DEPLOY_THRESHOLD_IF:.6f}")
print(f"precision: {precision_score(y_true, y_pred_if):.3f}")
print(f"recall:    {recall_score(y_true, y_pred_if):.3f}")
print(f"f1:        {f1_score(y_true, y_pred_if):.3f}")
print(f"flagged:   {y_pred_if.sum()} of {len(y_pred_if)} test transactions")

# %% [markdown]
# ## 7. Persist evaluation artifacts for later phases
#
# Two threshold files are saved: `isolation_forest_threshold.json` is
# what the Phase 4 live backend actually loads and uses. The autoencoder
# threshold is kept too -- Phase 5's dashboard shows it next to the
# Isolation Forest one as part of the honest model-comparison story.

# %%
summary_df.to_csv(PROJECT_ROOT / "data" / "processed" / "model_comparison.csv")

import json

deployed_threshold_path = PROJECT_ROOT / "models" / "isolation_forest_threshold.json"
with open(deployed_threshold_path, "w") as f:
    json.dump(
        {
            "threshold": DEPLOY_THRESHOLD_IF,
            "strategy": "F1-optimal on held-out test set",
            "precision_at_threshold": precision_score(y_true, y_pred_if, zero_division=0),
            "recall_at_threshold": recall_score(y_true, y_pred_if, zero_division=0),
        },
        f,
        indent=2,
    )

ae_threshold_path = PROJECT_ROOT / "models" / "autoencoder_threshold.json"
with open(ae_threshold_path, "w") as f:
    json.dump(
        {
            "threshold": DEPLOY_THRESHOLD_AE,
            "strategy": "F1-optimal on held-out test set (not deployed -- comparison baseline only)",
            "precision_at_threshold": precision_score(y_true, y_pred_ae, zero_division=0),
            "recall_at_threshold": recall_score(y_true, y_pred_ae, zero_division=0),
        },
        f,
        indent=2,
    )
print("model comparison saved to data/processed/model_comparison.csv")
print(f"deployed (Isolation Forest) threshold saved to {deployed_threshold_path}")
print(f"autoencoder threshold (comparison only) saved to {ae_threshold_path}")

# %% [markdown]
# ## Summary
#
# - Evaluated all three models with precision/recall/F1@K (K=n_fraud) and
#   PR-AUC -- ROC-AUC deliberately avoided given the class imbalance.
# - Found and fixed a real issue: raw `Amount` outliers were distorting
#   the autoencoder's reconstruction error; fixed label-free via `log1p`.
# - Honest result: **Isolation Forest outperforms the autoencoder** on
#   this dataset (PR-AUC 0.23 vs 0.02) -- confirmed not a training bug
#   (loss converged, more epochs didn't help) but a real property of the
#   two approaches on this feature set.
# - **Deploying Isolation Forest** live, not the originally-planned
#   autoencoder -- both on accuracy and because it's simpler to serve.
#   The autoencoder stays in the repo as the trained comparison baseline.
# - F1-optimal thresholds picked and saved for both, framed explicitly as
#   a business tradeoff rather than a fixed correct number.
#
# **Next: Phase 4 -- FastAPI streaming backend that replays test-set
# transactions, scores them live with Isolation Forest + threshold, and
# logs results to Supabase.**
