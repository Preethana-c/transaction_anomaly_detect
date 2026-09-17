# %% [markdown]
# # Phase 1 -- EDA & Preprocessing
#
# Dataset: Kaggle "Credit Card Fraud Detection" (`mlg-ulb/creditcardfraud`).
# 284,807 transactions made by European cardholders in September 2013,
# over two days. Features `V1`..`V28` are the output of a PCA transform
# applied to the original (undisclosed, sensitive) features. `Time` and
# `Amount` are the only two raw, un-transformed columns.
#
# See `README.md` -> "Getting the data" for download instructions if
# `data/raw/creditcard.csv` isn't present yet.
#
# ## Why this matters: the unsupervised framing
#
# `Class` (1 = fraud, 0 = legitimate) exists in this dataset, but in
# production fraud detection you very rarely have reliable, timely labels:
# confirmed fraud is discovered weeks later (if ever), labeling is
# expensive, and fraud patterns drift faster than any labeled dataset can
# be kept current. So the right framing is **unsupervised anomaly
# detection**: learn what "normal" transactions look like, and flag
# whatever doesn't fit -- without ever telling the model which rows were
# fraud.
#
# **Rule for this whole project: `Class` is not used for training, feature
# selection, or hyperparameter tuning of any kind. It is used exactly
# once, in Phase 3, to evaluate models that were trained blind to it.**

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Make `src` importable whether this is run from repo root or notebooks/
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import preprocessing as prep  # noqa: E402

sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Load the raw data

# %%
df = prep.load_raw()
print(f"shape: {df.shape}")
df.head()

# %%
df.info()

# %%
# Sanity check: no missing values expected in this dataset.
print("missing values per column:")
print(df.isna().sum().sum(), "total nulls")

# %% [markdown]
# ## 2. Class imbalance
#
# This is the defining characteristic of the problem. We look at it now
# purely to understand the data -- NOT to use it in training.

# %%
class_counts = df["Class"].value_counts()
class_pct = df["Class"].value_counts(normalize=True) * 100

print("Class counts:")
print(class_counts)
print("\nClass percentage:")
print(class_pct.round(4))

fraud_rate = class_pct.loc[1]
print(f"\nFraud rate: {fraud_rate:.4f}% ({class_counts.loc[1]} of {len(df)} transactions)")

# %% [markdown]
# ~0.17% fraud. This is realistic for real-world payment fraud -- and it's
# exactly why we don't (and shouldn't) artificially balance the classes
# with SMOTE/undersampling here: doing so would teach a model a fraud rate
# that doesn't exist in production traffic, and more importantly would
# require using the labels during training, which defeats the point of an
# unsupervised approach.

# %%
fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
ax.set_yscale("log")
ax.set_ylabel("count (log scale)")
ax.set_title(f"Class imbalance -- fraud is {fraud_rate:.3f}% of transactions")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "notebooks" / "figures" / "class_imbalance.png", dpi=120)
plt.show()

# %% [markdown]
# ## 3. Look at `Time` and `Amount` (the only unscaled columns)
#
# `V1`..`V28` are already PCA components with roughly zero mean and unit
# variance -- no further scaling needed or wanted (scaling them again, or
# differently, would distort the PCA basis). `Time` (seconds since the
# first transaction) and `Amount` (transaction value in dollars) are on
# their own raw scales and need standardizing before they go into a
# distance/reconstruction-based model like an autoencoder or One-Class SVM.

# %%
df[["Time", "Amount"]].describe()

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["Time"], bins=50, ax=axes[0])
axes[0].set_title("Time (seconds since first transaction)")
sns.histplot(df["Amount"], bins=50, ax=axes[1])
axes[1].set_yscale("log")
axes[1].set_title("Amount ($, log-count)")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "notebooks" / "figures" / "time_amount_raw.png", dpi=120)
plt.show()

# %% [markdown]
# `Amount` is heavily right-skewed (mostly small transactions, a long tail
# of large ones, up to five figures) -- another reason it needs scaling,
# since raw unscaled `Amount` would otherwise dominate distance-based
# scoring just because of its larger numeric range.
#
# It needs more than plain standardizing, though: a handful of very large
# (but perfectly legitimate) transactions in that tail turn into huge
# z-scores under a plain `StandardScaler`, which -- as Phase 3's
# evaluation caught -- ends up dominating the autoencoder's reconstruction
# error and makes it flag big purchases as "anomalous" for the wrong
# reason. So `Amount` gets `log1p`-transformed *before* standardizing,
# compressing that tail. This is purely a transform of `Amount`'s own
# distribution (see the skew above) -- it never touches `Class`.

# %% [markdown]
# ## 4. Scale `Time` and `Amount`
#
# We fit a `StandardScaler` on `Time` and `log1p(Amount)`. This fit uses
# feature values alone -- `Class` never enters this computation. The
# fitted scaler is saved to `models/scaler.joblib` so the exact same
# transform can be reapplied later, at inference time, in the FastAPI
# streaming service (Phase 4). Training a model on one scaling and then
# serving it with a different one is a classic, silent way to make a
# deployed model wrong -- saving the fitted scaler avoids that.

# %%
scaler = prep.fit_scaler(df)
df_processed = prep.apply_scaler(df, scaler)

prep.save_scaler(scaler)
print(f"scaler saved to {prep.SCALER_PATH}")

df_processed[["Time", "Time_scaled", "Amount", "Amount_scaled"]].describe()

# %% [markdown]
# ## 5. Save the processed dataset
#
# We keep `Class` in the saved file (evaluation needs it later) but it
# stays clearly separated: `src/preprocessing.split_features_and_label`
# is the only place that reads it back out, and every training script in
# Phase 2 calls that function and discards `y` immediately.

# %%
prep.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df_processed.to_csv(prep.PROCESSED_DATA_PATH, index=False)
print(f"processed data saved to {prep.PROCESSED_DATA_PATH}")
print(f"columns: {list(df_processed.columns)}")

# %%
X, y = prep.split_features_and_label(df_processed)
print(f"model input features ({len(prep.FEATURE_COLUMNS)}): {prep.FEATURE_COLUMNS}")
print(f"X shape: {X.shape}  (this is all Phase 2 training code will ever see)")
print(f"y shape: {y.shape}  (held aside, untouched until Phase 3 evaluation)")

# %% [markdown]
# ## Summary
#
# - 284,807 transactions, 492 fraud (~0.173%) -- realistic, extreme
#   imbalance, kept as-is.
# - `V1`-`V28` already scaled (PCA output) -- left untouched.
# - `Time` and `Amount` standardized; scaler persisted for reuse at
#   inference time.
# - `Class` is set aside. From this point forward it is evaluation-only.
#
# **Next: Phase 2 -- train Isolation Forest, One-Class SVM, and a PyTorch
# autoencoder on `X` only.**
