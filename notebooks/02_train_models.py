# %% [markdown]
# # Phase 2 -- Train unsupervised models (no labels)
#
# Three models, all trained on `X` only -- none of them ever see `Class`
# during fitting. We compute a continuous anomaly score per transaction
# from each; Phase 3 is where we finally look at `Class` to check how well
# that scoring worked.
#
# - **Isolation Forest** -- isolates points via random recursive splits;
#   anomalies need fewer splits to isolate, so they get a higher score.
#   Scales well (roughly linear), so it trains on the full training set.
# - **One-Class SVM** -- learns a boundary around the "normal" region in
#   feature space. Training is O(n^2)-O(n^3), so on ~228K rows it's
#   impractically slow (and memory-hungry) -- we train it on a random
#   subsample instead and say so plainly, rather than quietly waiting
#   an hour.
# - **Autoencoder (PyTorch)** -- the model we'll actually deploy in the
#   live demo. Reconstruction error is the anomaly score.

# %%
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import preprocessing as prep  # noqa: E402
from src.autoencoder import Autoencoder, reconstruction_error, save_model  # noqa: E402

torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## 1. Load processed data and split train/test
#
# Unstratified, fixed-seed split (see `preprocessing.train_test_split_df`
# docstring for why). Fraud rows are NOT removed from training -- an
# unsupervised model has no way to know which rows to remove.

# %%
df = prep.load_processed()
train_df, test_df = prep.train_test_split_df(df, test_size=0.2, random_state=42)

X_train, y_train = prep.split_features_and_label(train_df)
X_test, y_test = prep.split_features_and_label(test_df)

print(f"train: {X_train.shape}, fraud rows in train: {int(y_train.sum())} ({y_train.mean()*100:.4f}%)")
print(f"test:  {X_test.shape}, fraud rows in test:  {int(y_test.sum())} ({y_test.mean()*100:.4f}%)")
print("y_train is computed above only to PRINT this sanity check -- it is never passed to a .fit() call below.")

# %% [markdown]
# ## 2. Isolation Forest
#
# `contamination` is left at `"auto"` rather than set to the known fraud
# rate -- setting it to ~0.0017 would be quietly leaking label knowledge
# into the model. We also never call `.predict()` (which is what
# `contamination` actually affects); we use the continuous
# `score_samples` output as the anomaly score, so `contamination` doesn't
# influence our results at all here.

# %%
iso_forest = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)

t0 = time.time()
iso_forest.fit(X_train)
print(f"Isolation Forest trained in {time.time()-t0:.1f}s")

# score_samples: higher = more normal. Flip sign so higher = more anomalous,
# consistent with the other two models' score convention.
iso_forest_scores_test = -iso_forest.score_samples(X_test)
print(f"test anomaly score range: [{iso_forest_scores_test.min():.4f}, {iso_forest_scores_test.max():.4f}]")

joblib.dump(iso_forest, PROJECT_ROOT / "models" / "isolation_forest.joblib")

# %% [markdown]
# ## 3. One-Class SVM
#
# `nu` (~upper bound on the fraction of training points allowed to be
# margin errors / outliers) is set to a generic small value used commonly
# as a starting point for this model -- **not** derived from the actual
# 0.0017 fraud rate, which we're not allowed to look at yet. It's a model
# hyperparameter guess, same as picking `n_estimators=200` above.
#
# Training subsample: One-Class SVM's RBF kernel is O(n^2)-O(n^3) in time
# and O(n^2) in memory. At ~228K training rows that's not just slow, it's
# not going to finish. We fit on a random 10,000-row subsample of the
# training set instead. This is a real limitation of the algorithm, not a
# free lunch -- it sees far less data than the other two models, and
# that's part of why we compare all three fairly in Phase 3 rather than
# assuming any one of them is best a priori.

# %%
OCSVM_SUBSAMPLE_SIZE = 10_000
rng = np.random.RandomState(42)
subsample_idx = rng.choice(len(X_train), size=OCSVM_SUBSAMPLE_SIZE, replace=False)
X_train_subsample = X_train.iloc[subsample_idx]

ocsvm = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")

t0 = time.time()
ocsvm.fit(X_train_subsample)
print(f"One-Class SVM trained on {OCSVM_SUBSAMPLE_SIZE} rows in {time.time()-t0:.1f}s")

# decision_function: higher = more normal (distance inside the boundary).
# Flip sign, same convention as Isolation Forest above.
t0 = time.time()
ocsvm_scores_test = -ocsvm.decision_function(X_test)
print(f"scored {len(X_test)} test rows in {time.time()-t0:.1f}s")
print(f"test anomaly score range: [{ocsvm_scores_test.min():.4f}, {ocsvm_scores_test.max():.4f}]")

joblib.dump(ocsvm, PROJECT_ROOT / "models" / "one_class_svm.joblib")

# %% [markdown]
# ## 4. Autoencoder (PyTorch) -- the model we'll deploy
#
# Trained with plain MSE reconstruction loss on `X_train` only. No labels
# anywhere in the loop -- not in the loss, not in early stopping (we just
# train for a fixed number of epochs and inspect the loss curve).

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"training on {DEVICE}")

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=256, shuffle=True)

model = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

N_EPOCHS = 15
history = []

t0 = time.time()
for epoch in range(1, N_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for (batch,) in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        reconstructed = model(batch)
        loss = loss_fn(reconstructed, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)
    epoch_loss /= len(X_train_tensor)
    history.append(epoch_loss)
    print(f"epoch {epoch:2d}/{N_EPOCHS}  train MSE: {epoch_loss:.5f}")

print(f"autoencoder trained in {time.time()-t0:.1f}s")

save_model(model.cpu())
print(f"autoencoder weights saved to {Path('models/autoencoder.pt')}")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(range(1, N_EPOCHS + 1), history, marker="o")
ax.set_xlabel("epoch")
ax.set_ylabel("train reconstruction MSE")
ax.set_title("Autoencoder training loss")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "notebooks" / "figures" / "autoencoder_training_loss.png", dpi=120)
plt.show()

# %%
model.eval()
autoencoder_scores_test = reconstruction_error(model, X_test_tensor).numpy()
print(f"test anomaly score range: [{autoencoder_scores_test.min():.6f}, {autoencoder_scores_test.max():.6f}]")

# %% [markdown]
# ## 5. Save per-transaction anomaly scores for all three models
#
# One row per test-set transaction, one score column per model, plus
# `Class` -- kept here as a separate, clearly-labeled column so Phase 3
# can evaluate, while every cell above this one never referenced it.

# %%
scores_df = test_df[["Time", "Amount", "Class"]].copy()
scores_df["iso_forest_score"] = iso_forest_scores_test
scores_df["ocsvm_score"] = ocsvm_scores_test
scores_df["autoencoder_score"] = autoencoder_scores_test

scores_path = PROJECT_ROOT / "data" / "processed" / "test_scores.csv"
scores_df.to_csv(scores_path, index=False)
print(f"scores saved to {scores_path}")
scores_df.head()

# %% [markdown]
# ## Summary
#
# - All three models trained/scored with zero access to `Class`.
# - Isolation Forest: fast, trained on the full training set.
# - One-Class SVM: trained on a 10K-row subsample due to O(n^2)-O(n^3)
#   scaling -- a real tradeoff, not a shortcut we can ignore in Phase 3.
# - Autoencoder: trained on the full training set, will be the model
#   served in the live demo.
# - Anomaly scores for the test set saved to
#   `data/processed/test_scores.csv`.
#
# **Next: Phase 3 -- finally look at `Class`, evaluate all three with
# precision/recall/F1/PR-AUC, pick a deployment threshold for the
# autoencoder, and choose which model to deploy.**
