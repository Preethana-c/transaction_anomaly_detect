# Transaction Anomaly Detector

A real-time, **unsupervised** fraud/anomaly detection system for credit
card transactions, with a live-streaming demo (FastAPI + Supabase +
Streamlit), not just a static notebook.

> Status: Phase 1 (EDA & preprocessing) complete. See project phases below.

## Why unsupervised?

In real fraud detection, reliable labels are scarce: confirmed fraud is
often discovered weeks after the transaction (if ever), manual review is
expensive, and fraud patterns drift faster than any labeled dataset stays
current. This project trains anomaly detectors that never see the fraud
label during training -- they only learn what a "normal" transaction looks
like, and score deviation from that. The label is used exactly once, after
training, purely to check how well the unsupervised approach did.

## Getting the data

This project uses the Kaggle **Credit Card Fraud Detection** dataset
(`mlg-ulb/creditcardfraud`): 284,807 anonymized European card transactions
from September 2013, with PCA features `V1`-`V28`, plus `Time`, `Amount`,
and the `Class` label (1 = fraud, 0 = legitimate, ~0.17% fraud).

**Option A -- Kaggle CLI (recommended):**

```bash
pip install kaggle
# Get an API token: Kaggle account settings -> "Create New Token"
# This downloads kaggle.json -- place it at:
#   Windows: C:\Users\<you>\.kaggle\kaggle.json
#   macOS/Linux: ~/.kaggle/kaggle.json

kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

**Option B -- manual download:**

1. Go to the dataset page on Kaggle and download `creditcard.csv`
   (search "Credit Card Fraud Detection mlg-ulb" on kaggle.com).
2. Place the file at `data/raw/creditcard.csv` in this repo.

Either way, verify you end up with:

```
data/raw/creditcard.csv   (~150 MB, 284,807 rows)
```

`data/raw/` and `data/processed/` are gitignored -- the raw dataset is
never committed to this repo.

## Project structure

```
src/          reusable pipeline code (preprocessing, shared by notebooks and API)
notebooks/    EDA and model training (Phase 1-3), jupytext percent-format .py
api/          FastAPI streaming backend (Phase 4)
dashboard/    Streamlit live dashboard (Phase 5)
data/         raw/processed data (gitignored)
models/       fitted scaler + trained models (gitignored)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Project phases

1. **EDA & preprocessing** -- load data, examine class imbalance, scale
   `Time`/`Amount` only (`notebooks/01_eda_preprocessing.py`).
2. **Unsupervised training** -- Isolation Forest, One-Class SVM, PyTorch
   autoencoder, all trained without labels.
3. **Evaluation** -- labels used only now: precision/recall/F1/PR-AUC per
   model, threshold selection, model comparison and selection.
4. **Streaming backend** -- FastAPI service replaying transactions live,
   scoring with the trained autoencoder, logging to Supabase.
5. **Dashboard** -- Streamlit live feed + model comparison + plain-language
   explanation of reconstruction error.
6. **Deployment** -- Render (backend) + Streamlit Community Cloud (dashboard).

(Sections below are filled in as each phase lands.)
