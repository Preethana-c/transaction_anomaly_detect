# Transaction Anomaly Detector

A real-time, **unsupervised** fraud/anomaly detection system for credit
card transactions, with a live-streaming demo (FastAPI + Supabase +
Streamlit), not just a static notebook.

> Status: all 6 phases complete (EDA, training, evaluation, streaming
> backend, dashboard, deployment-ready). See project phases below.

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

`data/raw/` is gitignored -- the raw dataset is never committed to this
repo. Most of `data/processed/` and all of `models/` ARE committed
(they're small -- a few hundred KB to ~2MB each); only the full
`creditcard_processed.csv` (~155MB) is excluded. See "Deploying the
backend" below for why that split matters.

## Project structure

```
src/          reusable pipeline code (preprocessing, shared by notebooks and API)
notebooks/    EDA, model training, evaluation, deploy-prep (Phase 1-3, 6)
api/          FastAPI streaming backend (Phase 4) + api/requirements.txt (slim, for Render)
dashboard/    Streamlit live dashboard (Phase 5) + dashboard/requirements.txt (slim, for Streamlit Cloud)
data/raw/     the Kaggle CSV (gitignored -- you download it, see above)
data/processed/  committed: model_comparison.csv, test_scores.csv, demo_stream.csv
                 gitignored: creditcard_processed.csv (~155MB, regenerate via notebook 01)
models/       fitted scaler + all trained models + deployment thresholds (committed)
render.yaml   Render Blueprint config for one-click backend deploy
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
   `Time`/`log1p(Amount)` (`notebooks/01_eda_preprocessing.py`).
2. **Unsupervised training** -- Isolation Forest, One-Class SVM, PyTorch
   autoencoder, all trained without labels (`notebooks/02_train_models.py`).
3. **Evaluation** -- labels used only now: precision/recall/F1/PR-AUC per
   model, threshold selection, model comparison
   (`notebooks/03_evaluation.py`). **Result: Isolation Forest was deployed
   instead of the originally-planned autoencoder** -- it measurably
   outperformed it (PR-AUC 0.23 vs 0.02) after a real, label-free
   preprocessing bug (raw `Amount` outliers distorting the autoencoder's
   reconstruction error) was found and fixed. The autoencoder stays
   trained in the repo as the comparison baseline. See the notebook for
   the full writeup.
4. **Streaming backend** -- FastAPI service (`api/`) replaying a curated
   mix of real test-set transactions (every fraud row + a random sample
   of legitimate ones) with a short delay, scoring each live with the
   deployed Isolation Forest, and logging to Supabase. Also injects
   synthetic obvious-anomaly transactions periodically. See "Running the
   backend locally" and "Supabase setup" below.
5. **Dashboard** (`dashboard/`) -- Streamlit live feed + model comparison +
   plain-language explanation of anomaly scoring, with an interactive
   threshold slider. Talks only to the backend over HTTP.
6. **Deployment** -- `notebooks/04_build_demo_stream.py` precomputes the
   small demo-stream artifact the live backend needs (so it never
   depends on the full 155MB dataset at serve time); `api/requirements.txt`,
   `dashboard/requirements.txt`, and `render.yaml` are the slim,
   deployment-specific configs. See "Deploying the backend" and
   "Deploying the dashboard" below for exact steps.

## Supabase setup (for Phase 4/5)

The backend logs every streamed transaction + anomaly score to a
Supabase (Postgres) table. Without Supabase configured, it falls back to
an in-memory log automatically -- fine for local development, but the
deployed demo needs a real project so the dashboard (a separate service)
can read what the backend writes.

1. Create a free project at [supabase.com](https://supabase.com).
2. Open the SQL Editor and run [`api/schema.sql`](api/schema.sql) -- it
   creates the `transactions` table, indexes, and a public-read RLS policy.
3. Go to Project Settings -> API and copy:
   - **Project URL** -> `SUPABASE_URL`
   - **service_role secret key** -> `SUPABASE_KEY` (backend only --
     bypasses RLS, never expose it to the dashboard/browser)
4. Copy `.env.example` to `.env` in the repo root and fill in the values.

The dashboard never talks to Supabase directly -- it only calls the
FastAPI backend's HTTP endpoints (see below), which is what actually
reads/writes the database. So the dashboard needs no Supabase
credentials at all, just the backend's URL.

## Running the backend locally

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in Supabase values, or leave blank for in-memory storage
uvicorn api.main:app --reload
```

Endpoints (interactive docs at `http://127.0.0.1:8000/docs`):

| Endpoint | Description |
|---|---|
| `GET /health` | status + which storage backend is active |
| `POST /stream/start` | start the simulated live feed (`{"delay_seconds": 0.3}`) |
| `POST /stream/stop` | stop it |
| `GET /stream/status` | running state + live counts |
| `GET /transactions/recent?limit=50&flagged_only=false` | recent scored transactions |
| `GET /models/comparison` | Phase 3's model comparison table, for the dashboard |
| `GET /models/scores` | per-transaction test-set scores for all 3 models, for the dashboard's PR curves |

## Running the dashboard locally

With the backend already running (above):

```bash
export API_BASE_URL=http://127.0.0.1:8000   # PowerShell: $env:API_BASE_URL = "http://127.0.0.1:8000"
streamlit run dashboard/app.py
```

Three tabs:

- **Live Feed** -- start/stop the simulated stream, watch transactions and
  anomaly scores arrive in real time, flagged ones highlighted.
- **Model Comparison** -- the Phase 3 evaluation table, interactive PR
  curves for all three models, and the threshold tradeoff table --
  the rigor behind why Isolation Forest was deployed instead of the
  originally-planned autoencoder.
- **How This Works** -- a plain-language explanation of unsupervised
  anomaly detection and Isolation Forest, plus an interactive threshold
  slider so a non-technical viewer can feel the precision/recall tradeoff
  themselves rather than just reading about it.

## Deploying the backend (Render)

The backend is deployed from **`api/requirements.txt`**, not the root
one -- it excludes training-only dependencies (torch, matplotlib,
jupytext) the live API never imports, which keeps the build fast and
small on Render's free tier.

**What Render needs that isn't generated at deploy time:** the trained
model files under `models/` (scaler, Isolation Forest, threshold) and
`data/processed/demo_stream.csv` are all committed to this repo
specifically so the deployed backend can serve without retraining
anything or needing the ~155MB full dataset -- see `.gitignore` for why
only `creditcard_processed.csv` itself is excluded. If you've changed
the model, re-run `notebooks/02`, `03`, and `04` locally and commit the
updated files before deploying.

**Option A -- Blueprint (recommended):** in the Render dashboard, **New
+ -> Blueprint**, point it at this repo. It reads [`render.yaml`](render.yaml)
and creates the service for you; you'll be prompted for `SUPABASE_URL`
and `SUPABASE_KEY` (the two `sync: false` vars in that file).

**Option B -- manual Web Service:** New + -> Web Service, point it at
this repo, then set:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install -r api/requirements.txt` |
| Start Command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |

Then add environment variables `SUPABASE_URL`, `SUPABASE_KEY`,
`SUPABASE_TABLE=transactions`, `ALLOWED_ORIGINS` (set this to your
deployed Streamlit app's URL once you have it, e.g.
`https://your-app.streamlit.app`).

Once it's live, note the Render URL (e.g.
`https://transaction-anomaly-api.onrender.com`) -- the dashboard needs it next.

> Free-tier Render web services spin down after inactivity; the first
> request after idling takes ~30-60s to wake up. That's expected, not a bug.

## Deploying the dashboard (Streamlit Community Cloud)

1. Go to [share.streamlit.io](https://share.streamlit.io), **New app**,
   point it at this repo.
2. **Main file path**: `dashboard/app.py`. Streamlit Cloud auto-detects
   `dashboard/requirements.txt` since it lives alongside the app file --
   you don't need to point at it manually.
3. Under **Advanced settings -> Secrets**, add:
   ```toml
   API_BASE_URL = "https://transaction-anomaly-api.onrender.com"
   ```
   (your actual Render URL from the previous step).
4. Deploy. Once it's live, go back to the Render backend's environment
   variables and set `ALLOWED_ORIGINS` to this app's URL (CORS needs the
   dashboard's real origin, not `*`, once both are public).

That's the full loop: dashboard (Streamlit Cloud) -> backend (Render) ->
Supabase (Postgres) -> back to the dashboard.
