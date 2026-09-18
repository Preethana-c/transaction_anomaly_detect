"""
Backend configuration, read from environment variables (a `.env` file in
the repo root is loaded automatically for local dev -- see `.env.example`).

Supabase is optional at runtime: if `SUPABASE_URL`/`SUPABASE_KEY` aren't
set, the backend falls back to an in-memory transaction log so the API
can be developed and tested without a Supabase project. Deploying for
real requires both set -- see README.md "Deploying the backend".
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "transactions")
SUPABASE_CONFIGURED = bool(SUPABASE_URL and SUPABASE_KEY)

# Demo stream composition: every fraud row in the test set, plus a random
# sample of legitimate rows, shuffled together -- see api/streamer.py for
# why we don't replay all ~57K test rows verbatim.
DEMO_N_LEGIT = int(os.getenv("DEMO_N_LEGIT", "900"))
DEMO_SEED = int(os.getenv("DEMO_SEED", "42"))

DEFAULT_STREAM_DELAY_SECONDS = float(os.getenv("STREAM_DELAY_SECONDS", "0.3"))
SYNTHETIC_ANOMALY_EVERY_N = int(os.getenv("SYNTHETIC_ANOMALY_EVERY_N", "50"))

# CORS: the deployed Streamlit dashboard's origin, so it can call this API
# from the browser. "*" is fine for this demo (read-mostly, no auth secrets
# exposed) but is intentionally an env var so it's not hardcoded.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
