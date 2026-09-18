# %% [markdown]
# # Phase 6 prep -- build the demo-stream artifact for deployment
#
# `api/streamer.py`'s simulated live feed only needs a few hundred
# transactions (every fraud row in the test set + a random sample of
# legitimate ones -- see that file's docstring for why). Computing that
# from the full processed dataset at runtime meant the deployed backend
# needed the entire ~155MB `creditcard_processed.csv` just to serve a
# ~1,000-row demo -- too large to commit to git, and not something we
# want the deployed Render service downloading/regenerating on every
# cold start.
#
# This notebook computes that small subset once and saves it as its own
# small CSV, which *is* committed to the repo. `api/streamer.py` loads it
# directly -- no dependency on the full dataset at serve time at all.

# %%
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api import config  # noqa: E402
from src import preprocessing as prep  # noqa: E402

# %%
df = prep.load_processed()
_, test_df = prep.train_test_split_df(df, test_size=0.2, random_state=42)

fraud_rows = test_df[test_df["Class"] == 1]
legit_rows = test_df[test_df["Class"] == 0].sample(
    n=min(config.DEMO_N_LEGIT, len(test_df)), random_state=config.DEMO_SEED
)
demo_df = pd.concat([fraud_rows, legit_rows]).sample(frac=1, random_state=config.DEMO_SEED).reset_index(drop=True)

print(f"demo stream: {len(demo_df)} rows ({len(fraud_rows)} fraud + {len(legit_rows)} legitimate)")

# %%
out_path = PROJECT_ROOT / "data" / "processed" / "demo_stream.csv"
demo_df.to_csv(out_path, index=False)
print(f"saved to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

# %% [markdown]
# Committed to git (unlike `creditcard_processed.csv`) -- small enough
# (a few hundred KB) and it's exactly, only what the live demo needs.
