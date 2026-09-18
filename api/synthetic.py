"""
Synthetic obvious-anomaly transactions injected into the demo stream.

Real fraud in the test set is rare (98 rows) and its anomaly signal is
subtle -- that's the whole point of the model. For a *demo*, we also want
moments where detection is unmistakable to watch happen live, so every
`SYNTHETIC_ANOMALY_EVERY_N` transactions we inject one that's extreme on
purpose.

An extreme `Amount` alone turned out NOT to be enough: empirically, an
Amount-only outlier scored well below the deployed Isolation Forest's
threshold on every test sample tried (0/20 flagged) -- which makes sense
in hindsight, since Isolation Forest not being fooled by amount alone is
*why* it beat the autoencoder in Phase 3. So the synthetic transaction
also perturbs every PCA feature to an extreme value; that combination
was flagged 20/20 in testing, comfortably above threshold. It's a
deliberately unsubtle "everything about this transaction is weird" case
-- distinct from real fraud, which the model has to work much harder for.
"""

from __future__ import annotations

import random

import pandas as pd

from src import preprocessing as prep

# How many standard deviations out each perturbed PCA feature is pushed.
# V1-V28 are ~zero-mean, ~unit-variance PCA components, so this is
# comfortably outside the range any real transaction occupies.
PERTURBATION_MAGNITUDE = (14.0, 20.0)


def make_synthetic_transaction(base_row: pd.Series, synthetic_time: float) -> dict:
    row = base_row[prep.PCA_COLUMNS].to_dict()
    row["Time"] = synthetic_time
    row["Amount"] = round(random.uniform(15_000, 99_999), 2)
    for col in prep.PCA_COLUMNS:
        sign = random.choice([-1, 1])
        row[col] = sign * random.uniform(*PERTURBATION_MAGNITUDE)
    return row
