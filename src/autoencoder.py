"""
Autoencoder for unsupervised transaction anomaly scoring.

Defined once here so training (Phase 2), evaluation (Phase 3), and the
live FastAPI backend (Phase 4) all use the identical architecture --
loading a state_dict into a mismatched architecture would fail loudly,
but loading it into a *subtly* different one (wrong layer sizes that
still happen to load) would fail silently and produce garbage scores.

The idea: train the network to reconstruct NORMAL-looking transactions
well (it only ever sees unlabeled data, but the vast majority of rows --
99.83% -- are legitimate, so "reconstruct the input" and "reconstruct a
normal transaction" are almost the same objective). A transaction the
network reconstructs poorly is one that doesn't fit the pattern it
learned -- that reconstruction error is the anomaly score.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

INPUT_DIM = 30  # len(preprocessing.FEATURE_COLUMNS)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "autoencoder.pt"


class Autoencoder(nn.Module):
    """Small feedforward autoencoder: 30 -> 20 -> 14 -> 20 -> 30.

    Deliberately shallow/narrow -- this is 30 already-compact PCA features,
    not images. A bottleneck of 14 (half the input width) is enough to
    force the network to learn a compressed representation of "normal"
    without so much capacity that it just memorizes every row (which would
    make reconstruction error useless as an anomaly signal).
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 14),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(14, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def save_model(model: Autoencoder, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path = MODEL_PATH) -> Autoencoder:
    if not path.exists():
        raise FileNotFoundError(
            f"Autoencoder weights not found at {path}. Run notebooks/02_train_models.py first."
        )
    model = Autoencoder()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def reconstruction_error(model: Autoencoder, X: torch.Tensor) -> torch.Tensor:
    """Per-row mean squared reconstruction error -- the anomaly score.

    Higher = the network found this row harder to reconstruct = more
    anomalous relative to what it learned as "normal".
    """
    model.eval()
    reconstructed = model(X)
    return torch.mean((reconstructed - X) ** 2, dim=1)
