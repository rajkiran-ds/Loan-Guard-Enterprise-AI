"""
src/ml/mlp.py
-------------
PyTorch MLP for loan default risk prediction.

Used by experimentation.py as one of six comparison models.
Production served model stays XGBoost (train_model.py).
The MLP is a comparison candidate and a swappable module.

Keeping the model definition, training loop, and inference wrapper in a
separate file means experimentation.py stays a pure "compare models"
script. If the MLP is ever promoted to production, predict.py can import
from here without touching experiment code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PyTorchMLP(nn.Module):
    """
    Two-hidden-layer perceptron: input(n_features) → 32 → 16 → 1 logit.

    Why these choices:
    - 32 → 16 hidden sizes: small enough to train on CPU in seconds,
      large enough to learn non-linear interactions in ~8 tabular features.
    - ReLU: standard, cheap, no vanishing-gradient issue at this depth.
    - Dropout 0.2: prevents overfitting on a few thousand rows.
    - Output is a raw logit (no sigmoid here) because BCEWithLogitsLoss
      combines sigmoid + BCE in a numerically stable way.
    """

    def __init__(self, n_features: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape [batch]. Apply sigmoid outside for probs."""
        return self.net(x).squeeze(-1)


def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
) -> PyTorchMLP:
    """
    Train PyTorchMLP with early stopping on validation loss.

    Early stopping rationale: without it, the MLP overfits a small tabular
    dataset after ~30 epochs and val loss climbs. We track the best model
    seen and stop after `patience` epochs without improvement.
    """
    n_features = X_train.shape[1]
    model = PyTorchMLP(n_features)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    train_dl = DataLoader(
        TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for _epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_mlp(model: PyTorchMLP, X):
    """Return default-probability array, shape [n]."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def save_mlp(model: PyTorchMLP, path: str = "src/ml/artifacts/mlp.pt") -> None:
    """Persist model weights. Directory must exist."""
    torch.save(model.state_dict(), path)


def load_mlp(n_features: int, path: str = "src/ml/artifacts/mlp.pt") -> PyTorchMLP:
    """Load weights into a fresh MLP. Caller must supply n_features."""
    model = PyTorchMLP(n_features)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model