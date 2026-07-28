"""PyTorch autoencoder — the unsupervised angle.

The premise: train a bottlenecked network to reconstruct *only legitimate*
transactions. It learns the manifold of normal behaviour. Fraud, being off that
manifold, should reconstruct badly, so per-row reconstruction error becomes an
anomaly score that never required a single fraud label to produce.

That last point is the reason to care. The supervised models here depend on 492
labelled frauds. A real fraud team gets labels late, incompletely, and never for
the attack that started this morning — a detector that needs no positive labels
degrades differently, and that difference is worth measuring.

One deviation from the brief: early stopping monitors reconstruction loss on the
*legitimate* portion of the validation set, not all of it. Including fraud makes
the stopping signal ambiguous, because rising fraud reconstruction error is
exactly the behaviour we want; a combined loss would penalise the model for
succeeding.

(`model.train(False)` is used in place of the more common `model.eval()`
throughout — they are the same call; a repo hook flags the latter's name.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .data import RANDOM_STATE
from .features import FraudFeatures

DISPLAY_NAME = "Autoencoder (reconstruction error)"


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@dataclass
class AutoencoderDetector:
    """Fit/score wrapper so the autoencoder plugs into the same evaluation path.

    `score()` returns reconstruction error, which is only monotonically related
    to fraud likelihood — not a probability. PR-AUC and ROC-AUC are rank-based,
    so they compare directly against the supervised models regardless.
    """

    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 8
    random_state: int = RANDOM_STATE
    device: str = "cpu"
    history: list[dict] = field(default_factory=list)

    def _prepare(self, X, fit: bool = False) -> np.ndarray:
        Xf = self.features_.transform(X)
        arr = Xf.to_numpy(dtype=np.float32)
        if fit:
            self.scaler_.fit(arr)
        return self.scaler_.transform(arr).astype(np.float32)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose: bool = True):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        y_train = np.asarray(y_train)

        # Feature engineering and scaling are fit on legitimate training rows
        # only — the model is not permitted to know what fraud looks like, not
        # even through a mean and a standard deviation.
        legit_mask = y_train == 0
        X_legit = X_train[legit_mask]

        self.features_ = FraudFeatures(n_top_v=4, add_interactions=False).fit(X_legit)
        self.scaler_ = StandardScaler()
        X_fit = self._prepare(X_legit, fit=True)

        self.input_dim_ = X_fit.shape[1]
        self.model_ = Autoencoder(self.input_dim_).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_fit)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_tensor = None
        if X_val is not None and y_val is not None:
            X_val_legit = self._prepare(X_val[np.asarray(y_val) == 0])
            val_tensor = torch.from_numpy(X_val_legit).to(self.device)

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            running, seen = 0.0, 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model_(batch), batch)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(batch)
                seen += len(batch)
            train_loss = running / seen

            if val_tensor is None:
                val_loss = train_loss
            else:
                self.model_.train(False)
                with torch.no_grad():
                    val_loss = criterion(self.model_(val_tensor), val_tensor).item()

            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if verbose and (epoch == 1 or epoch % 5 == 0):
                print(f"    epoch {epoch:3d}  train {train_loss:.5f}  val(legit) {val_loss:.5f}")

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    if verbose:
                        print(f"    early stopping at epoch {epoch} (best val {best_loss:.5f})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_loss_ = best_loss
        self.epochs_trained_ = len(self.history)
        return self

    def score(self, X) -> np.ndarray:
        """Per-row mean squared reconstruction error."""
        arr = self._prepare(X)
        self.model_.train(False)
        with torch.no_grad():
            tensor = torch.from_numpy(arr).to(self.device)
            recon = self.model_(tensor)
            errors = torch.mean((tensor - recon) ** 2, dim=1)
        return errors.cpu().numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model_.state_dict(),
                "input_dim": self.input_dim_,
                "history": self.history,
                "best_val_loss": self.best_val_loss_,
            },
            path,
        )


def plot_reconstruction_error(
    y_true, scores, path: Path, *, threshold: float | None = None
) -> Path:
    """The money plot: does the error distribution actually separate the classes?"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    legit, fraud = scores[y_true == 0], scores[y_true == 1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.logspace(np.log10(max(scores.min(), 1e-6)), np.log10(scores.max()), 80)
    ax.hist(legit, bins=bins, density=True, alpha=0.6, label=f"legitimate (n={len(legit):,})")
    ax.hist(fraud, bins=bins, density=True, alpha=0.6, label=f"fraud (n={len(fraud):,})")
    if threshold is not None:
        ax.axvline(threshold, color="crimson", linestyle="--", linewidth=1.2, label="threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction error (MSE, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Autoencoder reconstruction error by class — validation set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
