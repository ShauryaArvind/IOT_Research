"""
Denoising Autoencoder (DAE) for network flow features (Phase 10).

Two ways to use it (both evaluated in evaluate_robustness.py):
  1. Denoiser: x -> DAE(x) -> classifier. The legacy pipeline. On the legacy
     numbers this costs a lot of clean accuracy (37% error at eps=0), so it
     must always be reported next to its clean metrics.
  2. Detector: reject any flow whose reconstruction error exceeds a threshold
     calibrated on clean VALIDATION flows (p95 / p99). The classifier sees the
     unmodified input, so clean accuracy is unchanged and the cost shows up
     as benign false-positive rate instead.

Training uses the shared protocol (data.py): train split for fitting, the
validation split for checkpoint selection and threshold calibration. The
isolated test set is not touched here.

Usage:
    python src/risk_engine/denoising_autoencoder.py --config legacy_balanced --seeds 42
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42
EPOCHS = 25
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NOISE_STD = 0.15          # Gaussian noise standard deviation for training
CORRUPTION_PROB = 0.20    # Probability of feature masking/perturbation


def pprint(*args, **kwargs):
    """Print wrapper forcing immediate stdout flush for real-time logging."""
    print(*args, **kwargs)
    sys.stdout.flush()


class DenoisingAutoencoder(nn.Module):
    """
    Symmetric Deep Denoising Autoencoder for tabular network flow features.

    Architecture (input_dim = number of feature columns, 76 in the committed data):
      Encoder: Input -> Linear(64) -> BatchNorm -> LeakyReLU -> Dropout(0.1)
                     -> Linear(32) -> BatchNorm -> LeakyReLU
                     -> Linear(16) [Bottleneck Latent Space]
      Decoder: Latent (16) -> Linear(32) -> BatchNorm -> LeakyReLU
                           -> Linear(64) -> BatchNorm -> LeakyReLU
                           -> Linear(input_dim) [Reconstructed Features]
    """

    def __init__(self, input_dim=76, latent_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, input_dim),
        )

    def encode(self, x):
        """Map feature vector into low-dimensional latent bottleneck."""
        return self.encoder(x)

    def decode(self, z):
        """Reconstruct feature vector from latent representation."""
        return self.decoder(z)

    def forward(self, x):
        """Pass input through encoder and decoder to reconstruct clean features."""
        return self.decoder(self.encoder(x))

    def sanitize(self, X):
        """Denoise a batch of (scaled) flow feature vectors. Same type in, same type out."""
        is_numpy = isinstance(X, np.ndarray)
        X_tensor = torch.FloatTensor(X) if is_numpy else X
        self.eval()
        with torch.no_grad():
            out = self.forward(X_tensor)
        return out.cpu().numpy() if is_numpy else out

    def reconstruction_error(self, X: torch.Tensor) -> torch.Tensor:
        """Differentiable per-sample MSE (used by the adaptive detector attack)."""
        return torch.mean((X - self.forward(X)) ** 2, dim=1)

    def compute_reconstruction_error(self, X):
        """Per-sample reconstruction MSE as a numpy array, shape (N,)."""
        X_tensor = torch.FloatTensor(X) if isinstance(X, np.ndarray) else X
        self.eval()
        with torch.no_grad():
            return self.reconstruction_error(X_tensor).cpu().numpy()


def inject_training_noise(x_clean, noise_std=NOISE_STD, corruption_prob=CORRUPTION_PROB):
    """
    Composite noise injection combining additive Gaussian noise,
    random uniform jitter, and feature dropout/masking.
    """
    noise = torch.randn_like(x_clean) * noise_std
    uniform_jitter = (torch.rand_like(x_clean) - 0.5) * 2.0 * noise_std
    mask = (torch.rand_like(x_clean) > corruption_prob).float()
    return (x_clean + noise + uniform_jitter) * mask


def dae_run_dir(config, seed):
    from training import run_dir
    return run_dir(config, 'dae', seed)


def train_denoising_autoencoder(config=None, seed=RANDOM_STATE, epochs=EPOCHS, splits=None):
    """Train the DAE on the train split, select on val, calibrate detector thresholds on val."""
    from training import DEFAULT_CONFIG, set_seed, splits_for_config

    config = config or DEFAULT_CONFIG
    set_seed(seed)
    pprint("=" * 75)
    pprint(f"Phase 10: Denoising Autoencoder | config={config} seed={seed}")
    pprint("=" * 75)

    if splits is None:
        splits = splits_for_config(config, seed)
    X_tr = torch.FloatTensor(splits.X_train)
    X_val = torch.FloatTensor(splits.X_val)
    input_dim = X_tr.shape[1]

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(X_tr), batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=4096, shuffle=False)

    model = DenoisingAutoencoder(input_dim=input_dim, latent_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse = nn.MSELoss()

    out_dir = dae_run_dir(config, seed)
    os.makedirs(out_dir, exist_ok=True)
    best_val, best_state = float('inf'), None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (xb,) in train_loader:
            recon = model(inject_training_noise(xb))
            loss = mse(recon, xb) + 0.1 * (1.0 - F.cosine_similarity(recon, xb, dim=1).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                val_loss += mse(model(inject_training_noise(xb)), xb).item() * len(xb)
        val_loss /= len(X_val)
        pprint(f"  Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {train_loss/len(train_loader):.5f} "
               f"| Val MSE (noisy->clean): {val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, out_dir / 'dae.pth')

    clean_err = model.compute_reconstruction_error(splits.X_val)
    meta = {
        'config': config, 'seed': seed, 'input_dim': input_dim, 'best_val_mse': best_val,
        'threshold_p95': float(np.percentile(clean_err, 95)),
        'threshold_p99': float(np.percentile(clean_err, 99)),
        'val_clean_error_mean': float(clean_err.mean()),
        'data': splits.info,
    }
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    pprint(f"  Clean val reconstruction error: mean {meta['val_clean_error_mean']:.4f} | "
           f"p95 {meta['threshold_p95']:.4f} | p99 {meta['threshold_p99']:.4f}")
    pprint(f"  Saved -> {out_dir}")
    return model, meta


def load_dae(config, seed):
    d = dae_run_dir(config, seed)
    with open(d / 'meta.json') as f:
        meta = json.load(f)
    model = DenoisingAutoencoder(input_dim=meta['input_dim'], latent_dim=16)
    model.load_state_dict(torch.load(d / 'dae.pth', map_location='cpu'))
    model.eval()
    return model, meta


def main():
    from training import DEFAULT_CONFIG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--seeds', type=int, nargs='+', default=[RANDOM_STATE])
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    a = ap.parse_args()
    for s in a.seeds:
        train_denoising_autoencoder(a.config, s, a.epochs)


if __name__ == '__main__':
    main()
