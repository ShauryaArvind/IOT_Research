"""
Denoising Autoencoder (DAE) for Network Flow Feature Sanitization & Anomaly Detection
======================================================================================
Part of Phase 10: Advanced Adversarial Robustness & Denoising Autoencoders.

Functions as a defense pre-filtering layer placed ahead of the classifier.
When adversarial perturbations (e.g. FGSM, PGD) shift feature vectors away from
the natural manifold of network flows, the DAE reconstructs the perturbed inputs
back onto the clean distribution manifold prior to classifier inference.

Key Capabilities:
  1. Denoising Reconstruction: Removes high-frequency adversarial noise and perturbations.
  2. Latent Projection: Compresses 78 features into a compact 16-dim representation.
  3. Reconstruction Anomaly Detection: Computes per-flow MSE reconstruction error.
     Anomalous perturbation magnitudes can be routed directly to the Zero-Trust Engine.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
X_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'X_train_final_balanced.csv')
Y_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'y_train_final_balanced.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DAE_MODEL_PATH = os.path.join(MODELS_DIR, 'denoising_autoencoder.pth')

RANDOM_STATE = 42
TEST_SIZE = 0.2
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
    Symmetric Deep Denoising Autoencoder for tabular IoT network flow representations.
    
    Architecture:
      Encoder: Input (78) -> Linear(64) -> BatchNorm -> LeakyReLU -> Dropout(0.1)
                          -> Linear(32) -> BatchNorm -> LeakyReLU
                          -> Linear(16) [Bottleneck Latent Space]
      Decoder: Latent (16) -> Linear(32) -> BatchNorm -> LeakyReLU
                           -> Linear(64) -> BatchNorm -> LeakyReLU
                           -> Linear(78) [Reconstructed Clean Features]
    """

    def __init__(self, input_dim=78, latent_dim=16):
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
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def sanitize(self, X):
        """
        Sanitize / denoise a batch of network flow feature vectors.

        Args:
            X: Input feature vectors (numpy array or torch.Tensor)

        Returns:
            Sanitized feature vectors of identical shape and type.
        """
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X

        self.eval()
        with torch.no_grad():
            sanitized_tensor = self.forward(X_tensor)

        if is_numpy:
            return sanitized_tensor.cpu().numpy()
        return sanitized_tensor

    def compute_reconstruction_error(self, X):
        """
        Computes per-sample Mean Squared Error (MSE) reconstruction loss.
        High reconstruction error signals out-of-distribution or adversarial manipulation.

        Args:
            X: Input feature vectors (numpy array or torch.Tensor)

        Returns:
            numpy array of reconstruction errors (shape: (N,))
        """
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X

        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(X_tensor)
            mse_per_sample = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

        return mse_per_sample.cpu().numpy()


def inject_training_noise(x_clean, noise_std=NOISE_STD, corruption_prob=CORRUPTION_PROB):
    """
    Composite noise injection combining additive Gaussian noise,
    random uniform jitter, and feature dropout/masking.
    """
    noise = torch.randn_like(x_clean) * noise_std
    uniform_jitter = (torch.rand_like(x_clean) - 0.5) * 2.0 * noise_std

    # Apply random zero-masking
    mask = (torch.rand_like(x_clean) > corruption_prob).float()

    x_noisy = (x_clean + noise + uniform_jitter) * mask
    return x_noisy


def load_and_prepare_data():
    """Load dataset and perform standardized train/test split."""
    pprint("  Loading raw dataset from CSV...")
    X = pd.read_csv(X_CSV_PATH).replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    y = pd.read_csv(Y_CSV_PATH).iloc[:, 0].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_denoising_autoencoder():
    """Trains the Denoising Autoencoder to reconstruct clean flow features from noisy inputs."""
    pprint("=" * 75)
    pprint("Phase 10: Training Denoising Autoencoder (DAE) Pre-Filtering Sanitizer")
    pprint("=" * 75)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_prepare_data()
    input_dim = X_train_scaled.shape[1]

    pprint(f"  Input Features: {input_dim} | Training Samples: {X_train_scaled.shape[0]} | Test: {X_test_scaled.shape[0]}")

    X_train_t = torch.FloatTensor(X_train_scaled)
    X_test_t = torch.FloatTensor(X_test_scaled)

    train_dataset = TensorDataset(X_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(X_test_t)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DenoisingAutoencoder(input_dim=input_dim, latent_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    mse_criterion = nn.MSELoss()

    os.makedirs(MODELS_DIR, exist_ok=True)
    best_val_loss = float('inf')

    pprint(f"\n  Starting DAE training for {EPOCHS} epochs...")
    pprint("-" * 75)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for (batch_x,) in train_loader:
            batch_x_noisy = inject_training_noise(batch_x, noise_std=NOISE_STD, corruption_prob=CORRUPTION_PROB)

            optimizer.zero_grad()
            batch_recon = model(batch_x_noisy)

            # Combined loss: MSE reconstruction + Cosine direction similarity
            mse_loss = mse_criterion(batch_recon, batch_x)
            cosine_loss = 1.0 - F.cosine_similarity(batch_recon, batch_x, dim=1).mean()
            loss = mse_loss + 0.1 * cosine_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation on test set (with test noise)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in test_loader:
                batch_x_noisy = inject_training_noise(batch_x, noise_std=NOISE_STD, corruption_prob=CORRUPTION_PROB)
                batch_recon = model(batch_x_noisy)
                loss = mse_criterion(batch_recon, batch_x)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)

        pprint(f"  Epoch [{epoch+1:02d}/{EPOCHS:02d}] | Train Loss: {avg_train_loss:.5f} | Val MSE: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), DAE_MODEL_PATH)

    pprint("-" * 75)
    pprint(f"DAE Training Complete! Best Val MSE: {best_val_loss:.5f}")
    pprint(f"Checkpoint saved to: {DAE_MODEL_PATH}")

    # Compute baseline reconstruction error on clean test flows for anomaly thresholding
    model.load_state_dict(torch.load(DAE_MODEL_PATH))
    clean_errors = model.compute_reconstruction_error(X_test_scaled)
    p95_error = float(np.percentile(clean_errors, 95))
    p99_error = float(np.percentile(clean_errors, 99))
    pprint(f"Clean Reconstruction Error Profile -> Mean: {clean_errors.mean():.4f} | 95th-pct: {p95_error:.4f} | 99th-pct: {p99_error:.4f}")

    return model, scaler


if __name__ == '__main__':
    train_denoising_autoencoder()
