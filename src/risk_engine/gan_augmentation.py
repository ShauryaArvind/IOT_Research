"""
Conditional GAN for Class Imbalance Correction
================================================
Generates synthetic samples of WEAK/minority attack classes to supplement
real training data, as an alternative/addition to undersampling + class
weighting.

Concept borrowed from FLVision's Auxiliary-Classifier GAN generator design
(conditional on class label), reimplemented standalone in PyTorch to match
this project's existing pipeline -- no federated learning or GAN-vs-real
discrimination infrastructure needed for this use case.

How it works:
    - Generator: takes random noise + a target class label -> outputs a
      fake feature vector meant to look like a real sample of that class.
    - Discriminator: takes a feature vector + its claimed class label ->
      predicts whether it's REAL (from your dataset) or FAKE (generator-made).
    - The two train against each other: Generator tries to fool the
      Discriminator, Discriminator tries not to be fooled. Over many rounds,
      the Generator gets good enough to produce realistic synthetic samples.

The trained Generator creates extra rows for minority attack classes. It is
used in two places:
  * data.resample(method='cgan') calls augment_with_cgan() on the training
    portion of the leakage-free split (this is the 'cgan' config of the
    balancing ablation, see ablation.py);
  * running this file directly writes synthetic rows to synthetic_data/ for
    inspection.

Note: no reported result before the protocol fix used cGAN output. The
README's old claim that classes 3, 4, 5, 8, 9 were "augmented via cGAN" was
not true, and classes 4 and 5 are not minority classes (6190 and 5923
isolated-test rows).
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'synthetic_data')

RANDOM_STATE = 42
NOISE_DIM = 32
GAN_EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
MAX_ROWS_PER_CLASS_FOR_GAN_TRAINING = 8000  # cap so GAN training stays fast


class Generator(nn.Module):
    """Takes noise + class label -> outputs a synthetic feature vector."""

    def __init__(self, noise_dim, num_classes, feature_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, feature_dim),
            nn.Tanh(),  # matches StandardScaler-ish range reasonably well
        )

    def forward(self, noise, labels):
        label_input = self.label_embedding(labels)
        x = torch.cat([noise, label_input], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    """Takes a feature vector + claimed class label -> real or fake?"""

    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        label_input = self.label_embedding(labels)
        combined = torch.cat([x, label_input], dim=1)
        return self.model(combined)


def _cap_per_class(X, y, cap, seed):
    """Cap rows per class so GAN training runs in reasonable time."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        ix = np.where(y == c)[0]
        if len(ix) > cap:
            ix = rng.choice(ix, size=cap, replace=False)
        keep.append(ix)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


def train_gan(X_train, y_train, num_classes, epochs=GAN_EPOCHS, seed=RANDOM_STATE, verbose=True):
    """Trains ONE conditional GAN across all classes present in X_train/y_train."""
    torch.manual_seed(seed)
    feature_dim = X_train.shape[1]

    # Scale to roughly [-1, 1] to match the Generator's Tanh output range
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled = np.clip(X_scaled / 4.0, -1.0, 1.0)  # rough squash into [-1,1]

    X_t = torch.FloatTensor(X_scaled)
    y_t = torch.LongTensor(y_train)

    generator = Generator(NOISE_DIM, num_classes, feature_dim)
    discriminator = Discriminator(num_classes, feature_dim)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    if verbose:
        print(f"Training Conditional GAN for {epochs} epochs...")
    for epoch in range(epochs):
        d_loss_total, g_loss_total = 0.0, 0.0

        for real_X, real_y in loader:
            batch_size = real_X.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # ---- Train Discriminator ----
            d_optimizer.zero_grad()

            real_pred = discriminator(real_X, real_y)
            d_real_loss = criterion(real_pred, real_labels)

            noise = torch.randn(batch_size, NOISE_DIM)
            gen_labels = real_y[torch.randperm(batch_size)]  # only classes present in the data
            fake_X = generator(noise, gen_labels)
            fake_pred = discriminator(fake_X.detach(), gen_labels)
            d_fake_loss = criterion(fake_pred, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # ---- Train Generator ----
            g_optimizer.zero_grad()
            fake_pred_for_g = discriminator(fake_X, gen_labels)
            g_loss = criterion(fake_pred_for_g, real_labels)  # wants discriminator to say "real"
            g_loss.backward()
            g_optimizer.step()

            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()

        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | D loss: {d_loss_total/len(loader):.4f} | "
                  f"G loss: {g_loss_total/len(loader):.4f}")

    return generator, scaler


def generate_synthetic_samples(generator, scaler, class_id, n_samples, feature_min, feature_max):
    """Generate n_samples rows for one class, un-squash and un-scale them, then
    clip each feature to the observed min/max of that class's real rows."""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, NOISE_DIM)
        labels = torch.full((n_samples,), class_id, dtype=torch.long)
        synthetic_scaled = generator(noise, labels).numpy()

    synthetic_original = scaler.inverse_transform(synthetic_scaled * 4.0)
    return np.clip(synthetic_original, feature_min, feature_max)


def augment_with_cgan(X, y, need, seed=RANDOM_STATE, epochs=GAN_EPOCHS, num_classes=10):
    """Train a cGAN on (X, y) and return `need[c]` synthetic rows per class c,
    in the same feature space as X. Used by data.resample(method='cgan')."""
    X_gan, y_gan = _cap_per_class(X, y, MAX_ROWS_PER_CLASS_FOR_GAN_TRAINING, seed)
    generator, scaler = train_gan(X_gan, y_gan, num_classes, epochs=epochs, seed=seed)
    X_new, y_new = [], []
    for c, n in need.items():
        if n <= 0:
            continue
        real = X[y == c]
        X_new.append(generate_synthetic_samples(generator, scaler, c, n, real.min(0), real.max(0)))
        y_new.append(np.full(n, c, dtype=np.int64))
    if not X_new:
        return np.empty((0, X.shape[1]), np.float32), np.empty(0, np.int64)
    return np.vstack(X_new).astype(np.float32), np.concatenate(y_new)


def main():
    from data import BENIGN, load_splits, raw_train_available

    print("=" * 70)
    print("Conditional GAN: Synthetic Minority-Class Sample Generation")
    print("=" * 70)
    resampling = 'none' if raw_train_available() else 'legacy_balanced'
    if resampling == 'legacy_balanced':
        print("WARNING: X_train_raw.csv not found; training the cGAN on the SMOTE+Tomek output.")
    splits = load_splits(resampling=resampling, seed=RANDOM_STATE)
    X, y = splits.X_train, splits.y_train
    counts = np.bincount(y, minlength=10)
    target = int(np.delete(counts, BENIGN).max())
    need = {c: target - int(counts[c]) for c in range(10) if c != BENIGN and 0 < counts[c] < target}
    print(f"  Real class counts: {counts.tolist()} -> generating {need}")

    X_new, y_new = augment_with_cgan(X, y, need)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(splits.prep.inverse(X_new), columns=splits.prep.feature_names).to_csv(
        os.path.join(OUTPUT_DIR, 'X_synthetic.csv'), index=False)
    pd.DataFrame(y_new, columns=['Label']).to_csv(os.path.join(OUTPUT_DIR, 'y_synthetic.csv'), index=False)
    print(f"\nSaved {len(y_new)} synthetic rows (raw feature units) to {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
