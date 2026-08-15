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

We only train this on your WEAK classes (configurable), then use the
trained Generator to create extra synthetic rows for those classes only,
which get added to the real (undersampled) training set before training
the actual classifier.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
X_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'X_train_final_balanced.csv')
Y_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'y_train_final_balanced.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'synthetic_data')

TEST_SIZE = 0.2
RANDOM_STATE = 42
NOISE_DIM = 32
GAN_EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
MAX_ROWS_PER_CLASS_FOR_GAN_TRAINING = 8000  # cap so GAN training stays fast

# Which classes to generate extra synthetic samples for, and how many each.
# These are the classes that struggled most in the real classifier's results.
WEAK_CLASSES_TO_BOOST = {
    3: 15000,
    4: 15000,
    5: 15000,
    8: 15000,
    9: 15000,
}


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


def load_training_data():
    X = pd.read_csv(X_CSV_PATH)
    y = pd.read_csv(Y_CSV_PATH).iloc[:, 0]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = X.values.astype(np.float32)
    y = y.values.astype(np.int64)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Cap rows per class so GAN training runs in reasonable time -- the GAN
    # only needs enough examples per class to learn its distribution, not
    # every single row.
    rng = np.random.default_rng(RANDOM_STATE)
    keep_idx = []
    for c in np.unique(y_train):
        class_idx = np.where(y_train == c)[0]
        if len(class_idx) > MAX_ROWS_PER_CLASS_FOR_GAN_TRAINING:
            class_idx = rng.choice(class_idx, size=MAX_ROWS_PER_CLASS_FOR_GAN_TRAINING, replace=False)
        keep_idx.append(class_idx)
    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)

    return X_train[keep_idx], y_train[keep_idx]


def train_gan(X_train, y_train, num_classes):
    """Trains ONE conditional GAN across all classes present in X_train/y_train
    (using real data as the 'real' examples for the discriminator to learn
    from), so the Generator learns each class's distinct patterns."""
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

    print(f"Training Conditional GAN for {GAN_EPOCHS} epochs...")
    for epoch in range(GAN_EPOCHS):
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
            gen_labels = torch.randint(0, num_classes, (batch_size,))
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

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}/{GAN_EPOCHS} | D loss: {d_loss_total/len(loader):.4f} | "
                  f"G loss: {g_loss_total/len(loader):.4f}")

    return generator, scaler


def generate_synthetic_samples(generator, scaler, class_id, n_samples, feature_min, feature_max):
    """Generate n_samples synthetic feature rows for a specific class,
    un-scale them back to the original feature space, then clip each
    feature to the real, observed min/max range -- prevents physically
    impossible values (e.g. negative packet counts) that a GAN can
    otherwise produce."""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, NOISE_DIM)
        labels = torch.full((n_samples,), class_id, dtype=torch.long)
        synthetic_scaled = generator(noise, labels).numpy()

    synthetic_unsquashed = synthetic_scaled * 4.0
    synthetic_original = scaler.inverse_transform(synthetic_unsquashed)
    synthetic_original = np.clip(synthetic_original, feature_min, feature_max)
    return synthetic_original


def main():
    print("=" * 70)
    print("Conditional GAN: Synthetic Minority-Class Sample Generation")
    print("=" * 70)

    print("\n[1/3] Loading real training data...")
    X_train, y_train = load_training_data()
    num_classes = int(y_train.max()) + 1
    print(f"  Loaded {X_train.shape[0]} rows, {X_train.shape[1]} features, {num_classes} classes")
    print(f"  Real class counts: {np.bincount(y_train, minlength=num_classes)}")

    print("\n[2/3] Training the Conditional GAN on real data...")
    generator, scaler = train_gan(X_train, y_train, num_classes)

    print("\n[3/3] Generating synthetic samples for weak classes...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    feature_min = X_train.min(axis=0)
    feature_max = X_train.max(axis=0)

    all_synthetic_X = []
    all_synthetic_y = []
    for class_id, n_samples in WEAK_CLASSES_TO_BOOST.items():
        synthetic_X = generate_synthetic_samples(
            generator, scaler, class_id, n_samples, feature_min, feature_max
        )
        all_synthetic_X.append(synthetic_X)
        all_synthetic_y.append(np.full(n_samples, class_id))
        print(f"  Generated {n_samples} synthetic rows for Class {class_id}")

    synthetic_X = np.vstack(all_synthetic_X)
    synthetic_y = np.concatenate(all_synthetic_y)

    # Save as CSVs, ready to be concatenated with your real training data
    feature_cols = pd.read_csv(X_CSV_PATH, nrows=1).columns.tolist()
    pd.DataFrame(synthetic_X, columns=feature_cols).to_csv(
        os.path.join(OUTPUT_DIR, 'X_synthetic.csv'), index=False
    )
    pd.DataFrame(synthetic_y, columns=['Label']).to_csv(
        os.path.join(OUTPUT_DIR, 'y_synthetic.csv'), index=False
    )

    print(f"\nSaved {len(synthetic_y)} synthetic rows to {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()