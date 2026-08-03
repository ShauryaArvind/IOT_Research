"""
Train Multiclass Network Risk Classifier
Trains an intrusion detection model on your balanced CICIDS-style dataset.

Adapted from the original repo's binary train_baseline.py to:
  1. Load your own CSVs (X_train_final_balanced.csv / y_train_final_balanced.csv)
     instead of downloading CICIDS-2017 fresh.
  2. Handle MULTICLASS labels (0 = benign, 1-9 = attack types) instead of
     binary malicious/benign.
  3. Undersample the majority class (Class 0) so it doesn't dominate training.
  4. Apply class-weighted CrossEntropyLoss on top of the undersampling, to
     handle whatever imbalance remains among the attack classes.
  5. Report per-class precision/recall/F1 and a confusion matrix each epoch,
     since overall accuracy is misleading on imbalanced data.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)
from Network_classifier import NetworkRiskClassifier

# ---------------------------------------------------------------------------
# CONFIG — edit these paths/values for your setup
# ---------------------------------------------------------------------------
# Your CSVs live two folders up from this script (in the main project folder),
# not next to it — adjust if you move things around.
X_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'X_train_final_balanced.csv')
Y_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'y_train_final_balanced.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# If you've already narrowed down to your final 9 features, list their exact
# column names here. Leave as None to use every column in the X csv as-is.
FEATURE_COLUMNS = None  # e.g. ["Flow Duration", "Total Fwd Packet", ...]

MAJORITY_CLASS = 0          # the class to undersample (Class 0 = benign)
MAJORITY_CAP = 40_000       # cap Class 0 down to this many rows before training
TEST_SIZE = 0.2             # held-out test split, taken BEFORE undersampling
RANDOM_STATE = 42
EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# Manual per-class weight multipliers, applied AFTER the automatic
# inverse-frequency weights below. Use this to hand-correct classes that
# are over- or under-predicted (e.g. Class 9 was being over-guessed, so we
# dial its weight down rather than up).
MANUAL_WEIGHT_MULTIPLIERS = {
    9: 0.5,   # Class 9 was over-predicted (low precision) -> reduce its pull
}


def load_data():
    X = pd.read_csv(X_CSV_PATH)
    y = pd.read_csv(Y_CSV_PATH).iloc[:, 0]  # first column, whatever it's named

    if FEATURE_COLUMNS is not None:
        X = X[FEATURE_COLUMNS]

    # Clean inf/NaN the same way the repo's cicids_loader does
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X.values.astype(np.float32), y.values.astype(np.int64)


def undersample_majority_class(X, y, majority_class, cap, random_state):
    """Randomly drop rows of `majority_class` down to `cap` rows. Leaves all
    other classes untouched."""
    rng = np.random.default_rng(random_state)

    majority_idx = np.where(y == majority_class)[0]
    other_idx = np.where(y != majority_class)[0]

    if len(majority_idx) > cap:
        majority_idx = rng.choice(majority_idx, size=cap, replace=False)

    keep_idx = np.concatenate([majority_idx, other_idx])
    rng.shuffle(keep_idx)

    return X[keep_idx], y[keep_idx]


def compute_class_weights(y, num_classes):
    """Inverse-frequency class weights for CrossEntropyLoss, normalized so
    weights average to 1 (keeps loss magnitude comparable across runs)."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1)  # avoid divide-by-zero for absent classes
    weights = 1.0 / counts
    weights = weights * (num_classes / weights.sum())
    return torch.FloatTensor(weights)


def train_network_classifier():
    print("=" * 70)
    print("Training Multiclass Network Risk Classifier")
    print("=" * 70)

    print("\n[1/6] Loading data...")
    X, y = load_data()
    num_classes = int(y.max()) + 1
    print(f"  Loaded X: {X.shape}  |  y: {y.shape}  |  classes: {num_classes}")
    print(f"  Raw class counts: {np.bincount(y)}")

    print("\n[2/6] Splitting train/test (stratified, before undersampling)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    print(f"\n[3/6] Undersampling Class {MAJORITY_CLASS} in TRAIN only "
          f"(cap={MAJORITY_CAP})...")
    X_train, y_train = undersample_majority_class(
        X_train, y_train, MAJORITY_CLASS, MAJORITY_CAP, RANDOM_STATE
    )
    print(f"  Post-undersample train class counts: {np.bincount(y_train, minlength=num_classes)}")
    print(f"  Test class counts (untouched):        {np.bincount(y_test, minlength=num_classes)}")

    print("\n[4/6] Scaling features (fit on train only)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    class_weights = compute_class_weights(y_train, num_classes)
    for cls, multiplier in MANUAL_WEIGHT_MULTIPLIERS.items():
        class_weights[cls] *= multiplier
    print(f"  Class weights (for residual imbalance): {class_weights.numpy().round(3)}")

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)          # class indices, not one-hot
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n[5/6] Initializing model (input_dim={X_train.shape[1]}, "
          f"num_classes={num_classes})...")
    model = NetworkRiskClassifier(input_dim=X_train.shape[1], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n[6/6] Training for {EPOCHS} epochs...")
    print("-" * 70)

    best_f1_macro = 0.0
    model_path = os.path.join(MODELS_DIR, 'network_risk_classifier_multiclass.pth')
    os.makedirs(MODELS_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)          # raw logits (N, num_classes)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ---- Evaluation ----
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_preds = torch.argmax(test_logits, dim=1).numpy()

        y_true = y_test_t.numpy()
        acc = accuracy_score(y_true, test_preds)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, test_preds, labels=list(range(num_classes)), zero_division=0
        )
        f1_macro = f1.mean()

        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
            f"Acc: {acc:.4f} | Macro-F1: {f1_macro:.4f}"
        )

        if (epoch + 1) % 5 == 0:
            print(f"    per-class F1: {f1.round(3)}")

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            torch.save(model.state_dict(), model_path)

    print("-" * 70)
    print(f"\nTraining complete! Best macro-F1: {best_f1_macro:.4f}")
    print(f"Model saved to: {model_path}")

    # ---- Final detailed report on best-performing state ----
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        final_preds = torch.argmax(model(X_test_t), dim=1).numpy()

    y_true = y_test_t.numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, final_preds, labels=list(range(num_classes)), zero_division=0
    )

    print("\nPer-class results (best checkpoint):")
    print(f"{'Class':>6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for c in range(num_classes):
        print(f"{c:>6} {precision[c]:>10.4f} {recall[c]:>10.4f} {f1[c]:>10.4f} {support[c]:>10}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y_true, final_preds, labels=list(range(num_classes)))
    print(cm)

    print("=" * 70)
    return model


if __name__ == "__main__":
    train_network_classifier()