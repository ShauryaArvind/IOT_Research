import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)
import joblib

# Import the existing model architecture
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "risk_engine"))

from Network_classifier import NetworkRiskClassifier


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_X = "phase12/data/X_train_unknown_removed.csv"
TRAIN_Y = "phase12/data/y_train_unknown_removed.csv"

MODEL_DIR = "phase12/models"

RANDOM_STATE = 42
TEST_SIZE = 0.20

EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 0.001

MAJORITY_CLASS = 0
MAJORITY_CAP = 40_000

# Original class 9 becomes output class 8 after removing
# original Class 8.
MANUAL_WEIGHT_MULTIPLIERS = {
    8: 0.5
}


# Original labels → new contiguous model labels
ORIGINAL_TO_NEW = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    9: 8,
}

NEW_TO_ORIGINAL = {
    new: original
    for original, new in ORIGINAL_TO_NEW.items()
}


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    print("\nLoading Phase 12 training data...")

    X = pd.read_csv(TRAIN_X)
    y = pd.read_csv(TRAIN_Y)["Label"]

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Safety check:
    if 8 in y.values:
        raise ValueError(
            "ERROR: Class 8 is still present in the training data!"
        )

    print("\nOriginal training classes:")
    print(y.value_counts().sort_index())

    # Convert original labels to contiguous labels 0-8.
    y = y.map(ORIGINAL_TO_NEW)

    if y.isna().any():
        raise ValueError("Found a label that is not in the mapping.")

    # Clean numerical data exactly like the existing pipeline.
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = X.values.astype(np.float32)
    y = y.values.astype(np.int64)

    print("\nNew model labels:")
    print(np.bincount(y))

    return X, y


# ============================================================
# UNDERSAMPLING
# ============================================================

def undersample_majority_class(
    X,
    y,
    majority_class,
    cap,
    random_state
):

    rng = np.random.default_rng(random_state)

    majority_idx = np.where(y == majority_class)[0]
    other_idx = np.where(y != majority_class)[0]

    print(
        f"\nClass {majority_class} before undersampling: "
        f"{len(majority_idx)}"
    )

    if len(majority_idx) > cap:
        majority_idx = rng.choice(
            majority_idx,
            size=cap,
            replace=False
        )

    keep_idx = np.concatenate(
        [majority_idx, other_idx]
    )

    rng.shuffle(keep_idx)

    return X[keep_idx], y[keep_idx]


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(y, num_classes):

    counts = np.bincount(
        y,
        minlength=num_classes
    ).astype(np.float32)

    counts = np.maximum(counts, 1)

    weights = 1.0 / counts

    weights = weights * (
        num_classes / weights.sum()
    )

    return torch.FloatTensor(weights)


# ============================================================
# MAIN TRAINING
# ============================================================

def train():

    print("=" * 70)
    print("PHASE 12 — UNKNOWN ATTACK EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    X, y = load_data()

    NUM_CLASSES = 9

    print(
        f"\nNumber of classes seen during training: "
        f"{NUM_CLASSES}"
    )

    print(
        "Original Class 8 is COMPLETELY absent."
    )

    # --------------------------------------------------------
    # 2. Train/validation split
    # --------------------------------------------------------

    print("\nSplitting known classes into train/validation...")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # --------------------------------------------------------
    # 3. Undersample benign class
    # --------------------------------------------------------

    X_train, y_train = undersample_majority_class(
        X_train,
        y_train,
        MAJORITY_CLASS,
        MAJORITY_CAP,
        RANDOM_STATE
    )

    print("\nPost-undersampling training distribution:")

    for cls, count in enumerate(
        np.bincount(y_train, minlength=NUM_CLASSES)
    ):
        original_class = NEW_TO_ORIGINAL[cls]

        print(
            f"Model class {cls} "
            f"(original {original_class}): "
            f"{count}"
        )

    # --------------------------------------------------------
    # 4. Scale using TRAINING DATA ONLY
    # --------------------------------------------------------

    print("\nFitting StandardScaler on known training data only...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    # IMPORTANT:
    # Validation data is transformed using the same scaler.
    X_val = scaler.transform(X_val)

    # --------------------------------------------------------
    # 5. Class weights
    # --------------------------------------------------------

    class_weights = compute_class_weights(
        y_train,
        NUM_CLASSES
    )

    for cls, multiplier in MANUAL_WEIGHT_MULTIPLIERS.items():

        class_weights[cls] *= multiplier

    print("\nClass weights:")

    for cls, weight in enumerate(class_weights.numpy()):

        original_class = NEW_TO_ORIGINAL[cls]

        print(
            f"Model class {cls} "
            f"(original {original_class}): "
            f"{weight:.4f}"
        )

    # --------------------------------------------------------
    # 6. Convert to tensors
    # --------------------------------------------------------

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    train_dataset = TensorDataset(
        X_train_t,
        y_train_t
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # --------------------------------------------------------
    # 7. Initialize model
    # --------------------------------------------------------

    input_dim = X_train.shape[1]

    print("\nInitializing model:")
    print(f"Input features: {input_dim}")
    print(f"Output classes: {NUM_CLASSES}")

    model = NetworkRiskClassifier(
        input_dim=input_dim,
        num_classes=NUM_CLASSES
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # 8. Training
    # --------------------------------------------------------

    best_f1 = 0.0

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODEL_DIR,
        "classifier_without_class8.pth"
    )

    scaler_path = os.path.join(
        MODEL_DIR,
        "scaler_without_class8.joblib"
    )

    mapping_path = os.path.join(
        MODEL_DIR,
        "label_mapping.json"
    )

    print("\nStarting training...")
    print("-" * 70)

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()

            logits = model(batch_X)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            logits = model(X_val_t)

            predictions = torch.argmax(
                logits,
                dim=1
            ).numpy()

        y_true = y_val_t.numpy()

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_true,
                predictions,
                labels=list(range(NUM_CLASSES)),
                zero_division=0
            )
        )

        macro_f1 = f1.mean()

        average_loss = (
            total_loss / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro-F1: {macro_f1:.4f}"
        )

        # Save best model.
        if macro_f1 > best_f1:

            best_f1 = macro_f1

            torch.save(
                model.state_dict(),
                model_path
            )

    # --------------------------------------------------------
    # 9. Save scaler + label mapping
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        scaler_path
    )

    with open(
        mapping_path,
        "w"
    ) as f:

        json.dump(
            {
                "original_to_new": ORIGINAL_TO_NEW,
                "new_to_original": NEW_TO_ORIGINAL
            },
            f,
            indent=2
        )

    print("\n" + "=" * 70)

    print(
        f"Training complete!"
    )

    print(
        f"Best validation Macro-F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"\nModel saved to:\n"
        f"{model_path}"
    )

    print(
        f"Scaler saved to:\n"
        f"{scaler_path}"
    )

    print(
        f"Label mapping saved to:\n"
        f"{mapping_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    train()