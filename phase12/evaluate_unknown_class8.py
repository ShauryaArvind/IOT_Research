import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib

import sys
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "risk_engine"
    )
)

from Network_classifier import NetworkRiskClassifier


# ============================================================
# CONFIGURATION
# ============================================================

TEST_X = "phase12/data/X_test_class8.csv"
TEST_Y = "phase12/data/y_test_class8.csv"

MODEL_PATH = "phase12/models/classifier_without_class8.pth"
SCALER_PATH = "phase12/models/scaler_without_class8.joblib"
MAPPING_PATH = "phase12/models/label_mapping.json"

OUTPUT_PATH = "phase12/phase12_unknown_predictions.csv"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("PHASE 12 — UNKNOWN CLASS 8 EVALUATION")
print("=" * 70)

print("\nLoading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

# Determine input/output dimensions from saved weights.
input_dim = checkpoint["network.0.weight"].shape[1]

weight_keys = [
    k for k in checkpoint.keys()
    if k.endswith(".weight")
]

final_weight_key = weight_keys[-1]

num_classes = checkpoint[final_weight_key].shape[0]

print(f"Input features: {input_dim}")
print(f"Model output classes: {num_classes}")

if num_classes != 9:
    raise ValueError(
        f"Expected 9 output classes, found {num_classes}"
    )

model = NetworkRiskClassifier(
    input_dim=input_dim,
    num_classes=num_classes
)

model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully.")


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

with open(MAPPING_PATH, "r") as f:
    mapping = json.load(f)

new_to_original = {
    int(k): int(v)
    for k, v in mapping["new_to_original"].items()
}

print("\nModel label mapping:")

for new_label, original_label in new_to_original.items():

    print(
        f"Model output {new_label} "
        f"→ Original class {original_label}"
    )


# ============================================================
# LOAD UNKNOWN CLASS 8
# ============================================================

print("\nLoading unknown Class 8 test samples...")

X = pd.read_csv(TEST_X)

y = pd.read_csv(TEST_Y)["Label"]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Safety checks
if len(X) != len(y):
    raise ValueError(
        "X and y have different numbers of samples."
    )

if not all(y == 8):
    raise ValueError(
        "Test set contains labels other than Class 8."
    )

print(f"Confirmed: {len(X)} samples are Class 8.")

# Clean numerical values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)

X = X.values.astype(np.float32)


# ============================================================
# SCALE
# ============================================================

X_scaled = scaler.transform(X)

X_tensor = torch.FloatTensor(X_scaled)


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning unknown Class 8 samples through model...")

with torch.no_grad():

    logits = model(X_tensor)

    probabilities = F.softmax(
        logits,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )

    confidences = torch.max(
        probabilities,
        dim=1
    ).values


predictions = predictions.numpy()

confidences = confidences.numpy()

probabilities = probabilities.numpy()


# ============================================================
# MAP PREDICTIONS BACK TO ORIGINAL LABELS
# ============================================================

original_predictions = np.array([
    new_to_original[int(p)]
    for p in predictions
])


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results = pd.DataFrame({
    "sample_id": np.arange(len(X)),
    "true_class": y.values,
    "predicted_model_class": predictions,
    "predicted_original_class": original_predictions,
    "confidence": confidences
})


# Add probability for every known class
for model_class in range(num_classes):

    original_class = new_to_original[model_class]

    results[
        f"prob_class_{original_class}"
    ] = probabilities[:, model_class]


# Save results
results.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 12 RESULTS")
print("=" * 70)

print(
    f"\nTotal unknown Class 8 samples: "
    f"{len(results)}"
)

print("\nPrediction distribution:")

prediction_counts = (
    results["predicted_original_class"]
    .value_counts()
    .sort_index()
)

for predicted_class, count in prediction_counts.items():

    percentage = (
        count / len(results)
    ) * 100

    print(
        f"Class {predicted_class}: "
        f"{count} samples "
        f"({percentage:.2f}%)"
    )


# ============================================================
# CONFIDENCE STATISTICS
# ============================================================

print("\nConfidence statistics:")

print(
    f"Mean confidence:   "
    f"{results['confidence'].mean() * 100:.2f}%"
)

print(
    f"Median confidence: "
    f"{results['confidence'].median() * 100:.2f}%"
)

print(
    f"Minimum confidence:"
    f" {results['confidence'].min() * 100:.2f}%"
)

print(
    f"Maximum confidence:"
    f" {results['confidence'].max() * 100:.2f}%"
)

# High-confidence thresholds
for threshold in [0.70, 0.80, 0.90, 0.95]:

    rate = (
        results["confidence"] >= threshold
    ).mean() * 100

    print(
        f"Confidence >= {threshold * 100:.0f}%: "
        f"{rate:.2f}%"
    )


# ============================================================
# MOST COMMON WRONG LABEL
# ============================================================

most_common = (
    results["predicted_original_class"]
    .value_counts()
    .idxmax()
)

most_common_count = (
    results["predicted_original_class"]
    .value_counts()
    .max()
)

most_common_percentage = (
    most_common_count / len(results)
) * 100


print("\nMost common prediction:")

print(
    f"Unknown Class 8 → "
    f"Class {most_common}"
)

print(
    f"Frequency: "
    f"{most_common_count}/{len(results)} "
    f"({most_common_percentage:.2f}%)"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print(
    "The Phase 12 unknown-attack baseline has been generated."
)

print(
    f"Detailed results saved to:\n"
    f"{OUTPUT_PATH}"
)

print("=" * 70)