import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.risk_engine.novelty_detector import DAENoveltyDetector


print("Loading datasets...")

X_train = pd.read_csv("X_train_final_balanced.csv")

# Use 5,000 known test samples for comparison
X_known = pd.read_csv(
    "X_test_isolated.csv"
).iloc[:5000]

# Our synthetic suspicious traffic
X_suspicious = pd.read_csv(
    "phase13_suspicious_test.csv"
)

print("Training data:", X_train.shape)
print("Known test sample:", X_known.shape)
print("Suspicious test:", X_suspicious.shape)


# --------------------------------------------------
# Initialize detector
# --------------------------------------------------

print("\nInitializing Phase 13 detector...")

detector = DAENoveltyDetector()


# --------------------------------------------------
# Train Isolation Forest
# --------------------------------------------------

print("\nTraining Isolation Forest...")

detector.fit_isolation_forest(
    X_train,
    sample_size=100000
)

print("Isolation Forest ready!")


# --------------------------------------------------
# Calibrate using known TRAINING data
# --------------------------------------------------

print("\nCalibrating thresholds...")

calibration_sample = X_train.iloc[:5000]

thresholds = detector.calibrate_thresholds(
    calibration_sample
)

print("\nThresholds:")

for name, value in thresholds.items():
    print(f"{name}: {value}")


# --------------------------------------------------
# Evaluate known traffic
# --------------------------------------------------

print("\nEvaluating known traffic...")

known_results = detector.calculate_novelty_risk(
    X_known
)

known_risk = known_results["novelty_risk"]


# --------------------------------------------------
# Evaluate suspicious traffic
# --------------------------------------------------

print("\nEvaluating suspicious traffic...")

suspicious_results = detector.calculate_novelty_risk(
    X_suspicious
)

suspicious_risk = suspicious_results["novelty_risk"]


# --------------------------------------------------
# Print distributions
# --------------------------------------------------

print("\n======================================")
print("KNOWN TRAFFIC")
print("======================================")

print("Mean:", known_risk.mean())
print("Median:", np.median(known_risk))
print("90th percentile:", np.percentile(known_risk, 90))
print("95th percentile:", np.percentile(known_risk, 95))
print("99th percentile:", np.percentile(known_risk, 99))


print("\n======================================")
print("SUSPICIOUS TRAFFIC")
print("======================================")

print("Mean:", suspicious_risk.mean())
print("Median:", np.median(suspicious_risk))
print("90th percentile:", np.percentile(suspicious_risk, 90))
print("95th percentile:", np.percentile(suspicious_risk, 95))
print("99th percentile:", np.percentile(suspicious_risk, 99))


# --------------------------------------------------
# Create labels
# 0 = known
# 1 = suspicious
# --------------------------------------------------

y_true = np.concatenate([
    np.zeros(len(known_risk)),
    np.ones(len(suspicious_risk))
])

y_scores = np.concatenate([
    known_risk,
    suspicious_risk
])


# --------------------------------------------------
# ROC-AUC
# --------------------------------------------------

auc = roc_auc_score(
    y_true,
    y_scores
)

print("\n======================================")
print("ROC-AUC")
print("======================================")

print("ROC-AUC:", auc)


# --------------------------------------------------
# Select threshold from known traffic
# --------------------------------------------------

threshold = np.percentile(
    known_risk,
    95
)

print("\n======================================")
print("SELECTED NOVELTY THRESHOLD")
print("======================================")

print("Threshold:", threshold)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = (
    y_scores >= threshold
).astype(int)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

print(cm)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Known",
            "Suspicious"
        ]
    )
)


# --------------------------------------------------
# Detection percentages
# --------------------------------------------------

known_flagged = np.mean(
    known_risk >= threshold
) * 100

suspicious_detected = np.mean(
    suspicious_risk >= threshold
) * 100

print("\n======================================")
print("DETECTION SUMMARY")
print("======================================")

print(
    f"Known traffic incorrectly flagged: "
    f"{known_flagged:.2f}%"
)

print(
    f"Suspicious traffic detected: "
    f"{suspicious_detected:.2f}%"
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

comparison = pd.DataFrame({
    "novelty_risk": y_scores,
    "actual": y_true,
    "predicted_suspicious": y_pred
})

comparison.to_csv(
    "phase13_known_vs_suspicious.csv",
    index=False
)

print(
    "\nSaved: phase13_known_vs_suspicious.csv"
)