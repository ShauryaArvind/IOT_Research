import pandas as pd
import numpy as np

from src.risk_engine.novelty_detector import DAENoveltyDetector


print("Loading data...")

X_train = pd.read_csv("X_train_final_balanced.csv")
X_test = pd.read_csv("X_test_isolated.csv")
y_test = pd.read_csv("y_test_isolated.csv")

print("Training data:", X_train.shape)
print("Test data:", X_test.shape)
print("Test labels:", y_test.shape)


print("\nInitializing Phase 13 detector...")

detector = DAENoveltyDetector()


print("\nTraining Isolation Forest...")

detector.fit_isolation_forest(
    X_train,
    sample_size=100000
)

print("Isolation Forest ready!")


# Calibrate ONLY using training data
print("\nCalibrating thresholds...")

calibration_sample = X_train.iloc[:5000]

thresholds = detector.calibrate_thresholds(
    calibration_sample
)

print("\nThresholds:")

for name, value in thresholds.items():
    print(f"{name}: {value}")


# Evaluate untouched test set in batches
print("\nEvaluating untouched test set...")

batch_size = 5000

all_risks = []
all_dae = []
all_confidence = []
all_isolation = []

for start in range(0, len(X_test), batch_size):

    end = min(
        start + batch_size,
        len(X_test)
    )

    X_batch = X_test.iloc[start:end]

    results = detector.calculate_novelty_risk(
        X_batch
    )

    all_risks.extend(
        results["novelty_risk"]
    )

    all_dae.extend(
        results["dae_risk"]
    )

    all_confidence.extend(
        results["confidence_risk"]
    )

    all_isolation.extend(
        results["isolation_risk"]
    )

    print(
        f"Processed {end}/{len(X_test)}"
    )


all_risks = np.array(all_risks)


print("\n===== PHASE 13 KNOWN TEST RESULTS =====")

print("Number of test samples:", len(all_risks))

print("Minimum novelty risk:", all_risks.min())
print("Maximum novelty risk:", all_risks.max())
print("Mean novelty risk:", all_risks.mean())
print("Median novelty risk:", np.median(all_risks))

print("\nPercentiles:")

for p in [50, 75, 90, 95, 99]:

    print(
        f"{p}th percentile:",
        np.percentile(all_risks, p)
    )


# Save results
results_df = pd.DataFrame({
    "novelty_risk": all_risks,
    "dae_risk": all_dae,
    "confidence_risk": all_confidence,
    "isolation_risk": all_isolation
})

results_df.to_csv(
    "phase13_known_test_results.csv",
    index=False
)

print(
    "\nSaved: phase13_known_test_results.csv"
)