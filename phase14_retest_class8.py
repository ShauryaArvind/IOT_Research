import numpy as np
import pandas as pd

from src.risk_engine.novelty_detector import DAENoveltyDetector
from src.risk_engine.zero_trust_engine import ZeroTrustEngine


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_X = "X_train_final_balanced.csv"

UNKNOWN_X = "phase12/data/X_test_class8.csv"
UNKNOWN_Y = "phase12/data/y_test_class8.csv"

OUTPUT_FILE = "phase14_class8_results.csv"

CALIBRATION_SIZE = 5000
ISOLATION_SAMPLE_SIZE = 100000

BATCH_SIZE = 500

# Exact threshold used by Phase 13 Zero-Trust engine.
NOVELTY_THRESHOLD = ZeroTrustEngine.NOVELTY_RISK_THRESHOLD


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PHASE 14 — RETEST HELD-OUT CLASS 8")
print("=" * 70)


# ============================================================
# 1. LOAD KNOWN TRAINING DATA
# ============================================================

print("\nLoading known training data...")

X_train = pd.read_csv(TRAIN_X)

print("Training shape:", X_train.shape)


# ============================================================
# 2. LOAD HELD-OUT CLASS 8
# ============================================================

print("\nLoading held-out Class 8 data...")

X_unknown = pd.read_csv(UNKNOWN_X)
y_unknown = pd.read_csv(UNKNOWN_Y)["Label"]

print("Unknown X shape:", X_unknown.shape)
print("Unknown y shape:", y_unknown.shape)


# Safety checks
if len(X_unknown) != len(y_unknown):
    raise ValueError(
        "X and y have different numbers of samples."
    )

if not all(y_unknown == 8):
    raise ValueError(
        "Held-out test data contains labels other than Class 8."
    )

print(
    f"Confirmed: {len(X_unknown)} samples are "
    "genuinely held-out Class 8."
)


# ============================================================
# 3. INITIALIZE PHASE 13 DETECTOR
# ============================================================

print("\nInitializing Phase 13 novelty detector...")

detector = DAENoveltyDetector()

print("Detector initialized.")


# ============================================================
# 4. TRAIN ISOLATION FOREST ON KNOWN DATA
# ============================================================

print("\nTraining Isolation Forest on known traffic...")

detector.fit_isolation_forest(
    X_train,
    sample_size=ISOLATION_SAMPLE_SIZE
)

print("Isolation Forest ready.")


# ============================================================
# 5. CALIBRATE THRESHOLDS USING KNOWN TRAFFIC ONLY
# ============================================================

print(
    f"\nCalibrating thresholds using "
    f"{CALIBRATION_SIZE} known samples..."
)

X_known = X_train.iloc[:CALIBRATION_SIZE]

thresholds = detector.calibrate_thresholds(
    X_known
)

print("\nPhase 13 learned thresholds:")

for name, value in thresholds.items():
    print(f"{name}: {value}")


# ============================================================
# 6. RUN CLASS 8 THROUGH PHASE 13
# ============================================================

print("\nRunning Class 8 through Phase 13...")

all_dae = []
all_confidence = []
all_isolation = []
all_novelty = []

for start in range(
    0,
    len(X_unknown),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(X_unknown)
    )

    X_batch = X_unknown.iloc[start:end]

    results = detector.calculate_novelty_risk(
        X_batch
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

    all_novelty.extend(
        results["novelty_risk"]
    )

    print(
        f"Processed {end}/{len(X_unknown)}"
    )


# Convert to numpy arrays
all_dae = np.asarray(all_dae)
all_confidence = np.asarray(all_confidence)
all_isolation = np.asarray(all_isolation)
all_novelty = np.asarray(all_novelty)


# ============================================================
# 7. DETERMINE SUSPICIOUS / NOT SUSPICIOUS
# ============================================================

flagged = (
    all_novelty > NOVELTY_THRESHOLD
)

not_flagged = ~flagged


# ============================================================
# 8. CREATE RESULT TABLE
# ============================================================

results_df = pd.DataFrame({

    "sample_id":
        np.arange(len(X_unknown)),

    "true_class":
        y_unknown.values,

    "dae_risk":
        all_dae,

    "confidence_risk":
        all_confidence,

    "isolation_risk":
        all_isolation,

    "novelty_risk":
        all_novelty,

    "novelty_threshold":
        NOVELTY_THRESHOLD,

    "flagged_suspicious":
        flagged
})


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 9. RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 14 RESULTS")
print("=" * 70)

total = len(results_df)

num_flagged = int(
    flagged.sum()
)

num_not_flagged = int(
    not_flagged.sum()
)

detection_rate = (
    num_flagged / total
) * 100

miss_rate = (
    num_not_flagged / total
) * 100


print(
    f"\nTotal unknown Class 8 samples: "
    f"{total}"
)

print(
    f"Flagged as suspicious: "
    f"{num_flagged}"
)

print(
    f"Not flagged: "
    f"{num_not_flagged}"
)

print(
    f"\nUnknown attack detection rate: "
    f"{detection_rate:.2f}%"
)

print(
    f"Unknown attack miss rate: "
    f"{miss_rate:.2f}%"
)


# ============================================================
# 10. NOVELTY RISK STATISTICS
# ============================================================

print("\nNovelty risk statistics:")

print(
    f"Mean:   "
    f"{all_novelty.mean():.4f}"
)

print(
    f"Median: "
    f"{np.median(all_novelty):.4f}"
)

print(
    f"Min:    "
    f"{all_novelty.min():.4f}"
)

print(
    f"Max:    "
    f"{all_novelty.max():.4f}"
)


# ============================================================
# 11. INDIVIDUAL SIGNAL STATISTICS
# ============================================================

print("\nMean component risks:")

print(
    f"DAE risk:         "
    f"{all_dae.mean():.4f}"
)

print(
    f"Confidence risk:  "
    f"{all_confidence.mean():.4f}"
)

print(
    f"Isolation risk:   "
    f"{all_isolation.mean():.4f}"
)


# ============================================================
# 12. PERCENTAGES ABOVE VARIOUS RISK LEVELS
# ============================================================

print("\nNovelty-risk distribution:")

for threshold in [
    0.25,
    0.40,
    0.50,
    0.60,
    0.6437319422245168,
    0.70,
    0.80,
    0.90
]:

    percentage = (
        (all_novelty >= threshold).mean()
        * 100
    )

    print(
        f"Risk >= {threshold:.4f}: "
        f"{percentage:.2f}%"
    )


# ============================================================
# 13. FINAL
# ============================================================

print("\n" + "=" * 70)

print(
    "Phase 14 Class 8 retest complete."
)

print(
    f"Results saved to: "
    f"{OUTPUT_FILE}"
)

print("=" * 70)