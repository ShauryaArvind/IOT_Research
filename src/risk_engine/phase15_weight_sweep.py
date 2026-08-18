import pandas as pd
import itertools


# ============================================================
# PHASE 15 — NOVELTY RISK WEIGHT SWEEP
# ============================================================

UNKNOWN_FILE = "phase14_class8_results.csv"
KNOWN_FILE = "phase13_known_test_results.csv"

OUTPUT_FILE = "phase15_weight_sweep.csv"

# Keep the current threshold for a fair comparison.
THRESHOLD = 0.6437319422245168


# Candidate weights.
# Each tuple is:
# (DAE weight, confidence weight, isolation weight)

WEIGHT_CONFIGS = [
    (0.40, 0.30, 0.30),
    (0.30, 0.20, 0.50),
    (0.25, 0.15, 0.60),
    (0.20, 0.10, 0.70),
    (0.20, 0.20, 0.60),
    (0.30, 0.10, 0.60),
    (0.35, 0.15, 0.50),
]


print("=" * 70)
print("PHASE 15 — NOVELTY RISK WEIGHT SWEEP")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading unknown Class 8 results...")

unknown = pd.read_csv(UNKNOWN_FILE)

print(
    f"Unknown samples: {len(unknown)}"
)


print("\nLoading known test results...")

known = pd.read_csv(KNOWN_FILE)

print(
    f"Known samples: {len(known)}"
)


required_columns = [
    "dae_risk",
    "confidence_risk",
    "isolation_risk"
]

for column in required_columns:

    if column not in unknown.columns:
        raise ValueError(
            f"{column} missing from unknown dataset"
        )

    if column not in known.columns:
        raise ValueError(
            f"{column} missing from known dataset"
        )


# ============================================================
# TEST WEIGHT CONFIGURATIONS
# ============================================================

results = []


for dae_weight, confidence_weight, isolation_weight in WEIGHT_CONFIGS:

    # Calculate new novelty risk for UNKNOWN Class 8.
    unknown_risk = (
        dae_weight * unknown["dae_risk"]
        +
        confidence_weight * unknown["confidence_risk"]
        +
        isolation_weight * unknown["isolation_risk"]
    )

    # Calculate new novelty risk for KNOWN traffic.
    known_risk = (
        dae_weight * known["dae_risk"]
        +
        confidence_weight * known["confidence_risk"]
        +
        isolation_weight * known["isolation_risk"]
    )


    # Unknown attack detection.
    unknown_flagged = (
        unknown_risk >= THRESHOLD
    )

    unknown_detection_rate = (
        unknown_flagged.mean() * 100
    )


    # Known traffic false positives.
    known_flagged = (
        known_risk >= THRESHOLD
    )

    false_positive_rate = (
        known_flagged.mean() * 100
    )


    # Specificity.
    specificity = (
        100 - false_positive_rate
    )


    # Balanced accuracy.
    balanced_accuracy = (
        unknown_detection_rate
        +
        specificity
    ) / 2


    results.append({

        "dae_weight": dae_weight,

        "confidence_weight":
            confidence_weight,

        "isolation_weight":
            isolation_weight,

        "unknown_detection_percent":
            unknown_detection_rate,

        "known_false_positive_percent":
            false_positive_rate,

        "specificity_percent":
            specificity,

        "balanced_accuracy_percent":
            balanced_accuracy
    })


results = pd.DataFrame(results)


# ============================================================
# SORT BY BALANCED ACCURACY
# ============================================================

results = results.sort_values(
    "balanced_accuracy_percent",
    ascending=False
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT
# ============================================================

print("\n")
print("=" * 70)
print("WEIGHT SWEEP RESULTS")
print("=" * 70)

print(
    results.to_string(
        index=False,
        formatters={
            "dae_weight":
                "{:.2f}".format,

            "confidence_weight":
                "{:.2f}".format,

            "isolation_weight":
                "{:.2f}".format,

            "unknown_detection_percent":
                "{:.2f}".format,

            "known_false_positive_percent":
                "{:.2f}".format,

            "specificity_percent":
                "{:.2f}".format,

            "balanced_accuracy_percent":
                "{:.2f}".format
        }
    )
)


print("\n" + "=" * 70)

print(
    f"Results saved to: {OUTPUT_FILE}"
)

print("=" * 70)