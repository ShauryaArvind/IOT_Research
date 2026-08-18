import pandas as pd


# ============================================================
# PHASE 15 — COMBINED WEIGHT + THRESHOLD SWEEP
# ============================================================

UNKNOWN_FILE = "phase14_class8_results.csv"
KNOWN_FILE = "phase13_known_test_results.csv"

OUTPUT_FILE = "phase15_combined_sweep.csv"


# Candidate weights
WEIGHT_CONFIGS = [
    (0.40, 0.30, 0.30),
    (0.30, 0.20, 0.50),
    (0.35, 0.15, 0.50),
    (0.20, 0.20, 0.60),
    (0.25, 0.15, 0.60),
]


# Candidate thresholds
THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.6437,
    0.65,
    0.70,
]


print("=" * 70)
print("PHASE 15 — COMBINED WEIGHT + THRESHOLD SWEEP")
print("=" * 70)


unknown = pd.read_csv(UNKNOWN_FILE)
known = pd.read_csv(KNOWN_FILE)


required = [
    "dae_risk",
    "confidence_risk",
    "isolation_risk"
]

for column in required:

    if column not in unknown.columns:
        raise ValueError(
            f"{column} missing from unknown dataset"
        )

    if column not in known.columns:
        raise ValueError(
            f"{column} missing from known dataset"
        )


results = []


for dae_weight, confidence_weight, isolation_weight in WEIGHT_CONFIGS:

    # --------------------------------------------------------
    # Calculate novelty risk
    # --------------------------------------------------------

    unknown_risk = (
        dae_weight * unknown["dae_risk"]
        +
        confidence_weight * unknown["confidence_risk"]
        +
        isolation_weight * unknown["isolation_risk"]
    )

    known_risk = (
        dae_weight * known["dae_risk"]
        +
        confidence_weight * known["confidence_risk"]
        +
        isolation_weight * known["isolation_risk"]
    )


    for threshold in THRESHOLDS:

        unknown_detection = (
            (unknown_risk >= threshold).mean() * 100
        )

        false_positive = (
            (known_risk >= threshold).mean() * 100
        )

        specificity = 100 - false_positive


        # Balanced accuracy
        balanced_accuracy = (
            unknown_detection + specificity
        ) / 2


        results.append({

            "dae_weight": dae_weight,

            "confidence_weight":
                confidence_weight,

            "isolation_weight":
                isolation_weight,

            "threshold":
                threshold,

            "unknown_detection_percent":
                unknown_detection,

            "known_false_positive_percent":
                false_positive,

            "specificity_percent":
                specificity,

            "balanced_accuracy_percent":
                balanced_accuracy
        })


results = pd.DataFrame(results)


# Sort best balanced accuracy first
results = results.sort_values(
    "balanced_accuracy_percent",
    ascending=False
)


results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 70)
print("TOP 15 CONFIGURATIONS")
print("=" * 70)


print(
    results.head(15).to_string(
        index=False,
        formatters={
            "dae_weight":
                "{:.2f}".format,

            "confidence_weight":
                "{:.2f}".format,

            "isolation_weight":
                "{:.2f}".format,

            "threshold":
                "{:.4f}".format,

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
