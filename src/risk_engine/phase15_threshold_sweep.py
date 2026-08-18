import pandas as pd
import numpy as np


# ============================================================
# PHASE 15 — THRESHOLD SWEEP
# ============================================================

PHASE14_RESULTS = "phase14_class8_results.csv"

OUTPUT_FILE = "phase15_threshold_sweep.csv"

# Thresholds we want to test.
THRESHOLDS = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.6437319422245168,
    0.65,
    0.70,
]


print("=" * 70)
print("PHASE 15 — NOVELTY THRESHOLD SWEEP")
print("=" * 70)


# ============================================================
# LOAD PHASE 14 RESULTS
# ============================================================

print("\nLoading Phase 14 results...")

df = pd.read_csv(PHASE14_RESULTS)

print(
    f"Loaded {len(df)} samples."
)

if "novelty_risk" not in df.columns:
    raise ValueError(
        "novelty_risk column not found."
    )


# ============================================================
# TEST EACH THRESHOLD
# ============================================================

rows = []

for threshold in THRESHOLDS:

    flagged = (
        df["novelty_risk"] >= threshold
    )

    flagged_count = int(
        flagged.sum()
    )

    detection_rate = (
        flagged.mean() * 100
    )

    rows.append({
        "threshold": threshold,
        "flagged_samples": flagged_count,
        "total_samples": len(df),
        "detection_rate_percent": detection_rate,
        "missed_samples": len(df) - flagged_count,
        "miss_rate_percent":
            100 - detection_rate
    })


results = pd.DataFrame(rows)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

print(
    results.to_string(
        index=False,
        formatters={
            "threshold": "{:.4f}".format,
            "detection_rate_percent": "{:.2f}".format,
            "miss_rate_percent": "{:.2f}".format
        }
    )
)


print("\n" + "=" * 70)

print(
    f"Results saved to: "
    f"{OUTPUT_FILE}"
)

print("=" * 70)