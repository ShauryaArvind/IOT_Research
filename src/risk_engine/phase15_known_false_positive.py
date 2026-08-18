import pandas as pd


# ============================================================
# PHASE 15 — KNOWN TRAFFIC FALSE-POSITIVE TEST
# ============================================================

KNOWN_RESULTS = "phase13_known_test_results.csv"
OUTPUT_FILE = "phase15_known_false_positive.csv"

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
print("PHASE 15 — KNOWN TRAFFIC FALSE-POSITIVE TEST")
print("=" * 70)

print("\nLoading known test results...")

df = pd.read_csv(KNOWN_RESULTS)

print(f"Loaded {len(df)} known samples.")

if "novelty_risk" not in df.columns:
    raise ValueError("novelty_risk column not found.")


rows = []

for threshold in THRESHOLDS:

    flagged = df["novelty_risk"] >= threshold

    flagged_count = int(flagged.sum())

    false_positive_rate = flagged.mean() * 100

    rows.append({
        "threshold": threshold,
        "known_samples_flagged": flagged_count,
        "known_samples_total": len(df),
        "false_positive_rate_percent": false_positive_rate,
        "known_samples_not_flagged":
            len(df) - flagged_count
    })


results = pd.DataFrame(rows)

results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 70)
print("KNOWN TRAFFIC RESULTS")
print("=" * 70)

print(
    results.to_string(
        index=False,
        formatters={
            "threshold": "{:.4f}".format,
            "false_positive_rate_percent": "{:.4f}".format
        }
    )
)

print("\n" + "=" * 70)
print(f"Results saved to: {OUTPUT_FILE}")
print("=" * 70)