import pandas as pd
import numpy as np


UNKNOWN_FILE = "phase14_class8_results.csv"
KNOWN_FILE = "phase13_known_test_results.csv"

unknown = pd.read_csv(UNKNOWN_FILE)
known = pd.read_csv(KNOWN_FILE)


print("=" * 70)
print("PHASE 15 — NOVELTY SCORE DISTRIBUTION ANALYSIS")
print("=" * 70)

print(f"\nUnknown Class 8 samples: {len(unknown)}")
print(f"Known samples: {len(known)}")


# ------------------------------------------------------------
# Analyze the three individual signals
# ------------------------------------------------------------

signals = [
    "dae_risk",
    "confidence_risk",
    "isolation_risk"
]


for signal in signals:

    print("\n" + "=" * 70)
    print(signal.upper())
    print("=" * 70)

    u = unknown[signal]
    k = known[signal]

    print("\nUnknown Class 8:")
    print(
        f"  Mean:   {u.mean():.4f}"
    )
    print(
        f"  Median: {u.median():.4f}"
    )
    print(
        f"  90th:   {u.quantile(0.90):.4f}"
    )
    print(
        f"  95th:   {u.quantile(0.95):.4f}"
    )
    print(
        f"  Max:    {u.max():.4f}"
    )

    print("\nKnown traffic:")
    print(
        f"  Mean:   {k.mean():.4f}"
    )
    print(
        f"  Median: {k.median():.4f}"
    )
    print(
        f"  90th:   {k.quantile(0.90):.4f}"
    )
    print(
        f"  95th:   {k.quantile(0.95):.4f}"
    )
    print(
        f"  Max:    {k.max():.4f}"
    )


# ------------------------------------------------------------
# Current novelty-risk calculation
# ------------------------------------------------------------

unknown_risk = (
    0.40 * unknown["dae_risk"]
    + 0.30 * unknown["confidence_risk"]
    + 0.30 * unknown["isolation_risk"]
)

known_risk = (
    0.40 * known["dae_risk"]
    + 0.30 * known["confidence_risk"]
    + 0.30 * known["isolation_risk"]
)


print("\n" + "=" * 70)
print("CURRENT NOVELTY RISK")
print("=" * 70)


for name, values in [
    ("Unknown Class 8", unknown_risk),
    ("Known traffic", known_risk)
]:

    print(f"\n{name}:")

    print(
        f"  Mean:   {values.mean():.4f}"
    )

    print(
        f"  Median: {values.median():.4f}"
    )

    print(
        f"  75th:   {values.quantile(0.75):.4f}"
    )

    print(
        f"  90th:   {values.quantile(0.90):.4f}"
    )

    print(
        f"  95th:   {values.quantile(0.95):.4f}"
    )

    print(
        f"  99th:   {values.quantile(0.99):.4f}"
    )

    print(
        f"  Max:    {values.max():.4f}"
    )


# ------------------------------------------------------------
# Separation at current threshold
# ------------------------------------------------------------

threshold = 0.6437

unknown_flagged = (unknown_risk >= threshold).sum()
known_flagged = (known_risk >= threshold).sum()


print("\n" + "=" * 70)
print("CURRENT THRESHOLD")
print("=" * 70)

print(f"\nThreshold: {threshold}")

print(
    f"Unknown Class 8 flagged: "
    f"{unknown_flagged}/{len(unknown)} "
    f"({unknown_flagged / len(unknown) * 100:.2f}%)"
)

print(
    f"Known traffic flagged: "
    f"{known_flagged}/{len(known)} "
    f"({known_flagged / len(known) * 100:.2f}%)"
)


print("\n" + "=" * 70)