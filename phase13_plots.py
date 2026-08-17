import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


# Load comparison results
df = pd.read_csv(
    "phase13_known_vs_suspicious.csv"
)

known = df[df["actual"] == 0]["novelty_risk"]
suspicious = df[df["actual"] == 1]["novelty_risk"]

threshold = 0.6437319422245168


# --------------------------------------------------
# 1. Novelty risk distribution
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    known,
    bins=50,
    alpha=0.6,
    label="Known Traffic"
)

plt.hist(
    suspicious,
    bins=50,
    alpha=0.6,
    label="Suspicious Traffic"
)

plt.axvline(
    threshold,
    linestyle="--",
    label=f"Threshold = {threshold:.3f}"
)

plt.xlabel("Novelty Risk")
plt.ylabel("Number of Samples")
plt.title("Phase 13: Known vs Suspicious Novelty Risk")
plt.legend()
plt.tight_layout()

plt.savefig(
    "phase13_novelty_distribution.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 2. ROC Curve
# --------------------------------------------------

y_true = df["actual"]
y_score = df["novelty_risk"]

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_score
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Phase 13 ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    "phase13_roc_curve.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 3. Individual component risks
# --------------------------------------------------

components = [
    "dae_risk",
    "confidence_risk",
    "isolation_risk"
]

for component in components:

    plt.figure(figsize=(9, 5))

    plt.hist(
        df[df["actual"] == 0][component],
        bins=50,
        alpha=0.6,
        label="Known"
    )

    plt.hist(
        df[df["actual"] == 1][component],
        bins=50,
        alpha=0.6,
        label="Suspicious"
    )

    plt.xlabel(component)
    plt.ylabel("Number of Samples")
    plt.title(
        f"Phase 13: {component} Distribution"
    )

    plt.legend()
    plt.tight_layout()

    filename = f"phase13_{component}.png"

    plt.savefig(
        filename,
        dpi=300
    )

    plt.show()


print("\nPlots generated successfully.")