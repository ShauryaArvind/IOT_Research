import pandas as pd
import numpy as np


# Load the untouched test set
X_test = pd.read_csv("X_test_isolated.csv")

print("Original test shape:", X_test.shape)


# Use a fixed sample for reproducibility
X_suspicious = X_test.iloc[:5000].copy()


# Convert to numpy
X = X_suspicious.to_numpy(dtype=np.float32)


# Fixed random generator
rng = np.random.default_rng(42)


# Create perturbation based on each feature's standard deviation
feature_std = X.std(axis=0)

# Avoid zero perturbation for constant features
feature_std[feature_std == 0] = 1.0


# Add controlled feature-space perturbation
noise = rng.normal(
    loc=0.0,
    scale=2.0 * feature_std,
    size=X.shape
)


X_perturbed = X + noise


# Save suspicious/perturbed traffic
suspicious_df = pd.DataFrame(
    X_perturbed,
    columns=X_suspicious.columns
)

suspicious_df.to_csv(
    "phase13_suspicious_test.csv",
    index=False
)


print("Suspicious test shape:", suspicious_df.shape)
print("Saved: phase13_suspicious_test.csv")