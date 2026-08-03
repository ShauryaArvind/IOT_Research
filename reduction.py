"""
UMAP dimensionality reduction on X_train_final_balanced.csv / y_train_final_balanced.csv
Dataset: network flow features (78 cols) + multiclass 'Label' (10 classes, 0 = benign)
"""

import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
X = pd.read_csv("X_train_final_balanced.csv")
y = pd.read_csv("y_train_final_balanced.csv")["Label"]

print("X shape:", X.shape)
print("y distribution:\n", y.value_counts())

# ---------------------------------------------------------
# 2. Clean data
#    Flow-based features (e.g. Bytes/s) commonly contain inf/-inf
#    from divide-by-zero (Flow Duration = 0). Replace with NaN,
#    then impute, since UMAP cannot handle inf/NaN.
# ---------------------------------------------------------
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

# Drop any constant columns (zero variance) - they add no signal
constant_cols = X.columns[X.nunique() <= 1]
if len(constant_cols) > 0:
    print("Dropping constant columns:", list(constant_cols))
    X = X.drop(columns=constant_cols)

# ---------------------------------------------------------
# 3. Scale features
#    UMAP (like most distance-based methods) needs standardized
#    features - flow duration is in microseconds while flags are 0/1.
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 4. (Optional) subsample for speed
#    488k rows x 78 dims is workable but slow (~10-30 min).
#    Set SAMPLE_SIZE = None to run on the full dataset.
# ---------------------------------------------------------
SAMPLE_SIZE = None # set to None for full data
if SAMPLE_SIZE and SAMPLE_SIZE < len(X_scaled):
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_scaled), SAMPLE_SIZE, replace=False)
    X_fit = X_scaled[idx]
    y_fit = y.iloc[idx].reset_index(drop=True)
else:
    X_fit = X_scaled
    y_fit = y

# ---------------------------------------------------------
# 5. Run UMAP
# ---------------------------------------------------------
reducer = umap.UMAP(
    n_neighbors=15,      # local vs global structure balance
    min_dist=0.1,        # how tightly points are packed
    n_components=2,      # 2D for visualization
    metric="euclidean",
    random_state=42,
    verbose=True,
)
embedding = reducer.fit_transform(X_fit)

# ---------------------------------------------------------
# 6. Save embedding
# ---------------------------------------------------------
embedding_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
embedding_df["Label"] = y_fit.values
embedding_df.to_csv("umap_embedding.csv", index=False)
print("Saved embedding to umap_embedding.csv")

# ---------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------
plt.figure(figsize=(10, 8))
classes = sorted(y_fit.unique())
cmap = plt.get_cmap("tab10")

for i, c in enumerate(classes):
    mask = y_fit.values == c
    plt.scatter(
        embedding[mask, 0], embedding[mask, 1],
        s=4, alpha=0.6, color=cmap(i % 10),
        label=f"Class {c}"
    )

plt.title("UMAP projection of network flow data")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(markerscale=4, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("umap_plot.png", dpi=150)
plt.show()
print("Saved plot to umap_plot.png")