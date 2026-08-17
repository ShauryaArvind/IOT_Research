import pandas as pd

from src.risk_engine.novelty_detector import DAENoveltyDetector


print("Loading balanced training data...")

X_train = pd.read_csv(
    "X_train_final_balanced.csv"
)

print("Training data shape:", X_train.shape)


print("\nInitializing Phase 13 detector...")

detector = DAENoveltyDetector()


print("\nTraining Isolation Forest...")

detector.fit_isolation_forest(
    X_train,
    sample_size=100000
)

print("Isolation Forest training completed!")


# Test on a small sample
X_test_sample = X_train.iloc[:100]

scores = detector.isolation_anomaly_score(
    X_test_sample
)


print("\nIsolation Forest anomaly scores:")
print(scores[:10])


print("\nAnomaly score statistics:")
print("Minimum:", scores.min())
print("Maximum:", scores.max())
print("Mean:", scores.mean())
print("Median:", pd.Series(scores).median())