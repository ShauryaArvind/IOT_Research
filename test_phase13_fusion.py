import pandas as pd

from src.risk_engine.novelty_detector import DAENoveltyDetector


print("Loading balanced training data...")

X_train = pd.read_csv(
    "X_train_final_balanced.csv"
)

print("Training data shape:", X_train.shape)


print("\nInitializing detector...")

detector = DAENoveltyDetector()


print("\nTraining Isolation Forest...")

detector.fit_isolation_forest(
    X_train,
    sample_size=100000
)

print("Isolation Forest ready!")


# Use a representative known-data sample
X_known = X_train.iloc[:5000]


print("\nCalibrating thresholds...")

thresholds = detector.calibrate_thresholds(
    X_known
)

print("\nLearned thresholds:")

for name, value in thresholds.items():
    print(f"{name}: {value}")


# Test on a smaller subset
X_test = X_train.iloc[:100]


print("\nCalculating novelty risk...")

results = detector.calculate_novelty_risk(
    X_test
)


print("\nFirst 10 DAE risks:")
print(results["dae_risk"][:10])


print("\nFirst 10 confidence risks:")
print(results["confidence_risk"][:10])


print("\nFirst 10 isolation risks:")
print(results["isolation_risk"][:10])


print("\nFirst 10 novelty risks:")
print(results["novelty_risk"][:10])


print("\nNovelty Risk Statistics:")

risk = results["novelty_risk"]

print("Minimum:", risk.min())
print("Maximum:", risk.max())
print("Mean:", risk.mean())
print("Median:", pd.Series(risk).median())
