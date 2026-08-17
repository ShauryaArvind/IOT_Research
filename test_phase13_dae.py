import pandas as pd

from src.risk_engine.novelty_detector import DAENoveltyDetector


# Load a small sample of the balanced dataset
X = pd.read_csv(
    "X_train_final_balanced.csv"
).iloc[:100]

print("Input shape:", X.shape)


# Load existing Phase 10 DAE
dae_detector = DAENoveltyDetector()


# Calculate reconstruction error
errors = dae_detector.reconstruction_error(X)


print("\nDAE reconstruction errors:")
print(errors[:10])

print("\nStatistics:")
print("Minimum:", errors.min())
print("Maximum:", errors.max())
print("Mean:", errors.mean())
print("Median:", pd.Series(errors).median())

# Calculate classifier confidence
predicted_class, confidence, probabilities = (
    dae_detector.classifier_confidence(X)
)

print("\nClassifier predictions:")
print(predicted_class[:10])

print("\nClassifier confidence:")
print(confidence[:10])

print("\nConfidence statistics:")
print("Minimum:", confidence.min())
print("Maximum:", confidence.max())
print("Mean:", confidence.mean())
print("Median:", pd.Series(confidence).median())