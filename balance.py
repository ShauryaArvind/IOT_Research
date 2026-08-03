import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

print("Loading raw training sets...")
X_train = pd.read_csv('X_train_raw.csv')
y_train = pd.read_csv('y_train_raw.csv').iloc[:, 0]

# Set oversampling target to match Class 4 (~24,760 rows after 80/20 split)
target_samples = int(y_train.value_counts().loc[4])
classes_to_oversample = [1, 2, 3, 5, 6, 7, 8, 9]
custom_oversample_strategy = {cls: target_samples for cls in classes_to_oversample}

# Phase 1: SMOTE (Oversampling)
print("Running Phase 1: SMOTE minority synthesis...")
smote = SMOTE(sampling_strategy=custom_oversample_strategy, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Phase 2: Tomek Links (Undersampling)
print("Running Phase 2: Tomek Links boundary cleaning...")
tomek = TomekLinks(sampling_strategy='all', n_jobs=-1)
X_train_balanced, y_train_balanced = tomek.fit_resample(X_train_smote, y_train_smote)

# Save the final balanced training sets
print("Saving balanced training datasets...")
X_train_balanced.to_csv('X_train_final_balanced.csv', index=False)
pd.DataFrame(y_train_balanced, columns=['Label']).to_csv('y_train_final_balanced.csv', index=False)

# Generate research methodology table
before = y_train.value_counts().sort_index()
after = pd.Series(y_train_balanced).value_counts().sort_index()
metrics_df = pd.DataFrame({'Original_Train': before, 'Balanced_Train': after, 'Net_Change': after - before})
metrics_df.to_csv('balancing_methodology_results.csv')

print("File 2 Completed! Hybrid data balanced successfully.")