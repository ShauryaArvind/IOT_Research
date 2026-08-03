import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading raw files...")
X = pd.read_csv('data.csv')
# Select the specific column name, or use .squeeze() to convert it to a 1D Series
y = pd.read_csv('Label.csv').squeeze()

# Stratified split to maintain 10-class proportions in the test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Save the isolated sets
X_train.to_csv('X_train_raw.csv', index=False)
y_train.to_csv('y_train_raw.csv', index=False)
X_test.to_csv('X_test_isolated.csv', index=False)
y_test.to_csv('y_test_isolated.csv', index=False)

print("File 1 Completed! Train and Test sets isolated.")