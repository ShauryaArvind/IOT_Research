import pandas as pd

print("Checking Final Dataset Proportions...\n")
y_train_final = pd.read_csv('y_train_final_balanced.csv')['Label']
y_test_final = pd.read_csv('y_test_isolated.csv')['Label']

print("=== FINAL BALANCED TRAINING SET ===")
print(y_train_final.value_counts().sort_index())
print(f"\nTotal Training Rows: {len(y_train_final)}")

print("\n=== UNTOUCHED TEST SET (FOR VALIDATION) ===")
print(y_test_final.value_counts().sort_index())
print(f"\nTotal Testing Rows: {len(y_test_final)}")