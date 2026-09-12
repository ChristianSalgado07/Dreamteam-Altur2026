import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Load the cached dataset and raw XGBoost model
valid_results = joblib.load("features_cache.pkl")
model = joblib.load("model.pkl")

# Extract features, labels, and dataset splits
X = np.array([r[0] for r in valid_results])
y = np.array([r[1] for r in valid_results])
splits = np.array([r[2] for r in valid_results])

available_splits = np.unique(splits)
print(f"Available data splits found: {available_splits}")

# Dynamically select the best available evaluation data
if "test" in available_splits:
    eval_mask = splits == "test"
    print("Evaluating strictly on 'test' split...")
elif "val" in available_splits:
    eval_mask = splits == "val"
    print("Evaluating on 'val' split...")
else:
    eval_mask = np.ones(len(splits), dtype=bool)
    print("No isolated test split found. Evaluating accuracy across the full dataset...")

X_eval = X[eval_mask]
y_eval = y[eval_mask]

# Generate predictions
predictions = model.predict(X_eval)

# Calculate and print metrics
accuracy = accuracy_score(y_eval, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_eval, predictions, target_names=["Human (0)", "Synthetic (1)"]))