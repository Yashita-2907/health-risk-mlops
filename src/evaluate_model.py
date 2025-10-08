import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

# -------------------------------
# Load test data and trained model
# -------------------------------
X_test_path = os.path.join("data", "processed", "X_test.csv")
y_test_path = os.path.join("data", "processed", "y_test.csv")
model_path = os.path.join("models", "diabetes_model.pkl")

# Load files
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).values.ravel()  # convert dataframe column to 1D array

# Load trained model
model = joblib.load(model_path)

# -------------------------------
# Evaluate the model
# -------------------------------
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("✅ Model Evaluation Results (Diabetes Dataset):")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# Save evaluation metrics
# -------------------------------
metrics = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1
}

os.makedirs("reports", exist_ok=True)
pd.DataFrame([metrics]).to_csv("reports/diabetes_metrics.csv", index=False)

print("\n💾 Evaluation metrics saved to 'reports/diabetes_metrics.csv'")
