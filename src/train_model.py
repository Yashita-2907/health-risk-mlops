# src/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import yaml
import os

# Read hyperparameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]
n_estimators = params["train"]["n_estimators"]
max_depth = params["train"]["max_depth"]
min_samples_split = params["train"]["min_samples_split"]

# Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# ---------------------- #
# 1️⃣ Train on Diabetes Data
# ---------------------- #
print("🔹 Training Diabetes Model...")
X_train_d = pd.read_csv("data/processed/X_train_diabetes.csv")
X_test_d = pd.read_csv("data/processed/X_test_diabetes.csv")
y_train_d = pd.read_csv("data/processed/y_train_diabetes.csv").values.ravel()
y_test_d = pd.read_csv("data/processed/y_test_diabetes.csv").values.ravel()

clf_diabetes = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    random_state=random_state
)
clf_diabetes.fit(X_train_d, y_train_d)

y_pred_d = clf_diabetes.predict(X_test_d)
acc_d = accuracy_score(y_test_d, y_pred_d)
report_d = classification_report(y_test_d, y_pred_d, output_dict=True)
print(f"✅ Diabetes Model Accuracy: {acc_d:.4f}")

joblib.dump(clf_diabetes, "models/diabetes_model.pkl")

# ---------------------- #
# 2️⃣ Train on Heart Disease Data
# ---------------------- #
print("🔹 Training Heart Disease Model...")
X_train_h = pd.read_csv("data/processed/X_train_heart.csv")
X_test_h = pd.read_csv("data/processed/X_test_heart.csv")
y_train_h = pd.read_csv("data/processed/y_train_heart.csv").values.ravel()
y_test_h = pd.read_csv("data/processed/y_test_heart.csv").values.ravel()

clf_heart = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    random_state=random_state
)
clf_heart.fit(X_train_h, y_train_h)

y_pred_h = clf_heart.predict(X_test_h)
acc_h = accuracy_score(y_test_h, y_pred_h)
report_h = classification_report(y_test_h, y_pred_h, output_dict=True)
print(f"✅ Heart Disease Model Accuracy: {acc_h:.4f}")

joblib.dump(clf_heart, "models/heart_model.pkl")

# ---------------------- #
# 3️⃣ Save Combined Metrics
# ---------------------- #
with open("metrics/scores.txt", "w") as f:
    f.write("=== Diabetes Model ===\n")
    f.write(f"Accuracy: {acc_d:.4f}\n")
    f.write(f"F1 (Class 0): {report_d['0']['f1-score']:.4f}\n")
    f.write(f"F1 (Class 1): {report_d['1']['f1-score']:.4f}\n\n")

    f.write("=== Heart Disease Model ===\n")
    f.write(f"Accuracy: {acc_h:.4f}\n")
    f.write(f"F1 (Class 0): {report_h['0']['f1-score']:.4f}\n")
    f.write(f"F1 (Class 1): {report_h['1']['f1-score']:.4f}\n")

print("🎯 Training complete! Models and metrics saved successfully.")
