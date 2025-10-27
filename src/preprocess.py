import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Ensure processed folder exists
os.makedirs("data/processed", exist_ok=True)

# ---------------------- #
# 1️⃣ Pima Diabetes Data
# ---------------------- #
print("🔹 Processing Pima Diabetes Dataset...")
diabetes = pd.read_csv("data/raw/pima_diabetes.csv")
diabetes = diabetes.dropna()

X_d = diabetes.drop("Outcome", axis=1)
y_d = diabetes["Outcome"]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)

X_train_d.to_csv("data/processed/X_train_diabetes.csv", index=False)
X_test_d.to_csv("data/processed/X_test_diabetes.csv", index=False)
y_train_d.to_csv("data/processed/y_train_diabetes.csv", index=False)
y_test_d.to_csv("data/processed/y_test_diabetes.csv", index=False)

print("✅ Diabetes data processed and saved!")
diabetes.to_csv("data/processed/clean_diabetes.csv", index=False)



# ---------------------- #
# 2️⃣ Heart Disease Data
# ---------------------- #
print("🔹 Processing Heart Disease Dataset...")
heart = pd.read_csv("data/raw/heart_disease_uci.csv")

# Drop rows with missing values
heart = heart.dropna()

# Convert categorical columns to numeric
heart = pd.get_dummies(heart, drop_first=True)

# Split features and target
X_h = heart.drop("num", axis=1)
y_h = heart["num"]

# Split into train/test
from sklearn.model_selection import train_test_split
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

# Save processed data
X_train_h.to_csv("data/processed/X_train_heart.csv", index=False)
X_test_h.to_csv("data/processed/X_test_heart.csv", index=False)
y_train_h.to_csv("data/processed/y_train_heart.csv", index=False)
y_test_h.to_csv("data/processed/y_test_heart.csv", index=False)
heart.to_csv("data/processed/clean_heart.csv", index=False)

print("✅ Heart disease data processed and saved!")




