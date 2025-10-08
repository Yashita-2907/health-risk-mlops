import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load raw data
df = pd.read_csv("data/raw/pima_diabetes.csv")

# Basic cleaning
df = df.dropna()

# Split into features (X) and target (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create processed folder if it doesn’t exist
os.makedirs("data/processed", exist_ok=True)

# Save processed data
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("✅ Preprocessing complete! Files saved to data/processed/")
