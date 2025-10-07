import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load Dataset
data = pd.read_csv('data/raw/diabetes.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, 'models/diabetes_model.pkl')
print("✅ Model saved successfully!")
