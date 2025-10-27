# src/evaluate_model.py
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model_path, X_test_path, y_test_path, model_name):
    # Load model and test data
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    # Predictions
    y_pred = model.predict(X_test)

    # Detect if it's multiclass or binary
    unique_labels = np.unique(y_test)
    avg_type = 'binary' if len(unique_labels) == 2 else 'weighted'

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
    rec = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg_type, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📊 {model_name} Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # Confusion Matrix Heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'metrics/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')

    # ROC Curve (only for binary)
    if len(unique_labels) == 2 and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(f'metrics/{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')

    return {
        'model': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }


if __name__ == "__main__":
    # Diabetes (binary)
    diabetes_results = evaluate_model(
        model_path="models/diabetes_model.pkl",
        X_test_path="data/processed/X_test_diabetes.csv",
        y_test_path="data/processed/y_test_diabetes.csv",
        model_name="Diabetes_Model"
    )

    # Heart Disease (possibly multiclass)
    heart_results = evaluate_model(
        model_path="models/heart_model.pkl",
        X_test_path="data/processed/X_test_heart.csv",
        y_test_path="data/processed/y_test_heart.csv",
        model_name="Heart_Disease_Model"
    )

    # Combine and save results
    results_df = pd.DataFrame([diabetes_results, heart_results])
    results_df.to_csv("metrics/all_model_scores.csv", index=False)
    print("\n✅ All model metrics saved to metrics/all_model_scores.csv")
