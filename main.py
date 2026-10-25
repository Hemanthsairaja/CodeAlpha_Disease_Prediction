"""Educational breast cancer classification with the public Wisconsin dataset."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from evaluation import evaluate

BASE = Path(__file__).resolve().parent

def train():
    output_dir = BASE/"outputs"
    output_dir.mkdir(exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    # sklearn target: 0 = malignant, 1 = benign
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=42)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ])
    model.fit(X_train, y_train)
    # Treat malignant (class 0) as positive for sensitivity and ROC-AUC.
    evaluate(model, X_test, y_test, output_dir, positive_label=0)
    joblib.dump(model, output_dir/"disease_model.joblib")
    print("Class labels: 0 = malignant, 1 = benign")
    print("Precision, recall, F1 and ROC-AUC use malignant as the positive class.")
    print("Saved outputs in:", output_dir)

def demo():
    model = joblib.load(BASE/"outputs"/"disease_model.joblib")
    example = load_breast_cancer(as_frame=True).data.iloc[[0]]
    predicted = int(model.predict(example)[0])
    print("Demo on dataset's first record:", "malignant" if predicted == 0 else "benign")
    print("This is NOT a medical diagnostic tool.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    demo() if args.demo else train()
