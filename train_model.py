from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_PATH = Path("results/mfcc_features.csv")
MODEL_PATH = Path("models/bird_svm_model.joblib")


def train_model():
    if not DATA_PATH.exists():
        print("Feature dataset not found. Run bird_classification.py first.")
        return

    # Load extracted MFCC features
    data = pd.read_csv(DATA_PATH)

    feature_columns = [
        column for column in data.columns
        if column.startswith("MFCC_")
    ]

    X = data[feature_columns]
    y = data["species"]

    if y.value_counts().min() < 2:
        print("At least 2 recordings per species are required.")
        return

    # Create the SVM pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            class_weight="balanced"
        ))
    ])

    # Evaluate using stratified cross-validation
    n_splits = min(3, int(y.value_counts().min()))
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(model, X, y, cv=cv)

    print(f"{n_splits}-fold cross-validation scores: {scores}")
    print(f"Mean cross-validation accuracy: {scores.mean():.2%}")

    # Train the final model on all available recordings
    model.fit(X, y)

    # Save model and feature information
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": model,
        "feature_columns": feature_columns,
        "classes": list(model.classes_)
    }, MODEL_PATH)

    print("\nSVM training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Classes: {', '.join(map(str, model.classes_))}")


if __name__ == "__main__":
    train_model()