import os
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)

# Load extracted features
data = pd.read_csv("results/mfcc_features.csv")

X = data.drop(columns=["filename", "species"])
y = data["species"]

# Load trained model
saved_data = joblib.load("models/bird_svm_model.joblib")
model = saved_data["model"]

# Perform cross-validation
minimum_class_count = y.value_counts().min()
n_splits = min(2, minimum_class_count)

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

predictions = cross_val_predict(model, X, y, cv=cv)

# Display evaluation results
print("\nBird Species Classification Report")
print("----------------------------------")
print(classification_report(y, predictions, zero_division=0))

accuracy = accuracy_score(y, predictions)
print(f"Cross-validation accuracy: {accuracy:.2%}")

# Generate confusion matrix
os.makedirs("results", exist_ok=True)

display = ConfusionMatrixDisplay.from_predictions(
    y,
    predictions,
    labels=saved_data["classes"],
    xticks_rotation=45,
    cmap="Blues"
)

display.ax_.set_title("Bird Species Classification - Confusion Matrix")
display.figure_.tight_layout()

display.figure_.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(display.figure_)

print("\nConfusion matrix saved to:")
print("results/confusion_matrix.png")