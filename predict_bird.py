import os
import numpy as np
import librosa
import joblib

MODEL_PATH = "models/bird_svm_model.joblib"
SAMPLE_RATE = 22050
N_MFCC = 20


def predict_bird(audio_path):
    if not os.path.isfile(audio_path):
        print("Error: Audio file not found:", audio_path)
        return

    saved_data = joblib.load(MODEL_PATH)
    model = saved_data["model"]
    feature_columns = saved_data["feature_columns"]

    try:
        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        features = np.mean(mfcc, axis=1).reshape(1, -1)

        # Keep feature order consistent with training
        import pandas as pd
        features_df = pd.DataFrame(
            features,
            columns=[f"MFCC_{i+1}" for i in range(N_MFCC)]
        )
        features_df = features_df[feature_columns]

        prediction = model.predict(features_df)[0]

        print("\nBird Species Prediction")
        print("-----------------------")
        print("Audio file:", audio_path)
        print("Predicted species:", prediction)

    except Exception as error:
        print("Prediction failed:", error)


if __name__ == "__main__":
    audio_path = input(
        "Enter the path of a bird audio file: "
    ).strip().strip('"').strip("'")

    predict_bird(audio_path)