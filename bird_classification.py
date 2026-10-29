from pathlib import Path
import librosa
import numpy as np
import pandas as pd

# Project paths
DATASET_PATH = Path("Dataset")
RESULTS_PATH = Path("results")

RESULTS_PATH.mkdir(exist_ok=True)

AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg"}

feature_rows = []

# Check dataset folder
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH.resolve()}"
    )

# Process each bird species
for species_folder in sorted(DATASET_PATH.iterdir()):

    if not species_folder.is_dir():
        continue

    species = species_folder.name

    for audio_file in sorted(species_folder.iterdir()):

        if audio_file.suffix.lower() not in AUDIO_FORMATS:
            continue

        try:
            # Load audio
            audio, sr = librosa.load(
                str(audio_file),
                sr=22050,
                mono=True
            )

            if len(audio) < 2048:
                continue

            # Extract 20 MFCC coefficients
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=20
            )

            # Calculate mean MFCC features
            mfcc_mean = np.mean(mfcc, axis=1)

            # Store features and label
            row = {
                "filename": audio_file.name,
                "species": species
            }

            for i, value in enumerate(mfcc_mean, start=1):
                row[f"MFCC_{i}"] = float(value)

            feature_rows.append(row)

        except Exception:
            continue

# Save features and display summary
if feature_rows:

    df = pd.DataFrame(feature_rows)

    output_file = RESULTS_PATH / "mfcc_features.csv"
    df.to_csv(output_file, index=False)

    print(f"Total recordings: {len(df)}")
    print("\nRecordings per species:")
    print(df["species"].value_counts().to_string())

else:
    print("No features extracted.")
