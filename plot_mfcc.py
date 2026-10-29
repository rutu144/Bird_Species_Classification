import os
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Select one audio recording
audio_path = "Dataset/Sparrow"

files = [
    os.path.join(audio_path, filename)
    for filename in os.listdir(audio_path)
    if filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))
]

if not files:
    raise FileNotFoundError("No audio recordings found in " + audio_path)

# Load the first recording
y, sr = librosa.load(files[0], sr=22050, mono=True)

# Extract MFCC features
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

# Plot and save
os.makedirs("results", exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 5))

img = librosa.display.specshow(
    mfcc,
    x_axis="time",
    sr=sr,
    ax=ax
)

fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.set_title("MFCC Features - Sparrow Audio")
ax.set_ylabel("MFCC Coefficient")

fig.tight_layout()
fig.savefig("results/mfcc_visualization.png", dpi=300)
plt.close(fig)

print("MFCC graph saved to results/mfcc_visualization.png")