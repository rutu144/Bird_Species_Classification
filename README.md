Bird Species Classification Using Audio and Machine Learning

A Python mini-project that classifies bird species from audio recordings using **MFCC (Mel-Frequency Cepstral Coefficients)** features and a **Support Vector Machine (SVM)** classifier.

The current prototype recognizes three classes:
- Asian Koel 
- Crow
- Sparrow

> **Note:** This is an educational prototype trained on a very small dataset (currently 8 recordings). Its accuracy is preliminary and should not be treated as reliable real-world performance.

## Project Objectives

- Load bird audio recordings and standardize them for analysis.
- Extract MFCC audio features.
- Train an SVM classifier to distinguish the selected bird classes.
- Evaluate predictions using cross-validation and a confusion matrix.
- Predict a likely species for an audio file supplied by the user.

## Project Structure

```text
Bird_Species_Classification/
├── Dataset/
│   ├── Asian coel/
│   ├── Crow/
│   └── Sparrow/
├── models/
│   └── bird_svm_model.joblib
├── results/
│   ├── mfcc_features.csv
│   ├── confusion_matrix.png
│   └── mfcc_visualization.png
├── bird_classification.py
├── train_model.py
├── evaluate_model.py
├── plot_mfcc.py
├── predict_bird.py
├── requirements.txt
└── README.md
```

The generated model and result files may be recreated by running the scripts below. If you publish this project, check the licences and sharing terms for all audio recordings before uploading them.

## Requirements

- Python 3.10 recommended
- pip
- The dependencies listed in `requirements.txt`

## Setup

Open a terminal in the project directory:

```bash
cd Bird_Species_Classification
```

Create and activate a virtual environment (recommended):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your system does not have Python 3.10 installed, use a compatible Python version supported by the installed versions of Librosa and scikit-learn.

## How to Run

Run the commands from the project root directory, in this order.

### 1. Extract MFCC Features

```bash
python bird_classification.py
```

This script reads the recordings from `Dataset/`, extracts MFCC features, and writes the feature table to:

```text
results/mfcc_features.csv
```

### 2. Train the SVM Model

```bash
python train_model.py
```

The training script reads the feature CSV, scales the features, evaluates the model using stratified cross-validation, trains the final SVM model, and saves it to:

```text
models/bird_svm_model.joblib
```

The reported cross-validation score can change with the data and split. Because the dataset is very small, this score is only an initial experiment.

### 3. Evaluate the Model

```bash
python evaluate_model.py
```

This script reports metrics such as precision, recall, F1-score, and cross-validation accuracy, and saves the confusion matrix to:

```text
results/confusion_matrix.png
```

### 4. Generate the MFCC Visualization

```bash
python plot_mfcc.py
```

The visualization is saved to:

```text
results/mfcc_visualization.png
```

### 5. Predict a Bird Species from an Audio File

```bash
python predict_bird.py
```

When prompted, enter the path to an audio file. For example, use a filename that actually exists in your dataset:

```text
Dataset/Sparrow/<your-audio-filename>.mp3
```

To list the available Sparrow recordings on Linux:

```bash
ls "Dataset/Sparrow"
```

The program prints the predicted class. Predictions are experimental and may be incorrect, especially for unfamiliar recordings, background noise, or non-bird sounds.

## Methodology

1. **Audio loading:** Load each recording as mono audio at a sample rate of 22,050 Hz.
2. **Feature extraction:** Calculate 20 MFCC coefficients and average them over time to create one feature vector per recording.
3. **Preprocessing:** Use `StandardScaler` to scale the feature values.
4. **Classification:** Train an SVM with an RBF kernel and balanced class weights.
5. **Evaluation:** Use cross-validation and a confusion matrix to inspect performance.
6. **Prediction:** Extract the same features from an input audio file and pass them to the trained model.

## Visualizations

### Confusion Matrix

The confusion matrix compares the true labels with the model's predicted labels. Values on the diagonal represent correct classifications; off-diagonal values represent misclassifications.

### MFCC Visualization

The MFCC plot visualizes how the audio's cepstral features vary over time. It helps illustrate the acoustic features used by the classifier; it does not, by itself, prove that species are correctly identified.

## Limitations and Future Improvements

- Increase the dataset substantially, ideally collecting many varied recordings per species.
- Use recordings from independent sources and avoid having near-duplicate clips in training and evaluation splits.
- Test on recordings not used during training or model selection.
- Compare SVM with other classifiers, such as Random Forest or k-nearest neighbours.
- Explore additional features such as spectral centroid, spectral bandwidth, zero-crossing rate, and chroma.
- Add a confidence or uncertainty display and a way to reject sounds that do not resemble the trained classes.
- Correct the folder label `Asian coel` to `Asian Koel` if desired, then regenerate features, retrain, and update all related labels consistently.

## Technologies Used

- Python
- Librosa
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Joblib
- MFCC feature extraction
- Support Vector Machine (SVM)

## Dataset and Attribution

The recordings were collected from Xeno-canto. Before publishing the dataset or audio files on GitHub, check the licence and attribution requirements for each recording. Include the original recording links, recordist credits, and applicable licences where required. If you cannot redistribute an audio file, publish the code and provide instructions for obtaining it from the original source instead.

## Disclaimer

This project is intended for educational and demonstration purposes. The current dataset is too small to support dependable real-world bird identification. Model outputs are predictions, not guaranteed identifications.
"""
path = Path("/mnt/data/README.md")
path.write_text(content, encoding="utf-8")
print(f"Created: {path}")
