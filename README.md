# 🎵 Music Genre Classification

## Overview

This project implements a **music genre classification** system using machine learning and deep learning techniques. The goal is to automatically identify the genre of a given audio track by analyzing its audio features.

---

## Features

- **Audio Feature Extraction:** Uses `librosa` to extract features like:
  - MFCCs
  - Chroma
  - Spectral Contrast
  - Tonnetz
- **Model Training:**
  - Traditional ML models: SVM, Random Forest
  - Deep Learning models: CNN
- **Evaluation Metrics:** Accuracy, confusion matrix, precision, recall
- **Jupyter Notebook Interface:** All code organized for step-by-step execution

---

## 🎧 Dataset

- **GTZAN Genre Collection**
  - 1000 audio tracks (30 sec each)
  - Genres: Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock
- **Format:** 22050Hz Mono 16-bit WAV files

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music-genre-classification.git
   cd music-genre-classification
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Download the GTZAN dataset and extract it into the `data/` directory.
2. Open the Jupyter notebook:
   ```bash
   cd notebooks
   jupyter notebook
   ```
3. Run the notebook cells to:
   - Load and visualize audio
   - Extract features
   - Train models
   - Evaluate classification results

---

## Example

```python
import librosa

# Load a sample audio file
y, sr = librosa.load('data/blues/blues.00000.wav')

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=y, sr=sr)

print(mfccs.shape)
```

---

## Results

- Achieved **high accuracy** on test set
- Confusion matrix and performance graphs provided in the notebook
- Comparative evaluation between ML and DL models

---
## Author
Kavin Balaji
