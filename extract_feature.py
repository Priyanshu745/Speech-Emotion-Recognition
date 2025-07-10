import pandas as pd
import numpy as np
import librosa
import os
from tqdm import tqdm

# --- Parameters ---
n_mfcc = 40          # Number of MFCC features
max_pad_len = 174    # Number of time steps

# --- Feature extraction function ---
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # Pad or trim MFCCs to fixed length
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs.T  # Shape: (174, 40)
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# --- Load CSVs ---
train_df = pd.read_csv('training_dataset.csv')
test_df = pd.read_csv('testing_dataset.csv')

# --- Extract training features ---
print("🎧 Extracting training features...")
X_train, y_train = [], []
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    features = extract_features(row['filepath'])
    if features is not None:
        X_train.append(features)
        y_train.append(row['emotion'])

# --- Extract testing features ---
print("🎧 Extracting testing features...")
X_test, y_test = [], []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    features = extract_features(row['filepath'])
    if features is not None:
        X_test.append(features)
        y_test.append(row['emotion'])

# --- Convert to arrays ---
X_train = np.array(X_train)  # Shape: (N, 174, 40)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# --- Save as .npy ---
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("\n✅ Features saved:")
print(" - X_train.npy")
print(" - y_train.npy")
print(" - X_test.npy")
print(" - y_test.npy")
