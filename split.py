import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from random import shuffle

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT_PATH = os.path.join(SCRIPT_DIR, 'dataset')

# Emotion mapping dictionaries
EMOTION_MAP = {
    'TESS': {
        'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
        'happy': 'happy', 'neutral': 'neutral', 'surprise': 'surprise',
        'sad': 'sad', 'ps': 'surprise'
    },
    'RAVDESS': {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise'
    },
    'CREMA-D': {
        'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 'HAP': 'happy',
        'NEU': 'neutral', 'SAD': 'sad'
    }
}

TARGET_EMOTIONS = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def collect_dataset_files(dataset_base_path):
    all_file_paths = []
    all_labels = []

    print(f"\n🔍 Collecting and balancing data from: {dataset_base_path}")

    # TESS
    tess_path = os.path.join(dataset_base_path, 'TESS')
    if os.path.exists(tess_path):
        for folder in os.listdir(tess_path):
            folder_path = os.path.join(tess_path, folder)
            if os.path.isdir(folder_path):
                raw_emotion = folder.split('_')[-1].lower()
                emotion = EMOTION_MAP['TESS'].get(raw_emotion)
                if emotion in TARGET_EMOTIONS:
                    for file in glob.glob(os.path.join(folder_path, '*.wav')):
                        all_file_paths.append(file)
                        all_labels.append(emotion)

    # RAVDESS
    ravdess_path = os.path.join(dataset_base_path, 'RAVDESS')
    if os.path.exists(ravdess_path):
        for file in glob.glob(os.path.join(ravdess_path, '**/*.wav'), recursive=True):
            filename = os.path.basename(file)
            parts = filename.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = EMOTION_MAP['RAVDESS'].get(emotion_code)
                if emotion in TARGET_EMOTIONS:
                    all_file_paths.append(file)
                    all_labels.append(emotion)

    # CREMA-D
    cremad_path = os.path.join(dataset_base_path, 'CREMA-D')
    if os.path.exists(cremad_path):
        for file in glob.glob(os.path.join(cremad_path, '*.wav')):
            filename = os.path.basename(file)
            parts = filename.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = EMOTION_MAP['CREMA-D'].get(emotion_code)
                if emotion in TARGET_EMOTIONS:
                    all_file_paths.append(file)
                    all_labels.append(emotion)

    return all_file_paths, all_labels

# --- Main ---
if __name__ == "__main__":
    file_paths, labels = collect_dataset_files(DATASET_ROOT_PATH)

    if len(file_paths) == 0:
        print("❌ ERROR: No audio files found.")
        exit()

    print(f"\n📊 Total samples collected: {len(file_paths)}")
    original_dist = Counter(labels)
    for emotion, count in sorted(original_dist.items()):
        print(f"  {emotion}: {count} samples")

    # Balance the dataset
    print("\n⚖️ Balancing dataset by downsampling to smallest emotion class...")
    min_count = min(original_dist.values())
    balanced_paths = []
    balanced_labels = []

    emotion_to_samples = {emotion: [] for emotion in TARGET_EMOTIONS}
    for path, label in zip(file_paths, labels):
        if label in emotion_to_samples:
            emotion_to_samples[label].append(path)

    for emotion in TARGET_EMOTIONS:
        paths = emotion_to_samples[emotion]
        shuffle(paths)
        balanced_paths.extend(paths[:min_count])
        balanced_labels.extend([emotion] * min_count)

    print(f"✅ Balanced dataset: {len(balanced_paths)} samples total ({min_count} per class)")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_paths, balanced_labels,
        test_size=0.3,
        stratify=balanced_labels,
        random_state=42
    )

    # Save to CSV
    train_df = pd.DataFrame({'filepath': X_train, 'emotion': y_train})
    test_df = pd.DataFrame({'filepath': X_test, 'emotion': y_test})

    train_df.to_csv('training_dataset.csv', index=False)
    test_df.to_csv('testing_dataset.csv', index=False)

    print("\n📁 Datasets saved:")
    print("  ✅ training_dataset.csv")
    print("  ✅ testing_dataset.csv")
