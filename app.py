import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# --- Load model and label encoder ---
model = tf.keras.models.load_model("saved_model/emotion_model.h5")
le = joblib.load("saved_model/label_encoder.pkl")

# --- Constants (should match training params) ---
N_MFCC = 40
MAX_LEN = 174

# --- Feature Extraction ---
def extract_features(audio_bytes, sr=16000):
    y, _ = librosa.load(BytesIO(audio_bytes), sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Pad or truncate to fixed size
    if mfccs.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :MAX_LEN]

    # Reshape for Conv2D: (1, 174, 40, 1)
    features = mfccs.T[np.newaxis, ..., np.newaxis]
    return features, y, sr

# --- Plot waveform ---
def plot_waveform(y, sr):
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

# --- Plot spectrogram ---
def plot_spectrogram(y, sr):
    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    st.pyplot(fig)

# --- Streamlit UI ---
st.set_page_config(page_title="Speech Emotion Recognizer", page_icon="🎙️")
st.title("🎙️ Speech Emotion Detection")
st.write("Upload a `.wav` file to detect the emotion.")

uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    try:
        # --- Extract features ---
        features, y, sr = extract_features(uploaded_file.read())

        # --- Visualize ---
        st.subheader("📊 Audio Visualization")
        plot_waveform(y, sr)
        plot_spectrogram(y, sr)

        # --- Predict ---
        prediction = model.predict(features)
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]

        # --- Results ---
        st.subheader("🎯 Prediction Results")
        st.markdown(f"**🧠 Predicted Emotion:** `{predicted_label.upper()}`")

        st.markdown("#### 🔎 Class Probabilities:")
        for label, prob in zip(le.classes_, prediction[0]):
            st.write(f"- {label}: `{prob:.2f}`")

    except Exception as e:
        st.error(f"⚠️ Error processing audio:\n\n{e}")
