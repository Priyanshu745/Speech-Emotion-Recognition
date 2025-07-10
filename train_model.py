import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, Reshape
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense, GlobalAveragePooling1D
from tensorflow.keras.layers import Attention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # for saving the encoder

# 🧪 Load features
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# 🏷️ Encode labels
encoder = LabelEncoder()
y_train_enc = to_categorical(encoder.fit_transform(y_train))
y_test_enc = to_categorical(encoder.transform(y_test))
num_classes = y_train_enc.shape[1]

# 🧠 Model Builder
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # CNN block
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    # Residual connection
    residual = Conv2D(128, (1, 1), padding='same')(inputs)
    residual = MaxPooling2D((2, 2))(residual)
    residual = MaxPooling2D((2, 2))(residual)

    x = Add()([x, residual])

    # Reshape for RNN
    shape = x.shape
    x = Reshape((shape[1] * shape[2], shape[3]))(x)  # (batch, time, features)

    # BiLSTM + GRU
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Dropout(0.4)(x)

    # Attention
    attention = Attention()([x, x])
    x = LayerNormalization()(attention)

    # Dense
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

# ⚙️ Training pipeline
def train_model():
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Reshape input for CNN
    X_train_reshaped = X_train[..., np.newaxis]
    X_test_reshaped = X_test[..., np.newaxis]

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    # Train
    history = model.fit(
        X_train_reshaped, y_train_enc,
        validation_data=(X_test_reshaped, y_test_enc),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )

    # 🔍 Evaluation
    y_pred = model.predict(X_test_reshaped)
    y_pred_labels = encoder.inverse_transform(np.argmax(y_pred, axis=1))
    y_true_labels = encoder.inverse_transform(np.argmax(y_test_enc, axis=1))

    print("\n📊 Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels))

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # 💾 Save model and encoder
    model.save("emotion_model.h5")
    joblib.dump(encoder, "label_encoder.pkl")
    print("\n✅ Model saved as 'emotion_model.h5'")
    print("✅ Label encoder saved as 'label_encoder.pkl'")

# ▶️ Run
if __name__ == "__main__":
    train_model()
