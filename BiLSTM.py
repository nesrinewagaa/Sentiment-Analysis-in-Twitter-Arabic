# =========================
# BiLSTM.py
# Arabic Tweet Sentiment Analysis - BiLSTM Model
# =========================

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# =========================
# 1. Load Preprocessed Data
# =========================
X_train = np.load('X_train.npy', allow_pickle=True)
X_test  = np.load('X_test.npy', allow_pickle=True)
Y_train = np.load('Y_train.npy', allow_pickle=True)
Y_test  = np.load('Y_test.npy', allow_pickle=True)

print("Data loaded successfully.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# =========================
# 2. Build BiLSTM Model
# =========================
embedding_dim = 100
dropout_rate = 0.5
lstm_units = 100
max_sequence_length = X_train.shape[1]
vocab_size = 20000

model = Sequential()

# Embedding Layer
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=max_sequence_length))

# Bidirectional LSTM Layer
model.add(layers.Bidirectional(
    layers.LSTM(units=lstm_units,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                return_sequences=True)
))

# Global Max Pooling
model.add(layers.GlobalMaxPool1D())

# Fully Connected Layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(dropout_rate))

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(dropout_rate))

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(dropout_rate))

# Output Layer (3 classes)
model.add(layers.Dense(3, activation='softmax'))

# Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# =========================
# 3. Train the Model
# =========================
history = model.fit(X_train, Y_train,
                    epochs=10,
                    batch_size=50,
                    validation_data=(X_test, Y_test))

# =========================
# 4. Plot Training History
# =========================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
