import os
import random
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

from data_loader import load_and_split_data
from tokenizer_utils import tokenize_data, MAX_LEN
from model_builder import build_model


# ============================
# Reproducibility
# ============================
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


# ============================
# Create directories
# ============================
os.makedirs("saved", exist_ok=True)


# ============================
# Load dataset
# ============================
print("\nLoading dataset...")

X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    "Balanced_SQL_Dataset.csv"
)

X_train_pad, X_val_pad, X_test_pad, tokenizer = tokenize_data(
    X_train, X_val, X_test
)


# ============================
# Build Attention-LSTM model
# ============================
print("\nBuilding Attention-LSTM model...")

model = build_model(
    vocab_size=len(tokenizer.word_index) + 1,
    max_len=MAX_LEN
)

model.summary()


# ============================
# Callbacks
# ============================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ============================
# Train model
# ============================
print("\nTraining model...")

model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# ============================
# Evaluate model
# ============================
print("\nEvaluating model...")

y_prob = model.predict(X_test_pad).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ============================
# Save model (Keras v3 SAFE)
# ============================
print("\nSaving model...")

model.save("saved/attention_lstm_model.keras")

print("✅ Model saved successfully at: saved/attention_lstm_model.keras")
