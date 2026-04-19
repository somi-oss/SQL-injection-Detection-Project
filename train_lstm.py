from tensorflow.keras.callbacks import EarlyStopping

from data_loader import load_and_split_data
from tokenizer_utils import tokenize_data, MAX_LEN
from model_builder import build_lstm_model

# Load data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    "Balanced_SQL_Dataset.csv"
)

# Tokenize
X_train_pad, X_val_pad, X_test_pad, tokenizer = tokenize_data(
    X_train, X_val, X_test
)

# Build model
model = build_lstm_model(
    vocab_size=len(tokenizer.word_index) + 1,
    max_len=MAX_LEN
)

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Train
model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=10,
    batch_size=128,
    callbacks=[early_stop]
)

# Save model
model.save("saved/lstm_model.h5")
print("✅ Plain LSTM model saved successfully")
