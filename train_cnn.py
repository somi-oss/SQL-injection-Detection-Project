from cnn_model import build_cnn_model
from data_loader import load_and_split_data
from tokenizer_utils import tokenize_data, MAX_LEN

X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    "Balanced_SQL_Dataset.csv"
)

X_train_pad, X_val_pad, X_test_pad, tokenizer = tokenize_data(
    X_train, X_val, X_test
)

vocab_size = len(tokenizer.word_index) + 1

model = build_cnn_model(vocab_size, MAX_LEN)

model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=10,
    batch_size=64
)
 
model.save("cnn_model.h5")
