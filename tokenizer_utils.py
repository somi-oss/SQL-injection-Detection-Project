import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 200

def tokenize_data(X_train, X_val, X_test):
    tokenizer = Tokenizer(char_level=True, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq   = tokenizer.texts_to_sequences(X_val)
    X_test_seq  = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post")
    X_val_pad   = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="post")
    X_test_pad  = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post")

    with open("saved/char_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    return X_train_pad, X_val_pad, X_test_pad, tokenizer
