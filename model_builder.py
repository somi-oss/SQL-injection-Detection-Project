from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from attention_layer import AttentionLayer


def build_model(vocab_size, max_len):
    # =========================
    # Input layer
    # =========================
    inputs = Input(shape=(max_len,), name="input_tokens")

    # =========================
    # Embedding layer
    # =========================
    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=64,
        input_length=max_len,
        name="embedding"
    )(inputs)

    # =========================
    # LSTM layer (sequence modeling)
    # =========================
    lstm_out = LSTM(
        units=64,
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.3,
        name="lstm"
    )(embedding)

    # =========================
    # Attention layer
    # =========================
    attention_out = AttentionLayer(name="attention")(lstm_out)

    # =========================
    # Fully connected layers
    # =========================
    dense = Dense(
        32,
        activation="relu",
        name="dense"
    )(attention_out)

    dropout = Dropout(
        0.3,
        name="dropout"
    )(dense)

    # =========================
    # Output layer
    # =========================
    outputs = Dense(
        1,
        activation="sigmoid",
        name="output"
    )(dropout)

    # =========================
    # Build & compile model
    # =========================
    model = Model(inputs=inputs, outputs=outputs, name="Attention_LSTM_Model")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def build_lstm_model(vocab_size, max_len):
    inputs = Input(shape=(max_len,), name="input_tokens")

    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=64,
        name="embedding"
    )(inputs)

    lstm_out = LSTM(
        units=64,
        dropout=0.3,
        recurrent_dropout=0.3,
        name="lstm"
    )(embedding)

    dense = Dense(32, activation="relu", name="dense")(lstm_out)
    dropout = Dropout(0.3, name="dropout")(dense)

    outputs = Dense(1, activation="sigmoid", name="output")(dropout)

    model = Model(inputs, outputs, name="Plain_LSTM_Model")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

