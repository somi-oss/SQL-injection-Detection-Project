from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

def build_cnn_model(vocab_size, max_len):
    inputs = Input(shape=(max_len,))
    embed = Embedding(vocab_size, 64)(inputs)

    conv = Conv1D(128, 5, activation="relu")(embed)
    pool = GlobalMaxPooling1D()(conv)

    dense = Dense(64, activation="relu")(pool)
    drop = Dropout(0.3)(dense)
    outputs = Dense(1, activation="sigmoid")(drop)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
