import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(csv_path):
    df = pd.read_csv(csv_path)

    texts = df["Query"].astype(str).values
    labels = df["Label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
