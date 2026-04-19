import numpy as np
import random
import tensorflow as tf
from collections import OrderedDict

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from tensorflow.keras.models import load_model
from attention_layer import AttentionLayer

from data_loader import load_and_split_data
from tokenizer_utils import tokenize_data
from tfidf_utils import get_tfidf_features

# =========================
# Set seeds for reproducibility
# =========================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# Load and preprocess data
# =========================
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    "Balanced_SQL_Dataset.csv"
)

# TF-IDF features for classical ML models
train_feats, test_feats = get_tfidf_features(X_train, X_test)

# Tokenized & padded sequences for deep learning models
_, _, X_test_pad, tokenizer = tokenize_data(X_train, X_val, X_test)

# =========================
# Train / Load models
# =========================

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(train_feats, y_train)

# SVM
svm = SVC(kernel="linear", probability=True)
svm.fit(train_feats, y_train)

# CNN
cnn = load_model("cnn_model.h5")

# LSTM
lstm = load_model("lstm_model.h5")

# Attention-LSTM
att = load_model("attention_lstm_model.h5", custom_objects={"AttentionLayer": AttentionLayer})

# =========================
# Prepare models for automatic comparison
# =========================
models = OrderedDict({
    "Logistic Regression": lr,
    "SVM": svm,
    "CNN": cnn,
    "LSTM": lstm,
    "Attention-LSTM": att
})

# =========================
# Compute metrics safely
# =========================
results = OrderedDict()

for name, model in models.items():
    try:
        # Predict depending on model type
        if name in ["Logistic Regression", "SVM"]:
            X_input = test_feats
            y_pred = model.predict(X_input)
        else:  # Keras models
            X_input = X_test_pad
            y_pred = (model.predict(X_input) > 0.5).astype(int)
        
        # Compute accuracy
        acc = accuracy_score(y_test, y_pred)
        
        # Compute precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        
        # Store results
        results[name] = (acc, precision, recall, f1)
    
    except Exception as e:
        # Handle any unexpected error and store N/A
        print(f"Warning: Could not compute metrics for {name}: {e}")
        results[name] = (None, None, None, None)

# =========================
# Print comparison table
# =========================
print("\nMODEL COMPARISON")
print("{:<25} {:<8} {:<8} {:<8} {:<8}".format(
    "Model", "Acc", "Prec", "Recall", "F1"
))
print("-" * 60)

for model, (acc, precision, recall, f1) in results.items():
    # Replace None with N/A for display
    acc_str = f"{acc:.4f}" if acc is not None else "N/A"
    prec_str = f"{precision:.4f}" if precision is not None else "N/A"
    recall_str = f"{recall:.4f}" if recall is not None else "N/A"
    f1_str = f"{f1:.4f}" if f1 is not None else "N/A"

    print(f"{model:<25} {acc_str:<8} {prec_str:<8} {recall_str:<8} {f1_str:<8}")
