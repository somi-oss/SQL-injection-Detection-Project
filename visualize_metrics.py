import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import load_model
from attention_layer import AttentionLayer

from data_loader import load_and_split_data
from tokenizer_utils import tokenize_data

# =====================================================
# Output directory
# =====================================================
os.makedirs("results/plots", exist_ok=True)

# =====================================================
# TF-IDF features
# =====================================================
def get_tfidf_features(train_texts, test_texts):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=20000
    )
    return vectorizer.fit_transform(train_texts), vectorizer.transform(test_texts)

# =====================================================
# Load data
# =====================================================
print("\nLoading dataset...")
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
    "Balanced_SQL_Dataset.csv"
)

print("Extracting TF-IDF features...")
train_feats, test_feats = get_tfidf_features(X_train, X_test)

print("Tokenizing for deep learning models...")
_, _, X_test_pad, _ = tokenize_data(X_train, X_val, X_test)

# =====================================================
# Containers
# =====================================================
metric_results = {}
prediction_probs = {}

# =====================================================
# Logistic Regression
# =====================================================
print("Evaluating Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(train_feats, y_train)

y_pred = lr.predict(test_feats)
y_prob = lr.predict_proba(test_feats)[:, 1]

acc = accuracy_score(y_test, y_pred)
p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

metric_results["Logistic Regression"] = (acc, p, r, f)
prediction_probs["Logistic Regression"] = y_prob

# =====================================================
# SVM (Linear)
# =====================================================
print("Evaluating SVM (Linear)...")
svm = SVC(kernel="linear", probability=True, random_state=42)
svm.fit(train_feats, y_train)

y_pred = svm.predict(test_feats)
y_prob = svm.predict_proba(test_feats)[:, 1]

acc = accuracy_score(y_test, y_pred)
p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

metric_results["SVM (Linear)"] = (acc, p, r, f)
prediction_probs["SVM (Linear)"] = y_prob

# =====================================================
# CNN
# =====================================================
print("Loading CNN...")
cnn_model = load_model("saved/cnn_model.h5", compile=False)

y_prob = cnn_model.predict(X_test_pad).flatten()
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

metric_results["CNN"] = (acc, p, r, f)
prediction_probs["CNN"] = y_prob

# =====================================================
# LSTM Baseline
# =====================================================
print("Loading LSTM (Baseline)...")
lstm_model = load_model("saved/lstm_model.h5", compile=False)

y_prob = lstm_model.predict(X_test_pad).flatten()
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

metric_results["LSTM (Baseline)"] = (acc, p, r, f)
prediction_probs["LSTM (Baseline)"] = y_prob

# =====================================================
# Attention-LSTM
# =====================================================
print("Loading Attention-LSTM...")
attention_model = load_model(
    "saved/attention_lstm_model.h5",
    custom_objects={"AttentionLayer": AttentionLayer},
    compile=False
)

y_prob = attention_model.predict(X_test_pad).flatten()
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

metric_results["Attention-LSTM"] = (acc, p, r, f)
prediction_probs["Attention-LSTM"] = y_prob

# =====================================================
# Model order
# =====================================================
models_list = [
    "Logistic Regression",
    "SVM (Linear)",
    "CNN",
    "LSTM (Baseline)",
    "Attention-LSTM"
]
def save_plot(fig, filename):
    filepath = os.path.join("results/plots", f"{filename}.pdf")
    fig.tight_layout()
    fig.savefig(filepath, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

# =====================================================
# FINAL CLEAR BAR CHART (BIGGER + READABLE TEXT)
# =====================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13
})

fig, ax = plt.subplots(figsize=(18, 9))   # Bigger figure for clearer PDF text

# ---------- Convert metrics to percentage ----------
accuracy_vals  = [metric_results[m][0] * 100 for m in models_list]
precision_vals = [metric_results[m][1] * 100 for m in models_list]
recall_vals    = [metric_results[m][2] * 100 for m in models_list]
f1_vals        = [metric_results[m][3] * 100 for m in models_list]

# ---------- X positions ----------
x = np.arange(len(models_list))
width = 0.18

# ---------- Plot grouped bars ----------
bars1 = ax.bar(x - 1.5 * width, accuracy_vals,  width, label="Accuracy (%)",  color="#4C72B0")
bars2 = ax.bar(x - 0.5 * width, precision_vals, width, label="Precision (%)", color="#55A868")
bars3 = ax.bar(x + 0.5 * width, recall_vals,    width, label="Recall (%)",    color="#C44E52")
bars4 = ax.bar(x + 1.5 * width, f1_vals,        width, label="F1-score (%)",  color="#8172B3")

# ---------- Title and labels ----------
ax.set_title(
    "Performance Comparison of SQL Injection Detection Models",
    fontsize=18,
    fontweight="bold",
    pad=28
)

ax.set_xlabel("Models", fontsize=14, fontweight="bold")
ax.set_ylabel("Performance (%)", fontsize=14, fontweight="bold")

# ---------- X-axis ----------
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=10, ha="right", fontsize=11)

# ---------- Y-axis ----------
ax.set_ylim(0, 105)
ax.tick_params(axis='y', labelsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# ---------- Add values on top of bars ----------
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.8,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# ---------- Legend (clear and readable) ----------
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=4,
    frameon=False,
    fontsize=11
)

# ---------- Add extra top margin ----------
plt.subplots_adjust(top=0.82)

# ---------- Save and show ----------
save_plot(fig, "performance_comparison_final_clean")
plt.show()
# =====================================================
# Confusion Matrices
# =====================================================
for m in models_list:
    cm = confusion_matrix(y_test, (prediction_probs[m] >= 0.5).astype(int))

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix – {m}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    save_plot(fig, f"confusion_matrix_{m}")

# =====================================================
# ROC Curves
# =====================================================
fig, ax = plt.subplots(figsize=(8, 6))
for m in models_list:
    fpr, tpr, _ = roc_curve(y_test, prediction_probs[m])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{m} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC-AUC Curve Comparison")
ax.legend(loc="lower right")

save_plot(fig, "roc_auc_curve")

print("\nAll plots saved in: results/plots/")
