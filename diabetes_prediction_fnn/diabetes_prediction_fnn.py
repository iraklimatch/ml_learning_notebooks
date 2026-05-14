"""
Practice activity: Predicting diabetes from medical features with a small FNN.

Course: Microsoft Foundations of AI and Machine Learning
Module: Applied Deep Learning

Dataset: Pima Indians Diabetes (768 patients, 8 features, binary outcome).

Architecture: Dense(64, relu) -> Dense(32, relu) -> Dense(1, sigmoid).
Loss: binary_crossentropy. Optimizer: Adam. 50 epochs, batch 32.

The activity calls for MinMaxScaler. We use it, but also flag a real
problem with the raw dataset: zero values in Glucose, BloodPressure,
SkinThickness, Insulin, and BMI are biologically impossible and almost
certainly encode 'missing'. Treating them as real values pulls those
features toward zero and hurts the model. We replace them with the
per-feature median before scaling - this is the same kind of cleanup any
real diagnostic ML pipeline would do.
"""

import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ====================================================================
# 1. Load and analyze
# ====================================================================
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

data = pd.read_csv(URL, names=COLS)
print("First 5 rows:")
print(data.head().to_string(index=False))

print(f"\nShape: {data.shape}")
print("\nMissing values (NaN check):")
print(data.isnull().sum().to_string())

print("\nDescribe:")
print(data.describe().round(2).to_string())

print("\nTarget distribution:")
print(data["Outcome"].value_counts().to_string())
prevalence = data["Outcome"].mean()
print(f"\nPositive rate (baseline accuracy if you always predict 0): "
      f"{(1 - prevalence)*100:.1f}%")


# ====================================================================
# 2. Clean: treat impossible zeros as missing, fill with median
# ====================================================================
SUSPECT_ZEROS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
zeros_before = (data[SUSPECT_ZEROS] == 0).sum()
print("\nZero counts in features that can't legitimately be zero:")
print(zeros_before.to_string())

for col in SUSPECT_ZEROS:
    median = data.loc[data[col] != 0, col].median()
    data[col] = data[col].replace(0, median)
print("After median-imputation, those columns now have 0 zeros.")


# ====================================================================
# 3. Preprocess
# ====================================================================
X = data.drop("Outcome", axis=1).values.astype("float32")
y = data["Outcome"].values.astype("float32")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")


# ====================================================================
# 4. Build, compile, train
# ====================================================================
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1,  activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

t0 = time.time()
history = model.fit(
    X_train, y_train,
    epochs=50, batch_size=32,
    validation_data=(X_test, y_test),
    verbose=2,
)
train_seconds = time.time() - t0


# ====================================================================
# 5. Evaluate
# ====================================================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)
auc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"\nTest loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_acc*100:.2f}%")
print(f"Test ROC AUC:  {auc:.4f}")
print(f"Confusion matrix [[TN FP],[FN TP]]:\n{cm}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["No diabetes", "Diabetes"]))


# ====================================================================
# 6. Plots: training curves + confusion matrix
# ====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(history.history["accuracy"], label="Train")
ax.plot(history.history["val_accuracy"], label="Test")
ax.set_title("Model accuracy")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(history.history["loss"], label="Train")
ax.plot(history.history["val_loss"], label="Test")
ax.set_title("Model loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle(f"Pima Diabetes FNN - final test accuracy {test_acc*100:.2f}%",
             fontsize=12)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Confusion-matrix heatmap
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
ax.set_title(f"Confusion matrix (acc {test_acc*100:.1f}%)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# ====================================================================
# Save histories so the notebook can reuse them
# ====================================================================
np.savez(
    "results.npz",
    test_acc=test_acc, test_loss=test_loss, auc=auc,
    train_acc=np.array(history.history["accuracy"]),
    val_acc=np.array(history.history["val_accuracy"]),
    train_loss=np.array(history.history["loss"]),
    val_loss=np.array(history.history["val_loss"]),
    confusion_matrix=cm,
    train_seconds=train_seconds,
    y_test=y_test, y_pred=y_pred, y_pred_prob=y_pred_prob,
)

print("\nSaved plots:  training_curves.png, confusion_matrix.png")
print("Saved arrays: results.npz")
