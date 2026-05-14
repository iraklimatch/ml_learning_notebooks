"""
Practice lab: FNN, CNN, and RNN side-by-side in TensorFlow/Keras.

Course: Microsoft Foundations of AI and Machine Learning
Module: Frameworks and Tools

Three architectures, three different data types, one Keras API:

  1. FNN -> Iris (tabular)        -> classification accuracy
  2. CNN -> CIFAR-10 (images)     -> classification accuracy
  3. RNN -> synthetic sine wave   -> mean squared error

The point isn't to beat each task - it's to feel how the same `Sequential`
API plus the right layer type handles wildly different inputs (4 features
vs 32x32x3 images vs sequences of length 50).
"""

import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ====================================================================
# 1. FNN on Iris
# ====================================================================
print("=" * 64)
print("1. FNN on Iris (tabular classification)")
print("=" * 64)

iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode labels (the activity uses categorical_crossentropy)
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

# Scale features so the FNN trains cleanly
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_onehot, test_size=0.2, random_state=SEED, stratify=y
)

model_fnn = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(3, activation="softmax"),
])
model_fnn.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model_fnn.summary()

t0 = time.time()
fnn_hist = model_fnn.fit(
    X_train, y_train,
    epochs=20, batch_size=32,
    validation_data=(X_test, y_test),
    verbose=2,
)
fnn_seconds = time.time() - t0
fnn_loss, fnn_acc = model_fnn.evaluate(X_test, y_test, verbose=0)
print(f"\nFNN test accuracy: {fnn_acc*100:.2f}%  (loss {fnn_loss:.4f})  "
      f"train time {fnn_seconds:.1f}s")


# ====================================================================
# 2. CNN on CIFAR-10
# ====================================================================
print("\n" + "=" * 64)
print("2. CNN on CIFAR-10 (image classification)")
print("=" * 64)

(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.cifar10.load_data()
)
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

model_cnn = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model_cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model_cnn.summary()

t0 = time.time()
cnn_hist = model_cnn.fit(
    train_images, train_labels,
    epochs=10, batch_size=64,
    validation_data=(test_images, test_labels),
    verbose=2,
)
cnn_seconds = time.time() - t0
cnn_loss, cnn_acc = model_cnn.evaluate(test_images, test_labels, verbose=0)
print(f"\nCNN test accuracy: {cnn_acc*100:.2f}%  (loss {cnn_loss:.4f})  "
      f"train time {cnn_seconds:.1f}s")


# ====================================================================
# 3. RNN on a synthetic sine wave
# ====================================================================
print("\n" + "=" * 64)
print("3. RNN on a sine wave (time-series regression)")
print("=" * 64)

# Generate a long sine wave. The activity uses 10,000 timesteps; we keep
# that. The model learns to predict the next value from the previous
# `seq_length` values.
t = np.linspace(0, 100, 10_000)
wave = np.sin(t).astype("float32").reshape(-1, 1)

SEQ_LENGTH = 50


def create_sequences(data, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i + seq_length])
        y_seq.append(data[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


X_seq, y_seq = create_sequences(wave, SEQ_LENGTH)
print(f"Sequences: X {X_seq.shape}  y {y_seq.shape}  (seq_length={SEQ_LENGTH})")

# Chronological split so the model is genuinely predicting the future
split = int(0.8 * len(X_seq))
X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

model_rnn = models.Sequential([
    layers.Input(shape=(SEQ_LENGTH, 1)),
    layers.SimpleRNN(128),
    layers.Dense(1),
])
model_rnn.compile(optimizer="adam", loss="mse")
model_rnn.summary()

t0 = time.time()
rnn_hist = model_rnn.fit(
    X_train_seq, y_train_seq,
    epochs=10, batch_size=32,
    validation_data=(X_test_seq, y_test_seq),
    verbose=2,
)
rnn_seconds = time.time() - t0
rnn_mse = model_rnn.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"\nRNN test MSE: {rnn_mse:.6f}  train time {rnn_seconds:.1f}s")

# Predict the next 200 points to eyeball it
y_pred = model_rnn.predict(X_test_seq[:200], verbose=0).flatten()


# ====================================================================
# 4. Comparison summary
# ====================================================================
print("\n" + "=" * 64)
print("4. Summary")
print("=" * 64)
print(f"{'Model':<8} {'Dataset':<12} {'Metric':<10} {'Value':>12} {'Train s':>10}")
print("-" * 56)
print(f"{'FNN':<8} {'Iris':<12} {'accuracy':<10} {fnn_acc*100:>11.2f}% {fnn_seconds:>10.1f}")
print(f"{'CNN':<8} {'CIFAR-10':<12} {'accuracy':<10} {cnn_acc*100:>11.2f}% {cnn_seconds:>10.1f}")
print(f"{'RNN':<8} {'sine wave':<12} {'MSE':<10} {rnn_mse:>12.6f} {rnn_seconds:>10.1f}")


# ====================================================================
# 5. Plots
# ====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# FNN: accuracy
ax = axes[0]
ax.plot(fnn_hist.history["accuracy"], label="train")
ax.plot(fnn_hist.history["val_accuracy"], label="val", linestyle="--")
ax.set_title(f"FNN on Iris\nfinal test acc {fnn_acc*100:.2f}%")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(True, alpha=0.3)

# CNN: accuracy
ax = axes[1]
ax.plot(cnn_hist.history["accuracy"], label="train")
ax.plot(cnn_hist.history["val_accuracy"], label="val", linestyle="--")
ax.set_title(f"CNN on CIFAR-10\nfinal test acc {cnn_acc*100:.2f}%")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(True, alpha=0.3)

# RNN: loss curves
ax = axes[2]
ax.plot(rnn_hist.history["loss"], label="train")
ax.plot(rnn_hist.history["val_loss"], label="val", linestyle="--")
ax.set_title(f"RNN on sine wave\nfinal test MSE {rnn_mse:.4f}")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle("FNN vs CNN vs RNN - learning curves", fontsize=13)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# RNN: predictions vs ground truth
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test_seq[:200].flatten(), label="True", linewidth=2)
ax.plot(y_pred, label="Predicted", linestyle="--")
ax.set_title("RNN predictions on the held-out sine wave (first 200 points)")
ax.set_xlabel("Timestep"); ax.set_ylabel("Value")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rnn_predictions.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# ====================================================================
# Save histories so the notebook can reuse them without retraining.
# ====================================================================
np.savez(
    "results.npz",
    fnn_acc=fnn_acc, fnn_loss=fnn_loss,
    fnn_train_acc=np.array(fnn_hist.history["accuracy"]),
    fnn_val_acc=np.array(fnn_hist.history["val_accuracy"]),
    fnn_train_loss=np.array(fnn_hist.history["loss"]),
    fnn_val_loss=np.array(fnn_hist.history["val_loss"]),
    cnn_acc=cnn_acc, cnn_loss=cnn_loss,
    cnn_train_acc=np.array(cnn_hist.history["accuracy"]),
    cnn_val_acc=np.array(cnn_hist.history["val_accuracy"]),
    cnn_train_loss=np.array(cnn_hist.history["loss"]),
    cnn_val_loss=np.array(cnn_hist.history["val_loss"]),
    rnn_mse=rnn_mse,
    rnn_train_loss=np.array(rnn_hist.history["loss"]),
    rnn_val_loss=np.array(rnn_hist.history["val_loss"]),
    rnn_true=y_test_seq[:200].flatten(),
    rnn_pred=y_pred,
    fnn_seconds=fnn_seconds, cnn_seconds=cnn_seconds, rnn_seconds=rnn_seconds,
)

print("\nSaved plots:  training_curves.png, rnn_predictions.png")
print("Saved arrays: results.npz")

# Reflection
#
# The FNN result on Iris (~80% test acc) looks low because 20 epochs *
# batch_size 32 = only 80 Adam updates total - Iris is easy but the
# training schedule is starved. Bump epochs to 100 (or shrink batch size)
# and accuracy moves to 95%+. Architecture is rarely the constraint on
# small tabular data; the training budget is.
#
# The CNN result on CIFAR-10 (~70% test acc) is exactly where this small
# 2-conv-block architecture caps out. Pushing past needs data
# augmentation, more depth, batch norm, or dropout.
#
# The RNN MSE is effectively zero - a 128-unit SimpleRNN looking back 50
# steps is wildly overpowered for a clean sine wave. The interesting
# follow-up is to add noise to the wave, or shorten the lookback window,
# and watch the MSE rise.
