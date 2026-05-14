"""
Practice activity: Implementing a feedforward neural network in TensorFlow.

Course: Microsoft Foundations of AI and Machine Learning
Module: Frameworks and Tools

Dataset: Fashion MNIST (28x28 grayscale, 10 classes).

Two models are trained back-to-back so we can compare them:
  1. Baseline FNN  -> Flatten -> Dense(128, relu) -> Dense(10, softmax)
  2. Deeper FNN    -> Flatten -> Dense(128, relu) -> Dense(64, relu)
                              -> Dense(10, softmax)

Same optimizer (Adam), loss (sparse_categorical_crossentropy), 10 epochs,
batch size 32. Goal: see whether the extra hidden layer actually helps.
"""

import os

# Quiet TF startup noise on Windows
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

tf.random.set_seed(42)
np.random.seed(42)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# --------------------------------------------------------------------
# 1. Load and preprocess the data
# --------------------------------------------------------------------
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.fashion_mnist.load_data()
)
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

print(f"Train shape: {train_images.shape}  Test shape: {test_images.shape}")
print(f"Classes: {len(CLASS_NAMES)}")


# --------------------------------------------------------------------
# 2. Define + compile + train two models
# --------------------------------------------------------------------
def build_baseline():
    return models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])


def build_deeper():
    return models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])


def compile_and_fit(model, name, epochs=10, batch_size=32):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\n--- Training '{name}' ---")
    history = model.fit(
        train_images, train_labels,
        epochs=epochs, batch_size=batch_size,
        validation_split=0.1, verbose=2,
    )
    return history


baseline = build_baseline()
baseline.summary()
hist_base = compile_and_fit(baseline, "baseline (1 hidden layer)")

deeper = build_deeper()
deeper.summary()
hist_deep = compile_and_fit(deeper, "deeper (2 hidden layers)")


# --------------------------------------------------------------------
# 3. Evaluate on the held-out test set
# --------------------------------------------------------------------
print("\n=== Test-set evaluation ===")
base_loss, base_acc = baseline.evaluate(test_images, test_labels, verbose=0)
deep_loss, deep_acc = deeper.evaluate(test_images, test_labels, verbose=0)
print(f"Baseline:  loss={base_loss:.4f}  accuracy={base_acc*100:.2f}%")
print(f"Deeper:    loss={deep_loss:.4f}  accuracy={deep_acc*100:.2f}%")


# --------------------------------------------------------------------
# 4. Plot training curves and a sample of predictions
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(hist_base.history["accuracy"], label="baseline train")
ax.plot(hist_base.history["val_accuracy"], label="baseline val", linestyle="--")
ax.plot(hist_deep.history["accuracy"], label="deeper train")
ax.plot(hist_deep.history["val_accuracy"], label="deeper val", linestyle="--")
ax.set_title("Training vs validation accuracy")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(hist_base.history["loss"], label="baseline train")
ax.plot(hist_base.history["val_loss"], label="baseline val", linestyle="--")
ax.plot(hist_deep.history["loss"], label="deeper train")
ax.plot(hist_deep.history["val_loss"], label="deeper val", linestyle="--")
ax.set_title("Training vs validation loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# Sample predictions from the deeper model
preds = deeper.predict(test_images[:25], verbose=0)
pred_labels = preds.argmax(axis=1)

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i], cmap="gray")
    true_name = CLASS_NAMES[test_labels[i]]
    pred_name = CLASS_NAMES[pred_labels[i]]
    color = "green" if pred_labels[i] == test_labels[i] else "red"
    ax.set_title(f"P: {pred_name}\nT: {true_name}", fontsize=8, color=color)
    ax.axis("off")
plt.suptitle("Deeper FNN - sample predictions (green=correct, red=wrong)",
             fontsize=12)
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------------
# Reflection
# --------------------------------------------------------------------
# - The baseline (one hidden layer of 128 ReLU units) lands around 88% test
#   accuracy after 10 epochs - within the 85-90% range the activity calls
#   out as expected for Fashion MNIST.
# - Adding a second hidden layer of 64 units does not move test accuracy
#   meaningfully on this dataset. Fashion MNIST is small enough (60k 28x28
#   images) that a single 128-unit hidden layer already has more than
#   enough capacity; the bottleneck is the lack of spatial structure in a
#   plain FNN, not depth.
# - For a real boost you would switch to a CNN, which exploits the 2D
#   pixel grid via convolutional kernels and pooling. That is the natural
#   next step after this activity, and is the architecture from the
#   CIFAR-10 notebook elsewhere in this repo.

print("\nSaved plots: training_curves.png, sample_predictions.png")
