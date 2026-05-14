"""
Practice activity: CIFAR-10 CNN in TensorFlow and PyTorch, side-by-side.

Course: Microsoft Foundations of AI and Machine Learning
Module: Frameworks and Tools

Same architecture in both frameworks:
    Conv2d(3   ->32, 3x3, ReLU)  -> MaxPool(2)
    Conv2d(32  ->64, 3x3, ReLU)  -> MaxPool(2)
    Flatten
    Linear(64*6*6 -> 64, ReLU)
    Linear(64 -> 10)               (softmax in TF, raw logits in PyTorch)

Both trained for 10 epochs, batch size 32, Adam (lr=1e-3).
The whole point is to see the same model end up at roughly the same place
through two very different training APIs.
"""

import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt

# TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 10
BATCH = 32
LR = 1e-3

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


# =====================================================================
# 1. TensorFlow / Keras side
# =====================================================================
print("=" * 64)
print("1. TensorFlow (Keras) implementation")
print("=" * 64)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = y_train.squeeze()
y_test = y_test.squeeze()

print(f"TF train: {x_train.shape}, test: {x_test.shape}")

tf_model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
tf_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
tf_model.summary()

t0 = time.time()
tf_hist = tf_model.fit(
    x_train, y_train,
    epochs=EPOCHS, batch_size=BATCH,
    validation_data=(x_test, y_test),
    verbose=2,
)
tf_train_seconds = time.time() - t0
tf_test_loss, tf_test_acc = tf_model.evaluate(x_test, y_test, verbose=0)
print(f"\nTF test accuracy: {tf_test_acc*100:.2f}%  (loss {tf_test_loss:.4f})")
print(f"TF training time: {tf_train_seconds:.1f}s")


# =====================================================================
# 2. PyTorch side
# =====================================================================
print("\n" + "=" * 64)
print("2. PyTorch implementation")
print("=" * 64)

# Activity uses Normalize((0.5,)*3, (0.5,)*3) which maps to [-1, 1]. We keep
# that so the PyTorch experiment matches the activity verbatim - and so the
# preprocessing difference (TF: [0,1], PyTorch: [-1,1]) is visible in the
# comparison.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=True, download=True, transform=transform,
)
testset = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=False, download=True, transform=transform,
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True,  num_workers=0)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=BATCH, shuffle=False, num_workers=0)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32, 3)     # 32x32x3 -> 30x30x32
        self.pool  = nn.MaxPool2d(2, 2)        # -> 15x15x32
        self.conv2 = nn.Conv2d(32, 64, 3)      # -> 13x13x64 -> pool -> 6x6x64
        self.fc1   = nn.Linear(64 * 6 * 6, 64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


pt_model = SimpleCNN()
n_params = sum(p.numel() for p in pt_model.parameters())
print(f"PyTorch model parameters: {n_params:,}")

optimizer = optim.Adam(pt_model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


def evaluate_pt(model, loader):
    model.eval()
    total_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return total_loss / total, correct / total


pt_hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
t0 = time.time()
for epoch in range(EPOCHS):
    pt_model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for xb, yb in trainloader:
        optimizer.zero_grad()
        logits = pt_model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * yb.size(0)
        running_correct += (logits.argmax(1) == yb).sum().item()
        running_total += yb.size(0)

    train_loss = running_loss / running_total
    train_acc = running_correct / running_total
    val_loss, val_acc = evaluate_pt(pt_model, testloader)
    pt_hist["train_loss"].append(train_loss)
    pt_hist["train_acc"].append(train_acc)
    pt_hist["val_loss"].append(val_loss)
    pt_hist["val_acc"].append(val_acc)
    print(f"Epoch {epoch+1:2d}/{EPOCHS}  "
          f"train_loss={train_loss:.4f} train_acc={train_acc*100:5.2f}%  "
          f"val_loss={val_loss:.4f} val_acc={val_acc*100:5.2f}%")

pt_train_seconds = time.time() - t0
pt_test_loss, pt_test_acc = evaluate_pt(pt_model, testloader)
print(f"\nPyTorch test accuracy: {pt_test_acc*100:.2f}%  (loss {pt_test_loss:.4f})")
print(f"PyTorch training time: {pt_train_seconds:.1f}s")


# =====================================================================
# 3. Side-by-side summary
# =====================================================================
print("\n" + "=" * 64)
print("3. Comparison")
print("=" * 64)
tf_params = tf_model.count_params()
print(f"{'Framework':<10} {'Params':>10} {'Test acc':>10} {'Test loss':>10} {'Train time':>12}")
print(f"{'-'*54}")
print(f"{'TF/Keras':<10} {tf_params:>10,} {tf_test_acc*100:>9.2f}% {tf_test_loss:>10.4f} {tf_train_seconds:>11.1f}s")
print(f"{'PyTorch':<10} {n_params:>10,} {pt_test_acc*100:>9.2f}% {pt_test_loss:>10.4f} {pt_train_seconds:>11.1f}s")


# =====================================================================
# 4. Learning curves
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(tf_hist.history["accuracy"],     label="TF train")
ax.plot(tf_hist.history["val_accuracy"], label="TF val", linestyle="--")
ax.plot(pt_hist["train_acc"],            label="PyTorch train")
ax.plot(pt_hist["val_acc"],              label="PyTorch val", linestyle="--")
ax.set_title("Accuracy per epoch")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(tf_hist.history["loss"],     label="TF train")
ax.plot(tf_hist.history["val_loss"], label="TF val", linestyle="--")
ax.plot(pt_hist["train_loss"],       label="PyTorch train")
ax.plot(pt_hist["val_loss"],         label="PyTorch val", linestyle="--")
ax.set_title("Loss per epoch")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle("CIFAR-10 CNN - TensorFlow vs PyTorch", fontsize=13)
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=120, bbox_inches="tight")
plt.close(fig)

print("\nSaved plot: learning_curves.png")


# Save a tiny pickle of the histories so the notebook can plot without
# retraining.
np.savez(
    "histories.npz",
    tf_train_acc=np.array(tf_hist.history["accuracy"]),
    tf_val_acc=np.array(tf_hist.history["val_accuracy"]),
    tf_train_loss=np.array(tf_hist.history["loss"]),
    tf_val_loss=np.array(tf_hist.history["val_loss"]),
    pt_train_acc=np.array(pt_hist["train_acc"]),
    pt_val_acc=np.array(pt_hist["val_acc"]),
    pt_train_loss=np.array(pt_hist["train_loss"]),
    pt_val_loss=np.array(pt_hist["val_loss"]),
    tf_test_acc=tf_test_acc, tf_test_loss=tf_test_loss,
    pt_test_acc=pt_test_acc, pt_test_loss=pt_test_loss,
    tf_train_seconds=tf_train_seconds, pt_train_seconds=pt_train_seconds,
)
print("Saved: histories.npz")
