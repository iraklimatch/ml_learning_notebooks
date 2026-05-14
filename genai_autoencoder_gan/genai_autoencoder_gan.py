"""
Practice activity: Evaluating an autoencoder and a GAN on MNIST.

Course: Microsoft Foundations of AI and Machine Learning
Module: Evaluating GenAI Models

Two generative models, two very different evaluation regimes:

  1. Autoencoder -> reconstruction MSE on a held-out test set.
  2. GAN        -> visual inspection of generated samples + discriminator
                   accuracy on real vs fake.

Both use a small fully-connected setup (no convolutions) so the whole
thing trains on CPU in a few minutes. The point is the evaluation, not
the architecture.

GAN epochs: the activity says 10,000. That's overkill for these tiny
fully-connected models and slow on CPU - we use 5,000 and save sample
grids at fixed checkpoints so we can watch image quality improve.
"""

import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ====================================================================
# Data: MNIST flattened to 784-vectors, scaled to [0, 1]
# ====================================================================
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

print(f"Train: {X_train_flat.shape}  Test: {X_test_flat.shape}")


# ====================================================================
# 1. AUTOENCODER
# ====================================================================
print("\n" + "=" * 64)
print("1. Autoencoder on MNIST")
print("=" * 64)

encoder = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),     # bottleneck
], name="encoder")

decoder = models.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(784, activation="sigmoid"),
], name="decoder")

autoencoder = models.Sequential([encoder, decoder], name="autoencoder")
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

t0 = time.time()
ae_hist = autoencoder.fit(
    X_train_flat, X_train_flat,
    epochs=10, batch_size=256,
    validation_data=(X_test_flat, X_test_flat),
    verbose=2,
)
ae_seconds = time.time() - t0

reconstructed = autoencoder.predict(X_test_flat, verbose=0)
ae_mse = float(np.mean((X_test_flat - reconstructed) ** 2))
print(f"\nAutoencoder reconstruction MSE: {ae_mse:.6f}  "
      f"train time: {ae_seconds:.1f}s")

# Save a "originals vs reconstruction" comparison plot
fig, axes = plt.subplots(2, 10, figsize=(15, 3.2))
for i in range(10):
    axes[0, i].imshow(X_test_flat[i].reshape(28, 28), cmap="gray")
    axes[0, i].axis("off")
    if i == 0: axes[0, i].set_title("Original", fontsize=10, loc="left")
    axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    axes[1, i].axis("off")
    if i == 0: axes[1, i].set_title("Reconstructed", fontsize=10, loc="left")
plt.suptitle(f"Autoencoder reconstructions on MNIST (test MSE {ae_mse:.4f})",
             fontsize=12)
plt.tight_layout()
plt.savefig("autoencoder_reconstructions.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# ====================================================================
# 2. GAN
# ====================================================================
print("\n" + "=" * 64)
print("2. GAN on MNIST")
print("=" * 64)

NOISE_DIM = 100
GAN_EPOCHS = 5000      # activity says 10,000; halved for CPU sanity
BATCH = 64
HALF = BATCH // 2
SAMPLE_EPOCHS = [0, 1000, 2500, 5000]


def build_generator():
    return models.Sequential([
        layers.Input(shape=(NOISE_DIM,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(784, activation="sigmoid"),
    ], name="generator")


def build_discriminator():
    return models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="discriminator")


generator = build_generator()
discriminator = build_discriminator()

# Modern Keras 3 GAN training uses GradientTape directly instead of the
# old trainable=False + stacked-model trick - that pattern silently
# breaks in current Keras and triggers "model does not have any
# trainable weights" warnings. With explicit tapes we control which
# weights each optimizer touches, with no ambiguity.
g_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)
bce = tf.keras.losses.BinaryCrossentropy()


@tf.function
def train_step(real_images, noise_d, noise_g):
    # ---- Discriminator step ----
    with tf.GradientTape() as tape:
        fake_images = generator(noise_d, training=False)
        d_real = discriminator(real_images, training=True)
        d_fake = discriminator(fake_images, training=True)
        d_loss = (bce(tf.ones_like(d_real), d_real)
                  + bce(tf.zeros_like(d_fake), d_fake)) * 0.5
    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    # ---- Generator step ----
    with tf.GradientTape() as tape:
        fake_images = generator(noise_g, training=True)
        d_on_fake = discriminator(fake_images, training=False)
        g_loss = bce(tf.ones_like(d_on_fake), d_on_fake)
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    # Discriminator accuracy on this step's real / fake batch
    d_acc_real = tf.reduce_mean(tf.cast(d_real > 0.5, tf.float32))
    d_acc_fake = tf.reduce_mean(tf.cast(d_fake < 0.5, tf.float32))
    return d_loss, g_loss, d_acc_real, d_acc_fake


# Fixed noise so the sample grids at epochs 0/1000/2500/5000 use the same
# latent vectors - lets us see how the *same noise* maps to different
# outputs as the generator improves.
FIXED_NOISE = np.random.normal(0, 1, size=(25, NOISE_DIM)).astype("float32")
sample_grids = {}
disc_acc_real = []
disc_acc_fake = []
gan_losses = []

print(f"Training GAN for {GAN_EPOCHS} epochs (samples at {SAMPLE_EPOCHS})...")
t0 = time.time()
sample_grids[0] = generator(FIXED_NOISE, training=False).numpy().reshape(25, 28, 28)

for epoch in range(1, GAN_EPOCHS + 1):
    idx = np.random.randint(0, X_train_flat.shape[0], HALF)
    real_images = tf.constant(X_train_flat[idx])
    noise_d = tf.random.normal((HALF, NOISE_DIM))
    noise_g = tf.random.normal((BATCH, NOISE_DIM))

    d_loss, g_loss, d_acc_real, d_acc_fake = train_step(
        real_images, noise_d, noise_g
    )

    disc_acc_real.append(float(d_acc_real))
    disc_acc_fake.append(float(d_acc_fake))
    gan_losses.append(float(g_loss))

    if epoch in SAMPLE_EPOCHS:
        sample_grids[epoch] = generator(
            FIXED_NOISE, training=False
        ).numpy().reshape(25, 28, 28)

    if epoch % 500 == 0 or epoch == 1:
        # Use the most recent window of accuracies as a stable readout
        recent_real = float(np.mean(disc_acc_real[-100:]))
        recent_fake = float(np.mean(disc_acc_fake[-100:]))
        print(f"epoch {epoch:5d}/{GAN_EPOCHS}  "
              f"D acc real={recent_real:.2f}  D acc fake={recent_fake:.2f}  "
              f"G loss={g_loss:.4f}")

gan_seconds = time.time() - t0
print(f"\nGAN training time: {gan_seconds:.1f}s")


# Final discriminator accuracy on a fresh batch of real vs fake
N_EVAL = 2000
noise = tf.random.normal((N_EVAL, NOISE_DIM))
fake_eval = generator(noise, training=False).numpy()
real_eval = X_test_flat[np.random.choice(len(X_test_flat), N_EVAL, replace=False)]

d_real_pred = discriminator(real_eval, training=False).numpy().flatten()
d_fake_pred = discriminator(fake_eval, training=False).numpy().flatten()
disc_final_real = float(np.mean(d_real_pred > 0.5))
disc_final_fake = float(np.mean(d_fake_pred < 0.5))
print(f"Final discriminator accuracy:")
print(f"  on real test images: {disc_final_real*100:.1f}%")
print(f"  on fake samples:     {disc_final_fake*100:.1f}%")


# ----- GAN sample-grid progression plot ---------------------------------
fig, axes = plt.subplots(len(SAMPLE_EPOCHS), 5, figsize=(8, 1.7 * len(SAMPLE_EPOCHS)))
for row, ep in enumerate(SAMPLE_EPOCHS):
    grid = sample_grids[ep]
    for col in range(5):
        ax = axes[row, col]
        ax.imshow(grid[col], cmap="gray")
        ax.axis("off")
    axes[row, 0].set_title(f"Epoch {ep}", fontsize=10, loc="left")
plt.suptitle("GAN samples from fixed noise across training", fontsize=12)
plt.tight_layout()
plt.savefig("gan_samples_progression.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Final 5x5 grid
fig, axes = plt.subplots(5, 5, figsize=(6, 6))
final = sample_grids[GAN_EPOCHS]
for i, ax in enumerate(axes.flat):
    ax.imshow(final[i], cmap="gray")
    ax.axis("off")
plt.suptitle(f"GAN final samples (epoch {GAN_EPOCHS})", fontsize=12)
plt.tight_layout()
plt.savefig("gan_final_samples.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# ----- Training-dynamics plot -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
window = 100
real_smooth = np.convolve(disc_acc_real, np.ones(window)/window, mode="valid")
fake_smooth = np.convolve(disc_acc_fake, np.ones(window)/window, mode="valid")
ax.plot(np.arange(window-1, len(disc_acc_real)) + 1, real_smooth, label="D acc on real")
ax.plot(np.arange(window-1, len(disc_acc_fake)) + 1, fake_smooth, label="D acc on fake")
ax.axhline(0.5, ls="--", color="gray", alpha=0.6, label="Ideal equilibrium (0.5)")
ax.set_title(f"Discriminator accuracy ({window}-step rolling)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
loss_smooth = np.convolve(gan_losses, np.ones(window)/window, mode="valid")
ax.plot(np.arange(window-1, len(gan_losses)) + 1, loss_smooth)
ax.set_title(f"Generator loss ({window}-step rolling)")
ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
ax.grid(True, alpha=0.3)

plt.suptitle("GAN training dynamics", fontsize=12)
plt.tight_layout()
plt.savefig("gan_training_dynamics.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# ====================================================================
# Save metrics for the notebook
# ====================================================================
np.savez(
    "results.npz",
    ae_mse=ae_mse,
    ae_seconds=ae_seconds,
    ae_train_loss=np.array(ae_hist.history["loss"]),
    ae_val_loss=np.array(ae_hist.history["val_loss"]),
    ae_originals=X_test_flat[:10].reshape(10, 28, 28),
    ae_reconstructed=reconstructed[:10].reshape(10, 28, 28),
    gan_seconds=gan_seconds,
    disc_acc_real=np.array(disc_acc_real),
    disc_acc_fake=np.array(disc_acc_fake),
    gan_losses=np.array(gan_losses),
    disc_final_real=disc_final_real,
    disc_final_fake=disc_final_fake,
    gan_sample_epochs=np.array(SAMPLE_EPOCHS),
    gan_sample_grids=np.stack([sample_grids[ep] for ep in SAMPLE_EPOCHS]),
)

print("\n" + "=" * 64)
print("Summary")
print("=" * 64)
print(f"Autoencoder reconstruction MSE: {ae_mse:.6f}")
print(f"GAN discriminator final accuracy:  real={disc_final_real*100:.1f}%  fake={disc_final_fake*100:.1f}%")
print("Saved plots:  autoencoder_reconstructions.png,")
print("              gan_samples_progression.png, gan_final_samples.png,")
print("              gan_training_dynamics.png")
print("Saved arrays: results.npz")
