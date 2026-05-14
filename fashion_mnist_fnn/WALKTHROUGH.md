# Fashion MNIST FNN - Walkthrough

Written reflection for the Microsoft *Foundations of AI and Machine Learning* solution walkthrough. The implementation lives in [`fashion_mnist_fnn.py`](fashion_mnist_fnn.py) and the executed notebook is in [`fashion_mnist_fnn.ipynb`](fashion_mnist_fnn.ipynb) - this file captures the *what* and *why* in prose.

## What I built

A small feedforward neural network in TensorFlow/Keras, trained on Fashion MNIST (60k training, 10k test, 28x28 grayscale, 10 clothing classes). I trained two variants back-to-back so the architecture comparison is in the same script:

1. **Baseline** - `Flatten -> Dense(128, relu) -> Dense(10, softmax)`
2. **Deeper** - `Flatten -> Dense(128, relu) -> Dense(64, relu) -> Dense(10, softmax)`

Both compiled with `optimizer='adam'`, `loss='sparse_categorical_crossentropy'`, `metrics=['accuracy']`, trained for 10 epochs at batch size 32, with a 10% validation split.

## Step-by-step (matching the walkthrough)

### 1. Load and normalize

```python
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.fashion_mnist.load_data()
)
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
```

Note: the walkthrough's printed snippet has a bug - it sets `test_images = train_images / 255.0`, which would overwrite the test set with the training set. The fix is to normalize `test_images` from itself. Catching this is the whole reason you read your own preprocessing code before kicking off a long training job.

### 2. Architecture

`Flatten` reshapes the 28x28 image into a 784-vector. The first `Dense(128, relu)` gives the network a non-linear projection - ReLU is the modern default because it avoids the vanishing-gradient problem that sigmoid/tanh hit in deeper stacks. The final `Dense(10, softmax)` produces a probability over the 10 classes.

### 3. Compile + 4. Train

`adam` adapts the per-parameter learning rate on the fly, which makes it forgiving when you don't tune anything else. `sparse_categorical_crossentropy` lets us pass integer labels (`0..9`) instead of one-hot vectors - convenient and exactly equivalent to `categorical_crossentropy` on one-hot targets.

### 5. Evaluate

```
Baseline:  loss=0.3897  accuracy=87.22%
Deeper:    loss=0.4271  accuracy=86.41%
```

Both inside the 85-90% range the walkthrough calls out as expected. The training curves are in [`training_curves.png`](training_curves.png) - both models start to overfit around epoch 4 (training accuracy keeps climbing toward 91%+ while validation plateaus around 87-88%).

## Experimentation notes

The walkthrough invites four variants to try. My take on each:

- **Add a hidden layer (already done above)** - no meaningful gain. Fashion MNIST is small enough that a single 128-unit layer has enough capacity; the bottleneck is the flattening, not depth.
- **Change neuron count** - dropping the hidden layer to 32 units would underfit; bumping it to 512 mostly just trains slower and overfits more aggressively (training accuracy higher, validation accuracy same).
- **Different activation** - `tanh` would land in roughly the same place but train marginally slower; `sigmoid` in a hidden layer is generally a bad idea because of vanishing gradients, and you'd see slower convergence.
- **SGD vs Adam** - vanilla `SGD` at the same learning rate would converge much more slowly. With `SGD(momentum=0.9)` you can get close to Adam but you have to tune the learning rate by hand.

The honest answer to "how do you push past ~88% on Fashion MNIST" is **switch to a CNN**. A plain FNN throws away the 2D pixel grid the moment it flattens, so it has to relearn spatial structure feature-by-feature. The CIFAR-10 notebook elsewhere in this repo (`../cifar10_tensorflow.ipynb`) uses the same TensorFlow/Keras API but with convolutions and pooling, and is the natural next step.

## Results summary

| Model | Hidden layers | Test accuracy | Test loss |
|---|---|---|---|
| Baseline | `Dense(128, relu)` | 87.22% | 0.3897 |
| Deeper | `Dense(128, relu) -> Dense(64, relu)` | 86.41% | 0.4271 |

Within the expected 85-90% range. The deeper model is slightly worse on test - extra parameters without more data or stronger regularization just gives the network more room to memorize the training set.
