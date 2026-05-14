# CIFAR-10 CNN: TensorFlow vs PyTorch

Comparison report for the Microsoft *Foundations of AI and Machine Learning* "Frameworks and Tools" activity. The implementation is in [`cifar10_tf_vs_pytorch.py`](cifar10_tf_vs_pytorch.py); the executed notebook is in [`cifar10_tf_vs_pytorch.ipynb`](cifar10_tf_vs_pytorch.ipynb).

## What was compared

The exact same architecture in both frameworks:

| Layer | Output shape | Parameters |
|---|---|---|
| `Conv2d(3 -> 32, 3x3, ReLU)` + `MaxPool(2)` | 15x15x32 | 896 |
| `Conv2d(32 -> 64, 3x3, ReLU)` + `MaxPool(2)` | 6x6x64 | 18,496 |
| `Flatten` | 2304 | - |
| `Dense(64, ReLU)` | 64 | 147,520 |
| `Dense(10, softmax/logits)` | 10 | 650 |
| **Total** | | **167,562** |

Training: 10 epochs, batch size 32, Adam (lr=1e-3), CrossEntropy loss. Run on CPU.

## Results

| Framework | Test accuracy | Test loss | Training time | Final train acc |
|---|---|---|---|---|
| TensorFlow / Keras | 67.56% | 1.1020 | 144.4s | 79.86% |
| PyTorch | 70.32% | 0.9599 | 573.8s | 82.83% |

Both inside the expected ~65-70% range for this small architecture - CIFAR-10 needs more depth, regularization, or data augmentation to push past that. The two frameworks land in the same neighborhood, as they should; the deltas come from preprocessing and runtime, not the model.

## What was different

**Preprocessing range.** TensorFlow used `images / 255.0` so pixels were in `[0, 1]`. PyTorch used the activity's `Normalize((0.5,)*3, (0.5,)*3)` so pixels were in `[-1, 1]`. The `[-1, 1]` input has zero mean, which is the textbook reason it tends to train slightly faster and more stably than `[0, 1]` - that probably explains some of the PyTorch accuracy edge.

**Training speed.** TensorFlow finished in 144s; PyTorch took 574s (4x slower) on the same architecture, same machine, CPU only. This is the `tf.function` graph-tracing payoff - Keras compiles the training step into an optimized graph, while PyTorch's eager-mode loop runs each op individually in Python. With CUDA the gap would mostly close; without it, TF/Keras eats PyTorch for breakfast on CPU.

**Lines of code.** TensorFlow training is literally three lines:

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)
```

PyTorch makes you write the whole loop:

```python
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in trainloader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
    # ... and the val loop yourself
```

Twice as much code for the same training. The flip side is that every step is visible; nothing is hidden behind `.fit()`.

## Ease of use

**TensorFlow / Keras wins on the standard path.** `Sequential` + `compile` + `fit` is two minutes of work and you have a baseline. The model summary is one method call. Validation, metrics, callbacks, model saving - all wired in by default.

**PyTorch wins when the model is unusual.** A custom forward pass, non-standard losses, partial training freezes - all of these are easier when you already own the loop. There is no `.fit()` to fight with.

For this activity the TF version was meaningfully easier to write. For research code I would reach for PyTorch every time.

## Debugging

**PyTorch's dynamic graph is genuinely better for debugging.** `print(x.shape)` inside `forward()` just works. You can drop a `breakpoint()` mid-forward-pass and inspect intermediate tensors. When something silently NaNs, you can step through the offending batch.

**TensorFlow's `tf.function` is opaque.** Inside a compiled graph, prints don't trigger and Python breakpoints don't bind. You can disable it with `model.run_eagerly = True`, but you then lose the speed advantage. For production this trade-off is fine; for hunting bugs it's painful.

The shape-mismatch error message I hit first was `64 * 6 * 6` for `fc1`. In PyTorch that errors at the `Linear` layer's first forward call - clear, you fix it. In Keras the model would just fail at fit time with a less direct trace.

## Which framework I preferred

For this assignment: **TensorFlow / Keras**. Smaller surface area, less boilerplate, faster on CPU. I trained the baseline, saw the result, and moved on in roughly half the wall-clock time the PyTorch version took.

In a real project I'd let the actual constraints decide. If the team already lives in the TF or PyTorch ecosystem, that's the answer. If neither, I'd pick TF/Keras for standard supervised tasks where the architecture is well-understood, and PyTorch the moment I expected to read intermediate gradients or write custom layers.

## Caveats

- 10 epochs is too few to fully separate the frameworks - both are still improving on training accuracy when they stop. With 30+ epochs you would expect both to overfit similarly and end up closer in test accuracy.
- CPU-only timing is brutal on PyTorch's eager mode. On a GPU the speed gap roughly closes.
- The accuracy gap is partly real (PyTorch's `[-1, 1]` preprocessing helps) and partly noise from initialization. Re-running with different seeds would smear the numbers by a few percentage points either way.
