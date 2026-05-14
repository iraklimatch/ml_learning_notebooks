# Evaluating GenAI Models: Autoencoder vs GAN on MNIST

Report for the Microsoft *Foundations of AI and Machine Learning* "Evaluating GenAI Models" activity. Implementation in [`genai_autoencoder_gan.py`](genai_autoencoder_gan.py); executed notebook in [`genai_autoencoder_gan.ipynb`](genai_autoencoder_gan.ipynb).

## Results

| Model | Architecture | Headline metric | Training time |
|---|---|---|---|
| Autoencoder | `784 -> 128 -> 64 -> 128 -> 784` (sigmoid out) | **Reconstruction MSE 0.0076** on test | ~20s |
| GAN | G: `100 -> 128 -> 784`; D: `784 -> 128 -> 1` | D acc real **91.5%**, D acc fake **93.0%** | ~29s |

## Performance of each model

**Autoencoder reconstruction quality.** Test-set MSE of 0.0076 on pixels in `[0, 1]` translates to root-MSE ~0.087, or about ~9% average pixel error. Visually, the reconstructions in `autoencoder_reconstructions.png` preserve the digit identity perfectly - every digit is the same digit after the round-trip - but lose fine pen strokes and edge detail. That's the expected behavior of a 64-d bottleneck compressing 784-d images: a 12x reduction in dimensionality forces the encoder to keep the high-energy structure (digit identity, stroke topology) and throw away the high-frequency noise (exact pixel intensities at the edges). For a tutorial autoencoder this is a clean result.

**GAN sample quality.** Visual inspection of `gan_samples_progression.png` shows the generator learning real structure: at epoch 0 the samples are pure uniform noise; at epoch 1000 they're vaguely digit-shaped blobs; by epoch 5000 several samples have visible digit topology (loops, ascenders, descenders), though they're still blurry and somewhat noisy. None would fool a human, but they're clearly *trying* to be digits rather than random pixels.

## How the autoencoder's MSE reflects its reconstruction ability

The autoencoder has the easiest evaluation story in deep learning: there's a ground-truth target for every input - the input itself. MSE measures exactly the right thing. Lower MSE means the reconstructed pixels are closer to the originals; it correlates almost perfectly with subjective "sharpness" of the reconstruction. The bottleneck width controls the trade-off: a smaller bottleneck (32-d) would force higher MSE and blurrier outputs; a larger bottleneck (256-d) would make MSE smaller and reconstructions sharper, but compress less. At 64-d we get sub-1% mean squared error, which corresponds to the visibly faithful reconstructions in the plot.

What MSE doesn't capture: perceptual quality. Two reconstructions with the same MSE can look very different. A reconstruction that shifts every pixel by a constant might have higher MSE than one that blurs the digit but keeps the mean - even though the shifted version is more recognizable. For tutorial work MSE is enough; for production image work people often use SSIM or LPIPS.

## GAN sample quality and what the discriminator learned

The discriminator ended at 91.5% accuracy on real test images and 93.0% on freshly generated fakes. That's a long way from the healthy equilibrium of ~50% on each, where D can't tell them apart. The training-dynamics plot tells the rest of the story: D's accuracy on fake hits 100% within the first 500 epochs and stays near-ceiling for most of training, while the generator's BCE loss bounces around 2.0-3.5 - never settling, but slowly forcing improvements out of the generator.

**What this means.** The discriminator is winning the adversarial game. At this scale - small fully-connected networks, no normalization, no convolutional structure - the discriminator's classification task is much easier than the generator's synthesis task. The generator IS still learning (the sample progression proves that) but it can never quite catch up. Real production GANs solve this with deeper convolutional generators, batch normalization, label smoothing, and architectural tricks like DCGAN or progressive growth.

**Why visual inspection matters more than the accuracy number here.** A discriminator accuracy of 93% on fakes could mean two very different things: (a) the generator is producing rubbish that no reasonable classifier would accept, or (b) the generator is producing decent samples but the discriminator is just very good. The progression plot resolves the ambiguity - we can see structure emerging across epochs, so the generator is learning, even if the discriminator is still ahead.

## Comparison and conclusion

The two models illustrate completely different evaluation regimes. The **autoencoder** has a single scalar metric (MSE) that captures exactly what we want; you can rank two autoencoders just by looking at the number. The **GAN** has no such metric. Discriminator accuracy is a proxy (low is good for sample quality, but only if D is also a strong critic); visual inspection is the most honest signal but doesn't scale; production work uses Fréchet Inception Distance (FID) or Inception Score, neither of which is meaningful for a 5,000-epoch FC model on MNIST. This is why GAN papers spend pages defending whichever metric they picked.

For the activity's MNIST setup, the autoencoder is decisively the better-evaluated and better-performing model. The GAN learns real structure but doesn't produce convincing digits at this architecture and training budget. To make this GAN competitive you'd need to swap to a DCGAN-style convolutional generator and discriminator, add batch normalization, and either train much longer or use a larger learning-rate-scheduled training regime - none of which are activity-scope changes.

## Implementation note

The activity's GAN code uses the classic Keras pattern of compiling the discriminator, setting `discriminator.trainable = False`, then stacking it on top of the generator and compiling the combined model. In current Keras 3 that pattern triggers a "model does not have any trainable weights" warning and the generator's gradients silently stop flowing in some setups. The implementation here uses an explicit `tf.GradientTape` training step instead, which gives unambiguous control over which weights each optimizer updates and trains roughly an order of magnitude faster on CPU because it avoids the per-step `.predict()` calls.
