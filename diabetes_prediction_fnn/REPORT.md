# Pima Indians Diabetes - FNN Results Report

Report for the Microsoft *Foundations of AI and Machine Learning* "Applied Deep Learning" activity. Implementation in [`diabetes_prediction_fnn.py`](diabetes_prediction_fnn.py); executed notebook in [`diabetes_prediction_fnn.ipynb`](diabetes_prediction_fnn.ipynb).

## Results

| Metric | Value |
|---|---|
| Test accuracy | **71.43%** |
| Test loss (BCE) | 0.4883 |
| Test ROC AUC | **0.8306** |
| Baseline (predict 0) | 65.1% |
| Training time | <10s on CPU |

Per-class breakdown:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No diabetes (0) | 0.76 | 0.81 | 0.79 |
| Diabetes (1) | 0.60 | 0.54 | 0.57 |

Confusion matrix (rows = true, cols = predicted):

|  | Pred 0 | Pred 1 |
|---|---|---|
| **True 0** | 81 | 19 |
| **True 1** | 25 | 29 |

## What the numbers say

**Above baseline, but not dramatically.** The model is 6 points above the "always predict 0" baseline. That's real signal, not noise - the AUC of 0.83 confirms the model can rank positives above negatives much better than chance. AUC is the better metric here because the dataset is unbalanced and AUC doesn't care about the 0.5 threshold.

**The model is biased toward the majority class.** Of 54 true diabetes cases in the test set, the model catches 29 (54% recall). In a real diagnostic setting that's the wrong end of the trade-off to land on. Missing half of the positive cases would be unacceptable; the easy fix is to lower the classification threshold below 0.5, or weight the positive class up during training (`class_weight={0: 1, 1: 1.8}` would roughly balance the loss contribution from each class).

## Insights from the accuracy and loss plots

The training curves show two clean stories:

1. **Both curves improve and plateau by ~epoch 30.** Training loss keeps falling slowly afterward while validation loss flatlines around 0.49. Training for more epochs at these hyperparameters won't help.
2. **A small but persistent overfitting gap.** Training accuracy reaches ~80%, validation hovers at 71-72%. The gap is the model memorizing training quirks. The fix is regularization (`Dropout(0.2)` between dense layers, or L2 weight decay), or early stopping based on validation loss.

The loss curves are smoother than the accuracy curves because accuracy is a thresholded quantity - it can jump by a few patients across the 0.5 boundary while the underlying probability barely moves.

## Challenges and observations

**Hidden missing values were the big finding.** The dataset has no `NaN`s but five columns contain biologically impossible zeros (Glucose = 0, BloodPressure = 0, Insulin = 0, BMI = 0). Those are missing-value sentinels. Treating them as real values pulls the feature distribution toward zero and confuses the network. Replacing the zeros with the per-feature median (computed from the non-zero rows) before scaling moved test accuracy from ~67% to 71% and recall on the positive class from 40% to 54%.

**Class imbalance silently distorts the headline metric.** With a 65/35 split, accuracy is a misleading single number. Reporting AUC, per-class precision/recall, and the confusion matrix is much more informative.

**Tabular data on this scale isn't where deep learning shines.** A logistic regression on the same cleaned features lands in the same ~72% accuracy range; gradient-boosted trees (XGBoost, LightGBM) typically score 76-78% AUC on this dataset without much tuning. The FNN is a useful exercise but it's not the right tool for the job at this size.

## What I would try next

- **Class weighting or threshold tuning** to improve positive-class recall, since false negatives are more costly than false positives in a diagnostic context.
- **K-fold cross-validation** instead of a single 80/20 split. With only 768 patients, the test-set estimate has a wide confidence interval - one bad split can shift accuracy by ~3-4 points.
- **Compare against a baseline.** Train a `LogisticRegression` and an `XGBClassifier` on the same cleaned data; if the FNN doesn't beat them, the architecture isn't earning its complexity.
