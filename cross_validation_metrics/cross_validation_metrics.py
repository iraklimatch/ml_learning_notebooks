"""
Practice activity: Evaluation metrics & cross-validation.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    make_scorer,
)


# 3. Load and prepare the data
data = {
    'StudyHours':    [1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass':          [0,  0,  0,  0,  0,  1,  1,  1,  1,  1],
}
df = pd.DataFrame(data)
print(df)

X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']


# 4. Evaluation metrics on a single train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print("\n--- Single train/test split ---")
print(f"Accuracy:  {accuracy}")
print(f"Precision: {precision}")
print(f"Recall:    {recall}")
print(f"F1-Score:  {f1}")


# 6. K-fold cross-validation (accuracy only)
# With only 10 rows we cap k at 5 so each fold still has 2 samples.
model = LogisticRegression()
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("\n--- 5-fold cross-validation (accuracy) ---")
print(f"Per-fold accuracies: {cv_scores}")
print(f"Mean accuracy:       {cv_scores.mean():.4f}")


# 7. Cross-validation with multiple metrics
# zero_division=0 keeps precision/recall/f1 defined on small/imbalanced folds.
scoring = {
    'accuracy':  'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall':    make_scorer(recall_score,    zero_division=0),
    'f1':        make_scorer(f1_score,        zero_division=0),
}
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

print("\n--- 5-fold cross-validation (multiple metrics) ---")
for metric in scoring:
    key = f"test_{metric}"
    print(f"{metric.capitalize():10s} per-fold: {cv_results[key]}  "
          f"mean: {cv_results[key].mean():.4f}")


# 8. Cross-validation with a regression model
X_reg = df[['StudyHours']]
y_reg = df['PrevExamScore']
reg_model = LinearRegression()

# Scikit-learn returns negative MSE/MAE so higher = better across all scorers.
reg_scoring = {
    'r2':           'r2',
    'neg_mse':      'neg_mean_squared_error',
    'neg_mae':      'neg_mean_absolute_error',
}
reg_results = cross_validate(reg_model, X_reg, y_reg, cv=5, scoring=reg_scoring)

print("\n--- 5-fold cross-validation (regression) ---")
print(f"R-squared per-fold: {reg_results['test_r2']}  mean: {reg_results['test_r2'].mean():.4f}")
print(f"MSE per-fold:       {-reg_results['test_neg_mse']}  mean: {-reg_results['test_neg_mse'].mean():.4f}")
print(f"MAE per-fold:       {-reg_results['test_neg_mae']}  mean: {-reg_results['test_neg_mae'].mean():.4f}")


# Key takeaways
#
# - A single 80/20 split gives one accuracy number that depends entirely on
#   which 2 rows land in the test set. With 10 rows, that's high variance.
# - K-fold rotates the test fold k times and averages, so the score reflects
#   the model rather than a lucky split.
# - For classification, average accuracy/precision/recall/F1 across folds.
#   For regression, use R-squared plus MAE/MSE (remember sklearn returns the
#   *negative* of MSE/MAE so that "higher is better" is consistent).
# - R-squared can be negative on tiny per-fold test sets when variance in the
#   held-out fold is small — that's expected on a 10-row dataset, not a bug.
