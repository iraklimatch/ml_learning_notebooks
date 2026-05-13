"""
Practice activity: LASSO regression for feature selection.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques
"""

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


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


# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. Apply LASSO with a single alpha
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("\n--- LASSO with alpha=0.1 ---")
print(f"R-squared: {r2:.4f}")


# 6. Inspect coefficients
print(f"LASSO Coefficients: {lasso_model.coef_}")
for name, coef in zip(X.columns, lasso_model.coef_):
    status = "kept" if coef != 0 else "dropped (shrunk to 0)"
    print(f"  {name:14s} coef={coef:+.6f}  [{status}]")


# 7. Tune alpha
print("\n--- Alpha sweep ---")
print(f"{'alpha':>8s}  {'R-squared':>12s}  coefficients")
for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
    m = Lasso(alpha=alpha)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    print(f"{alpha:>8.2f}  {r2_score(y_test, pred):>12.4f}  {m.coef_}")


# Key takeaways
#
# - LASSO adds an L1 penalty: the model balances squared error against the
#   sum of |coefficients|. Strong penalties (high alpha) push small
#   coefficients all the way to zero — that's the feature-selection effect.
# - At alpha=1.0 on this dataset both coefficients hit 0 and R-squared is
#   undefined / negative because the model just predicts the mean. That's
#   exactly when you've over-regularized.
# - StudyHours and PrevExamScore are perfectly collinear here (they grow
#   together), so LASSO tends to keep one and drop the other — that's the
#   normal behavior, not a bug.
