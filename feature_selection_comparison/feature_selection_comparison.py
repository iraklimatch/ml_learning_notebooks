"""
Practice activity: Comparing feature selection techniques.

Backward elimination (statsmodels p-values), forward selection
(greedy R-squared), and LASSO (L1 regularization) — applied to the
same dataset for comparison.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# 3. Load the dataset
data = {
    'StudyHours':    [1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass':          [0,  0,  0,  0,  0,  1,  1,  1,  1,  1],
}
df = pd.DataFrame(data)
print(df)

X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']


# 4. Backward elimination (statsmodels OLS, drop highest p-value > 0.05)
def backward_elimination(X, y, threshold=0.05):
    features = list(X.columns)
    while features:
        X_const = sm.add_constant(X[features], has_constant='add')
        model = sm.OLS(y, X_const).fit()
        # Ignore the intercept's p-value when looking for the worst feature.
        pvals = model.pvalues.drop('const', errors='ignore')
        worst_pval = pvals.max()
        if worst_pval > threshold:
            worst_feature = pvals.idxmax()
            print(f"  Drop {worst_feature!r:18s}  p-value = {worst_pval:.4f}")
            features.remove(worst_feature)
        else:
            break
    return features, model


print("\n--- Backward elimination ---")
print("Initial fit summary:")
initial_fit = sm.OLS(y, sm.add_constant(X)).fit()
print(initial_fit.summary().tables[1])

print("\nElimination rounds:")
be_features, be_model = backward_elimination(X, y)
print(f"\nSelected features (backward elimination): {be_features}")


# 5. Forward selection (greedy R-squared on held-out split)
def forward_selection(X, y):
    remaining = set(X.columns)
    selected = []
    best_score = 0.0

    while remaining:
        round_scores = []
        for feature in remaining:
            trial = selected + [feature]
            X_train, X_test, y_train, y_test = train_test_split(
                X[trial], y, test_size=0.2, random_state=42
            )
            m = LinearRegression().fit(X_train, y_train)
            round_scores.append((r2_score(y_test, m.predict(X_test)), feature))

        round_scores.sort(reverse=True)
        top_score, top_feature = round_scores[0]

        if top_score > best_score:
            selected.append(top_feature)
            remaining.remove(top_feature)
            best_score = top_score
            print(f"  Added {top_feature!r:18s}  R-squared = {top_score:.4f}")
        else:
            break

    return selected, best_score


print("\n--- Forward selection ---")
fs_features, fs_score = forward_selection(X, y)
print(f"\nSelected features (forward selection): {fs_features}")


# 6. LASSO (L1 regularization)
print("\n--- LASSO (alpha=0.1) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
lasso_r2 = r2_score(y_test, lasso.predict(X_test))

print(f"R-squared: {lasso_r2:.4f}")
for name, coef in zip(X.columns, lasso.coef_):
    status = "kept" if coef != 0 else "dropped (shrunk to 0)"
    print(f"  {name:14s} coef={coef:+.6f}  [{status}]")

lasso_features = [name for name, coef in zip(X.columns, lasso.coef_) if coef != 0]


# Comparison
print("\n=== Comparison of selected features ===")
print(f"  Backward elimination: {be_features}")
print(f"  Forward selection:    {fs_features}")
print(f"  LASSO (alpha=0.1):    {lasso_features}")


# Key takeaways
#
# - The three methods *disagree* on which feature to keep, and that disagreement
#   is the lesson. StudyHours and PrevExamScore are nearly perfectly collinear
#   (correlation > 0.99), so any one of them captures the signal — picking the
#   "right" one isn't really well-defined.
#
# - Backward elimination (statsmodels p-values): kept StudyHours.
#   With perfectly collinear features, OLS attributes all the variance to
#   whichever feature appears first in the design matrix; the other gets a
#   p-value near 1.0 and is dropped. P-value-based selection is fragile under
#   multicollinearity — VIF checks or a pre-dedupe step are usually needed.
#
# - Forward selection (greedy R-squared on holdout): kept PrevExamScore.
#   It tried each feature alone, PrevExamScore happened to score marginally
#   higher on this specific 80/20 split, and then StudyHours added nothing.
#   Stable here, but with a different random_state it could flip.
#
# - LASSO (alpha=0.1): kept PrevExamScore.
#   L1 doesn't choose "the right" feature among correlated ones either —
#   it picks one essentially at random based on which has slightly larger
#   marginal effect. Elastic Net is the usual fix when you want grouped
#   correlated features to survive together.
#
# - Practical advice: detect multicollinearity *before* running selection
#   (correlation matrix, VIF). With perfectly redundant features, none of
#   these methods can tell you which is "the cause" — they only tell you
#   one is sufficient.
