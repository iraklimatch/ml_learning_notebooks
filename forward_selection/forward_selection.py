"""
Practice activity: Forward feature selection.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# 1. Load and prepare the data
data = {
    'StudyHours':    [1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass':          [0,  0,  0,  0,  0,  1,  1,  1,  1,  1],
}
df = pd.DataFrame(data)
print(df)

X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']


# 2. Forward selection
def forward_selection(X, y):
    """Greedy forward feature selection by R-squared on a held-out split.

    Start with no features. Each round, try adding each remaining feature
    one at a time and keep the one that gives the highest R-squared on the
    test split. Stop when no remaining feature improves the score.
    """
    remaining_features = set(X.columns)
    selected_features = []
    best_score = 0.0

    while remaining_features:
        scores_this_round = []

        for feature in remaining_features:
            trial = selected_features + [feature]
            X_train, X_test, y_train, y_test = train_test_split(
                X[trial], y, test_size=0.2, random_state=42
            )
            model = LinearRegression()
            model.fit(X_train, y_train)
            score = r2_score(y_test, model.predict(X_test))
            scores_this_round.append((score, feature))

        # Best candidate this round
        scores_this_round.sort(reverse=True)
        top_score, top_feature = scores_this_round[0]

        if top_score > best_score:
            selected_features.append(top_feature)
            remaining_features.remove(top_feature)
            best_score = top_score
            print(f"  Added {top_feature!r:18s}  R-squared = {top_score:.4f}")
        else:
            # No remaining feature improved the score — stop.
            break

    return selected_features, best_score


print("\n--- Running forward selection ---")
best_features, best_score = forward_selection(X, y)
print(f"\nSelected features: {best_features}")
print(f"Best R-squared during selection: {best_score:.4f}")


# 4. Evaluate the final model on the selected features
X_train, X_test, y_train, y_test = train_test_split(
    X[best_features], y, test_size=0.2, random_state=42
)
final_model = LinearRegression()
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
final_r2 = r2_score(y_test, y_pred)

print(f"\nFinal R-squared with selected features: {final_r2:.4f}")
print(f"Coefficients: {dict(zip(best_features, final_model.coef_))}")
print(f"Intercept:    {final_model.intercept_:.4f}")


# Key takeaways
#
# - Forward selection is greedy: at each step it picks the feature whose
#   *marginal* contribution is largest given what's already selected.
#   It's fast but can miss feature combinations that only shine together.
# - On this dataset, StudyHours and PrevExamScore are highly correlated,
#   so once one is selected the other contributes very little extra R².
# - The stopping rule here ("no improvement") is the most common variant.
#   More sophisticated rules use p-values, AIC/BIC, or cross-validated
#   scores instead of a single test split.
