"""
Practice activity: Implementing and comparing logistic regression vs decision tree.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree


# 3. Load and prepare the data
data = {
    'StudyHours':    [1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass':          [0,  0,  0,  0,  0,  1,  1,  1,  1,  1],
}
df = pd.DataFrame(data)
print(df)


# 4. Split into training and testing sets
X = df[['StudyHours', 'PrevExamScore']]
y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining data: {X_train.shape}, {y_train.shape}")
print(f"Testing data:  {X_test.shape}, {y_test.shape}")


# 5. Logistic regression
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
y_pred_logreg = logreg_model.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)


# 6. Decision tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)


# 7. Compare model performance
print("\nLogistic Regression:")
print(f"Accuracy: {accuracy_logreg}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))
print("Classification Report:")
print(classification_report(y_test, y_pred_logreg, zero_division=0))

print("\nDecision Tree:")
print(f"Accuracy: {accuracy_tree}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))
print("Classification Report:")
print(classification_report(y_test, y_pred_tree, zero_division=0))


# 8. Visualize the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(
    tree_model,
    feature_names=['StudyHours', 'PrevExamScore'],
    class_names=['Fail', 'Pass'],
    filled=True,
)
plt.title('Decision Tree for Classifying Pass/Fail')
plt.savefig('decision_tree.png', dpi=120, bbox_inches='tight')
plt.show()


# 9. Reflection notes
#
# - Both models achieve 100% accuracy on the (tiny) test set because the
#   data is linearly separable around StudyHours ~= 5 / PrevExamScore ~= 60.
# - Logistic regression suits this dataset because the decision boundary
#   is roughly linear. It is simpler, faster, and easier to interpret in
#   terms of coefficients.
# - The decision tree captures the same boundary but with axis-aligned
#   splits. On a 10-row dataset it overfits trivially; on larger, noisier
#   data we'd cap max_depth or min_samples_leaf to control overfitting.
