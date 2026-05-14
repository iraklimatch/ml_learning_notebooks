# ML Learning Notebooks

Hands-on exercises from the Microsoft "Foundations of AI and Machine Learning" certification on Coursera. Each notebook walks through one practice activity from the course, with my own notes and reflections.

Run on Azure ML compute instances unless otherwise noted.

## Notebooks

- `cifar10_tensorflow.ipynb` - small CNN trained on CIFAR-10 using TensorFlow/Keras. Covers data loading, model build, training, evaluation, and save/reload.
- `churn_prediction/` - Scikit-learn Random Forest for telecom customer churn prediction, with deployment optimization (pruning) and joblib save. Includes the toy dataset, the trained `.pkl` model, and the notebook.
- `deployment_platform_evaluation/` - written exercise comparing Azure ML, AWS SageMaker, and Google Vertex AI for hosting an ML model in production. Picks Azure ML and explains why.
- `three_paradigms/` - one activity covering all three learning paradigms: linear regression for house prices (supervised), K-means customer segmentation (unsupervised), and a Q-learning tic-tac-toe agent trained vs a random opponent (RL). Includes the learning-curve plot for the RL agent.
