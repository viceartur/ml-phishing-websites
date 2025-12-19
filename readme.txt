Phishing Website Detection with Machine Learning.
This project evaluates multiple machine learning models for phishing website detection using the UCI Phishing Websites dataset.
The models are compared using ROC curves and classification accuracy, and feature importance is analyzed using a Random Forest classifier.

Authors: Artur Sopov & Sam Pagon

Dataset:
Source: UCI Machine Learning Repository
Dataset: Phishing Websites (ID: 327)
Features: Website-based characteristics
Target: Binary classification (phishing vs legitimate)
Class priors are computed to understand the class distribution before model training.

Models Implemented:
The following classifiers are trained and evaluated:

1. Logistic Regression
- Default
- Strong regularization (C = 0.001)
- Weak regularization (C = 100)

2. Support Vector Machine (SVM)
- Default polynomial kernel
- Tuned polynomial kernels with different degrees and C values

3. Decision Tree
- Default
- Max depth = 5 (entropy)
- Max depth = 15 (entropy)

4. Ensemble Methods
- Soft Voting Classifier (Logistic Regression + SVM + Decision Tree)
- Random Forest

Evaluation Metrics:
- Accuracy
- ROC Curve & AUC

ROC curves are generated for:
- Individual model families
- Final comparison across all models

Feature Importance:
Feature importance is computed using permutation importance on the trained Random Forest model.
Results highlight which website features contribute most to phishing detection accuracy.

Generated Outputs:
The script saves the following figures:
lr_roc_curves.png – Logistic Regression ROC curves
svm_roc_curves.png – SVM ROC curves
dt_roc_curves.png – Decision Tree ROC curves
all_roc_comparison.png – ROC comparison of all models
feature_importances.png – Random Forest feature importance

Requirements:
Python 3.8+
scikit-learn
matplotlib
pandas
ucimlrepo

Install dependencies with:
pip install scikit-learn matplotlib pandas ucimlrepo

How to Run:
python final_project.py

All results and plots will be saved in the project directory.

Purpose:
This project demonstrates:
- Model comparison for binary classification
- Effects of hyperparameter tuning
- Ensemble learning benefits
- Interpretability using permutation feature importance