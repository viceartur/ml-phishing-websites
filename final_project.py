from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
import pandas as pd

phishing_websites = fetch_ucirepo(id=327)
X = phishing_websites.data.features
Y = phishing_websites.data.targets
Y = Y.values.ravel()

# Dataset Class Priors
class_priors = pd.Series(Y).value_counts(normalize=True).sort_index()
print("Class priors (entire dataset):")
for cls, p in class_priors.items():
    print(f"P(Y={cls}) = {p:.4f}")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Linear Regression
# defaults: penalty='l2', tol=1e-4, C=1, max_iter=100
LR_clf_default = LogisticRegression().fit(X_train, Y_train)
LR_clf_strong_reg = LogisticRegression(C=0.001, max_iter=2000).fit(X_train, Y_train)
LR_clf_weak_reg = LogisticRegression(C=100, max_iter=2000).fit(X_train, Y_train)

fig, ax = plt.subplots(figsize=(10, 8))

models = [
    ("Default", LR_clf_default),
    ("Strong Reg (C=0.001)", LR_clf_strong_reg),
    ("Weak Reg (C=100)", LR_clf_weak_reg)
]

for name, model in models:
    acc = model.score(X_test, Y_test)
    RocCurveDisplay.from_estimator(
        model, X_test, Y_test, ax=ax,
        name=f"{name} (Acc: {acc:.4%})"
    )

ax.set_title("LR ROC curves")
ax.grid(linestyle='--')
plt.legend(loc="lower right")
plt.savefig("lr_roc_curves.png")
plt.close()

# Support Vector Machine
# defaults: kernel='poly', degree=3, C=1.0
SVM_clf_default = SVC(probability=True).fit(X_train, Y_train)
SVM_clf_hyper1 = SVC(probability=True, kernel='poly', degree=2, C=0.1).fit(X_train, Y_train)
SVM_clf_hyper2 = SVC(probability=True, kernel='poly', degree=4, C=10).fit(X_train, Y_train)

fig, ax = plt.subplots(figsize=(10, 8))

for name, model in [("Default", SVM_clf_default),
                    ("Hyper 1 (C=0.1, degree=2)", SVM_clf_hyper1),
                    ("Hyper 2 (C=10, degree=4)", SVM_clf_hyper2)]:
    acc = model.score(X_test, Y_test)
    RocCurveDisplay.from_estimator(model, X_test, Y_test, ax=ax,
                                   name=f"{name} (Acc: {acc:.4%})")

ax.set_title("SVM ROC curves")
ax.grid(linestyle='--')
plt.legend(loc="lower right")
plt.savefig("svm_roc_curves.png")
plt.close()

# Decision Tree
# defaults: criterion='gini', max_depth=None
DT_clf_default = DecisionTreeClassifier().fit(X_train, Y_train)
DT_clf_hyper1 = DecisionTreeClassifier(max_depth=5, criterion='entropy').fit(X_train, Y_train)
DT_clf_hyper2 = DecisionTreeClassifier(max_depth=15, criterion='entropy').fit(X_train, Y_train)

fig, ax = plt.subplots(figsize=(10, 8))

for name, model in [
    ("Default", DT_clf_default),
    ("Hyper 1 (max_depth=5)", DT_clf_hyper1),
    ("Hyper 2 (max_depth=15)", DT_clf_hyper2)
]:
    acc = model.score(X_test, Y_test)
    RocCurveDisplay.from_estimator(model, X_test, Y_test, ax=ax,
                                   name=f"{name} (Acc: {acc:.4%})")

ax.set_title("Decision Tree ROC curves")
ax.grid(linestyle='--')
plt.legend(loc="lower right")
plt.savefig("dt_roc_curves.png")
plt.close()

# Ensemble Voting
VE_ensemble = VotingClassifier(
    estimators=[('lr', LR_clf_default), ('svm', SVM_clf_hyper2), ('dt', DT_clf_hyper2)], voting='soft'
).fit(X_train, Y_train)

# Ensemble Random Forest
RF_ensemble = RandomForestClassifier().fit(X_train, Y_train)

# ROC curves
fig, ax = plt.subplots(figsize=(10, 8))

acc_lr = LR_clf_default.score(X_test, Y_test)
RocCurveDisplay.from_estimator(
    LR_clf_default, X_test, Y_test, ax=ax,
    name=f'LR default (Acc: {acc_lr:.4%})'
)

acc_svm = SVM_clf_hyper2.score(X_test, Y_test)
RocCurveDisplay.from_estimator(
    SVM_clf_hyper2, X_test, Y_test, ax=ax,
    name=f'SVM hyper2 (Acc: {acc_svm:.4%})'
)

acc_dt = DT_clf_hyper2.score(X_test, Y_test)
RocCurveDisplay.from_estimator(
    DT_clf_hyper2, X_test, Y_test, ax=ax,
    name=f'DT hyper2 (Acc: {acc_dt:.4%})'
)

acc_ve = VE_ensemble.score(X_test, Y_test)
RocCurveDisplay.from_estimator(
    VE_ensemble, X_test, Y_test, ax=ax,
    name=f'Voting Classifier (Acc: {acc_ve:.4%})'
)

acc_rf = RF_ensemble.score(X_test, Y_test)
RocCurveDisplay.from_estimator(
    RF_ensemble, X_test, Y_test, ax=ax,
    name=f'Random Forest (Acc: {acc_rf:.2%})'
)

ax.set_title("ROC Curve Comparison with Accuracy Scores")
ax.grid(linestyle='--')
plt.legend(loc="lower right")
plt.savefig("all_roc_comparison.png")
plt.close()

# Feature Importance
result = permutation_importance(
    RF_ensemble, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=2
)

forest_importances = pd.Series(result.importances_mean, index=phishing_websites.variables[:-1]['name'])

fig, ax = plt.subplots(figsize=(10, 8))
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.savefig("feature_importances.png")
plt.close()