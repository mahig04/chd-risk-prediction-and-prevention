import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay)
from sklearn.svm import SVC
from xgboost import XGBClassifier

from preprocess import build_preprocessor, load_xy

warnings.filterwarnings("ignore", category=FutureWarning)

THRESHOLD = 0.50
BEST_MODEL = "Logistic Regression"   # chosen from tuning CV ROC-AUC, BEFORE seeing test

X_train, y_train = load_xy("data/processed/train.csv")
X_test, y_test = load_xy("data/processed/test.csv")

with open("reports/best_params.json") as f:
    best = json.load(f)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

rows, fitted, curves, cms = [], {}, {}, {}
for name, clf in models.items():
    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("clf", clf),
    ])
    pipe.set_params(**{f"clf__{k}": v for k, v in best[name].items()})
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    fitted[name] = pipe
    curves[name] = roc_curve(y_test, proba)
    cms[name] = confusion_matrix(y_test, pred)
    rows.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, pred), 3),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 3),
        "Recall": round(recall_score(y_test, pred), 3),
        "F1": round(f1_score(y_test, pred), 3),
        "ROC-AUC": round(roc_auc_score(y_test, proba), 3),
    })
    print("done:", name)

baseline_acc = round(1 - y_test.mean(), 3)
table = pd.DataFrame(rows)
print(f"\nFINAL TEST RESULTS (threshold = {THRESHOLD}, n_test = {len(y_test)}, "
      f"positives = {int(y_test.sum())})\n")
print(table.to_string(index=False))
print(f"\nBaseline: predicting 'no CHD' for everyone gives accuracy = {baseline_acc}")
table.to_csv("reports/test_results.csv", index=False)

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for ax, (name, cm) in zip(axes.ravel(), cms.items()):
    ConfusionMatrixDisplay(cm, display_labels=["No CHD", "CHD"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("reports/figures/06_confusion_matrices.png", dpi=150)
plt.close()

# ROC curves
plt.figure(figsize=(6.5, 6))
for name, (fpr, tpr, _) in curves.items():
    auc = table.loc[table["Model"] == name, "ROC-AUC"].iloc[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (recall)")
plt.title("ROC curves on test set")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/07_roc_curves.png", dpi=150)
plt.close()

# Save the pre-chosen final model
joblib.dump(fitted[BEST_MODEL], "models/best_model.joblib")
print(f"\nSaved models/best_model.joblib ({BEST_MODEL}) and 2 plots in reports/figures/")