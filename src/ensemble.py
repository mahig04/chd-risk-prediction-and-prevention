import json
import warnings

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from xgboost import XGBClassifier

from preprocess import build_preprocessor, load_xy

warnings.filterwarnings("ignore", category=FutureWarning)

X_train, y_train = load_xy("data/processed/train.csv")
X_test, y_test = load_xy("data/processed/test.csv")

with open("reports/best_params.json") as f:
    best = json.load(f)

voting = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=2000, **best["Logistic Regression"])),
        ("rf", RandomForestClassifier(random_state=42, n_jobs=-1, **best["Random Forest"])),
        ("xgb", XGBClassifier(eval_metric="logloss", random_state=42, **best["XGBoost"])),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42, **best["SVM"])),
    ],
    voting="soft",
)

pipe = Pipeline([
    ("pre", build_preprocessor()),
    ("smote", SMOTE(random_state=42)),
    ("clf", voting),
])

# 1) 5-fold CV on TRAIN only
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
print("Voting Ensemble, 5-fold CV on TRAIN:")
for m in scoring:
    print(f"  {m:10s} {res['test_' + m].mean():.3f} +/- {res['test_' + m].std():.3f}")

# 2) Fit on full train, evaluate on test once
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
pred = (proba >= 0.50).astype(int)
row = {
    "Model": "Voting Ensemble",
    "Accuracy": round(accuracy_score(y_test, pred), 3),
    "Precision": round(precision_score(y_test, pred, zero_division=0), 3),
    "Recall": round(recall_score(y_test, pred), 3),
    "F1": round(f1_score(y_test, pred), 3),
    "ROC-AUC": round(roc_auc_score(y_test, proba), 3),
}
print("\nVoting Ensemble, TEST (threshold 0.50):")
print(pd.DataFrame([row]).to_string(index=False))
pd.DataFrame([row]).to_csv("reports/ensemble_test_result.csv", index=False)