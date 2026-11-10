import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from xgboost import XGBClassifier

from preprocess import load_xy, build_preprocessor

X_train, y_train = load_xy("data/processed/train.csv")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             eval_metric="logloss", random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42),
}

scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []
for name, clf in models.items():
    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("clf", clf),
    ])
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    row = {"Model": name}
    for m in scoring:
        row[m] = f"{res['test_' + m].mean():.3f} +/- {res['test_' + m].std():.3f}"
    rows.append(row)
    print("done:", name)

table = pd.DataFrame(rows)
print("\n5-fold CV on TRAIN only (SMOTE inside pipeline)\n")
print(table.to_string(index=False))
table.to_csv("reports/cv_results.csv", index=False)