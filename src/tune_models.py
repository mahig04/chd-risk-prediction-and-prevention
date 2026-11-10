import json
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from preprocess import load_xy, build_preprocessor

X_train, y_train = load_xy("data/processed/train.csv")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search_space = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000),
        {"clf__C": loguniform(1e-3, 1e2)},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {"clf__n_estimators": [100, 200, 300],
         "clf__max_depth": [3, 5, 8, None],
         "clf__min_samples_leaf": [1, 5, 10, 20]},
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=42),
        {"clf__n_estimators": [100, 200, 300],
         "clf__max_depth": [2, 3, 4],
         "clf__learning_rate": [0.01, 0.05, 0.1],
         "clf__subsample": [0.7, 0.85, 1.0]},
    ),
    "SVM": (
        SVC(kernel="rbf", probability=True, random_state=42),
        {"clf__C": loguniform(1e-2, 1e1),
         "clf__gamma": loguniform(1e-3, 1e-1)},
    ),
}

rows, best_params = [], {}
for name, (clf, params) in search_space.items():
    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("clf", clf),
    ])
    search = RandomizedSearchCV(pipe, params, n_iter=8, scoring="roc_auc",
                                cv=cv, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    clean = {k.replace("clf__", ""): (float(v) if hasattr(v, "item") else v)
             for k, v in search.best_params_.items()}
    best_params[name] = clean
    rows.append({"Model": name,
                 "Best CV ROC-AUC": round(search.best_score_, 3),
                 "Best params": clean})
    print("done:", name, "| best CV ROC-AUC =", round(search.best_score_, 3))

print("\nTuning results (5-fold CV on TRAIN only)\n")
print(pd.DataFrame(rows).to_string(index=False))

with open("reports/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2, default=str)
pd.DataFrame(rows).to_csv("reports/tuning_results.csv", index=False)