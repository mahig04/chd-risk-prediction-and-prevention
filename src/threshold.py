import json
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from xgboost import XGBClassifier

from preprocess import build_preprocessor, load_xy

warnings.filterwarnings("ignore", category=FutureWarning)

X_train, y_train = load_xy("data/processed/train.csv")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with open("reports/best_params.json") as f:
    best = json.load(f)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

grid = np.arange(0.20, 0.81, 0.05)
rows, chosen = [], {}

for name, clf in models.items():
    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("clf", clf),
    ])
    pipe.set_params(**{f"clf__{k}": v for k, v in best[name].items()})

    # out-of-fold probabilities: each patient is scored by a model that never saw them
    proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    def metrics_at(t):
        pred = (proba >= t).astype(int)
        return (precision_score(y_train, pred, zero_division=0),
                recall_score(y_train, pred),
                f1_score(y_train, pred))

    f1s = [metrics_at(t)[2] for t in grid]
    t_best = float(grid[int(np.argmax(f1s))])
    p, r, f = metrics_at(t_best)
    p5, r5, f5 = metrics_at(0.5)
    chosen[name] = round(t_best, 2)
    rows.append({"Model": name, "Best-F1 threshold": round(t_best, 2),
                 "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f, 3),
                 "F1 at 0.50": round(f5, 3), "Recall at 0.50": round(r5, 3)})
    print("done:", name)

table = pd.DataFrame(rows)
print("\nThreshold selection (out-of-fold on TRAIN only)\n")
print(table.to_string(index=False))

with open("reports/thresholds.json", "w") as f:
    json.dump(chosen, f, indent=2)
table.to_csv("reports/threshold_results.csv", index=False)