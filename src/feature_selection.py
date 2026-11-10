import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import load_xy, NUMERIC, BINARY

X_train, y_train = load_xy("data/processed/train.csv")


def make_pipe(drop):
    num = [c for c in NUMERIC if c not in drop]
    binary = [c for c in BINARY if c not in drop]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


experiments = {
    "All 14 features": [],
    "Drop diaBP": ["diaBP"],
    "Drop currentSmoker": ["currentSmoker"],
    "Drop prevalentHyp": ["prevalentHyp"],
    "Drop diaBP + currentSmoker + prevalentHyp": ["diaBP", "currentSmoker", "prevalentHyp"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("5-fold CV on TRAIN only (Logistic Regression, class_weight='balanced')\n")
for name, drop in experiments.items():
    scores = cross_val_score(make_pipe(drop), X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"{name:45s} ROC-AUC = {scores.mean():.3f} +/- {scores.std():.3f}")