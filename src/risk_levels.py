import json
import warnings

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from preprocess import build_preprocessor, load_xy

warnings.filterwarnings("ignore")

X_train, y_train = load_xy("data/processed/train.csv")
with open("reports/best_params.json") as f:
    best = json.load(f)["Logistic Regression"]

pipe = Pipeline([
    ("pre", build_preprocessor()),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=2000, **best)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# out-of-fold probabilities: every patient is scored by a model that never saw them
proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0001]
band = pd.cut(proba, bins=bins, right=False)
df = pd.DataFrame({"band": band, "y": y_train.values})
table = df.groupby("band", observed=True)["y"].agg(
    patients="count", chd_cases="sum", chd_rate="mean")

print("Overall CHD rate in train:", round(y_train.mean(), 3))
print("\nActual CHD rate by model-score band (out-of-fold, TRAIN only):\n")
print(table.round(3))
table.to_csv("reports/risk_band_table.csv")