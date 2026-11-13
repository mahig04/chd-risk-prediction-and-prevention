import joblib
import numpy as np
import pandas as pd
import shap

from preprocess import load_xy


class RiskExplainer:
    def __init__(self, model_path="models/best_model.joblib",
                 train_path="data/processed/train.csv"):
        self.model = joblib.load(model_path)
        self.pre = self.model.named_steps["pre"]
        self.clf = self.model.named_steps["clf"]
        X_train, _ = load_xy(train_path)
        Xt_train = self.pre.transform(X_train)
        self.names = [c.split("__", 1)[1] for c in Xt_train.columns]
        self.explainer = shap.LinearExplainer(self.clf, Xt_train)

    def explain(self, patient: dict):
        X = pd.DataFrame([patient])
        Xt = self.pre.transform(X)
        sv = self.explainer(Xt)
        values = sv.values[0]
        base = float(np.ravel(sv.base_values)[0])
        score = float(self.clf.decision_function(Xt)[0])
        risk = float(self.model.predict_proba(X)[0, 1])
        contrib = pd.DataFrame({
            "feature": self.names,
            "patient_value": [X[n].iloc[0] for n in self.names],
            "shap": values,
        })
        contrib = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index)
        return {
            "risk": risk,
            "base_value": base,
            "gap": abs(base + values.sum() - score),
            "contributions": contrib.reset_index(drop=True),
        }


if __name__ == "__main__":
    X_test, y_test = load_xy("data/processed/test.csv")
    ex = RiskExplainer()
    patient = X_test.iloc[579].to_dict()
    out = ex.explain(patient)
    direct = ex.model.predict_proba(X_test.iloc[[579]])[0, 1]
    print("Risk from explain():          ", round(out["risk"], 3))
    print("Risk from model.predict_proba:", round(float(direct), 3))
    print("Additivity gap:", out["gap"])
    print("\nTop 5 contributions (log-odds):")
    print(out["contributions"].head(5).round(3).to_string(index=False))