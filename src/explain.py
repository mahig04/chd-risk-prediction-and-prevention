import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from preprocess import load_xy

model = joblib.load("models/best_model.joblib")
pre = model.named_steps["pre"]
clf = model.named_steps["clf"]

X_train, y_train = load_xy("data/processed/train.csv")
X_test, y_test = load_xy("data/processed/test.csv")

Xt_train = pre.transform(X_train)
Xt_test = pre.transform(X_test)
names = [c.split("__", 1)[1] for c in Xt_train.columns]   # short names, only for plots

explainer = shap.LinearExplainer(clf, Xt_train)
sv = explainer(Xt_test)
sv.feature_names = names

# Sanity check: SHAP values + base value must add up to the model's own score
gap = np.abs(sv.values.sum(axis=1) + sv.base_values - clf.decision_function(Xt_test)).max()
print("Additivity check (should be ~0):", float(gap))

# Ranking by mean |SHAP|
ranking = pd.Series(np.abs(sv.values).mean(axis=0), index=names).sort_values(ascending=False)
print("\nMean |SHAP| per feature (higher = more influence):")
print(ranking.round(3))

# Plot 08: beeswarm (direction + size of effect)
shap.summary_plot(sv.values, Xt_test, feature_names=names, show=False)
plt.tight_layout()
plt.savefig("reports/figures/08_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 09: bar (overall importance)
shap.summary_plot(sv.values, Xt_test, feature_names=names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("reports/figures/09_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 10: one patient (highest predicted risk in the test set)
proba = model.predict_proba(X_test)[:, 1]
idx = int(np.argmax(proba))
print(f"\nPatient index {idx}: predicted risk = {proba[idx]:.3f}, actual outcome = {int(y_test.iloc[idx])}")
shap.plots.waterfall(sv[idx], show=False)
plt.savefig("reports/figures/10_shap_patient.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved plots 08, 09, 10 in reports/figures/")