import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("reports/figures", exist_ok=True)
df = pd.read_csv("data/raw/framingham.csv")

# Plot 3: numeric features, split by CHD outcome
num_cols = ["age", "sysBP", "totChol", "BMI"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.ravel(), num_cols):
    for label, name in [(0, "No CHD"), (1, "CHD")]:
        data = df.loc[df["TenYearCHD"] == label, col].dropna()
        ax.hist(data, bins=25, alpha=0.6, density=True, label=name)
    ax.set_title(col)
    ax.legend()
plt.suptitle("Feature distributions by CHD outcome")
plt.tight_layout()
plt.savefig("reports/figures/03_numeric_by_chd.png", dpi=150)
plt.close()

# Plot 4: CHD rate (%) by binary features
bin_cols = ["male", "currentSmoker", "diabetes", "prevalentHyp", "prevalentStroke", "BPMeds"]
rates = pd.DataFrame({c: df.groupby(c)["TenYearCHD"].mean() * 100 for c in bin_cols}).T
rates.columns = ["Feature = 0", "Feature = 1"]
print("CHD rate (%) by feature value:")
print(rates.round(1))

rates.plot(kind="bar")
plt.title("CHD rate (%) by binary feature")
plt.ylabel("% patients with CHD")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("reports/figures/04_chd_rate_by_binary.png", dpi=150)
plt.close()

print("Saved plots 03 and 04")