import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("reports/figures", exist_ok=True)
df = pd.read_csv("data/raw/framingham.csv")

corr = df.corr()

# Correlation of every feature with the target
print("Correlation with TenYearCHD (sorted):")
print(corr["TenYearCHD"].drop("TenYearCHD").sort_values(ascending=False).round(3))

# Pairs of features that are strongly correlated with each other
print("\nFeature pairs with |correlation| > 0.6:")
cols = corr.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if cols[i] != "TenYearCHD" and cols[j] != "TenYearCHD" and abs(corr.iloc[i, j]) > 0.6:
            print(f"{cols[i]} - {cols[j]}: {corr.iloc[i, j]:.2f}")

# Heatmap
plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("reports/figures/05_correlation_heatmap.png", dpi=150)
plt.close()
print("\nSaved plot 05")