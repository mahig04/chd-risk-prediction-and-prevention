import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("reports/figures", exist_ok=True)
df = pd.read_csv("data/raw/framingham.csv")

# Plot 1: class balance
counts = df["TenYearCHD"].value_counts().sort_index()
counts.plot(kind="bar")
plt.title("Target distribution (TenYearCHD)")
plt.xticks([0, 1], ["No CHD (0)", "CHD (1)"], rotation=0)
plt.ylabel("Number of patients")
for i, v in enumerate(counts):
    plt.text(i, v + 30, str(v), ha="center")
plt.tight_layout()
plt.savefig("reports/figures/01_target_distribution.png", dpi=150)
plt.close()

# Plot 2: missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values()
missing.plot(kind="barh")
plt.title("Missing values per column")
plt.xlabel("Number of missing values")
plt.tight_layout()
plt.savefig("reports/figures/02_missing_values.png", dpi=150)
plt.close()

print("Saved 2 plots in reports/figures/")