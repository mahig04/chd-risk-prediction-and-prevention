import pandas as pd

df = pd.read_csv("data/raw/framingham.csv")

print("Shape (rows, columns):", df.shape)
print("\nColumns and types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nTarget distribution (TenYearCHD):")
print(df["TenYearCHD"].value_counts())
print(df["TenYearCHD"].value_counts(normalize=True).round(3))
print("\nDuplicate rows:", df.duplicated().sum())