import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/framingham.csv")

X = df.drop(columns=["TenYearCHD"])
y = df["TenYearCHD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train = X_train.copy()
train["TenYearCHD"] = y_train
test = X_test.copy()
test["TenYearCHD"] = y_test

train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain CHD rate:", round(y_train.mean(), 3), "| positives:", int(y_train.sum()))
print("Test CHD rate:", round(y_test.mean(), 3), "| positives:", int(y_test.sum()))