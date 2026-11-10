import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "TenYearCHD"
DROP_COLS = ["education"]
NUMERIC = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
BINARY = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]


def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = df[TARGET]
    return X, y


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary_pipe = SimpleImputer(strategy="most_frequent")
    pre = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("bin", binary_pipe, BINARY),
    ])
    pre.set_output(transform="pandas")
    return pre


if __name__ == "__main__":
    X_train, y_train = load_xy("data/processed/train.csv")
    X_test, y_test = load_xy("data/processed/test.csv")

    pre = build_preprocessor()
    Xt_train = pre.fit_transform(X_train)   # learns from TRAIN only
    Xt_test = pre.transform(X_test)         # applies train's values to test

    print("Train shape:", Xt_train.shape, "| Test shape:", Xt_test.shape)
    print("Missing in train:", int(Xt_train.isnull().sum().sum()))
    print("Missing in test:", int(Xt_test.isnull().sum().sum()))
    num_cols = [c for c in Xt_train.columns if c.startswith("num__")]
    print("\nTrain mean (should be ~0):")
    print(Xt_train[num_cols].mean().round(3))
    print("\nTest mean (close to 0 but NOT exactly 0):")
    print(Xt_test[num_cols].mean().round(3))