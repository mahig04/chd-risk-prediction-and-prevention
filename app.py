import sys

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, "src")
from explain_utils import RiskExplainer
from preprocess import load_xy
from recommend import DISCLAIMER, get_recommendations

st.set_page_config(page_title="CHD Risk Prototype", layout="wide")


@st.cache_resource
def load_explainer():
    return RiskExplainer()


@st.cache_data
def load_ranges():
    X, _ = load_xy("data/processed/train.csv")
    return {c: (float(X[c].min()), float(X[c].max()), float(X[c].median())) for c in X.columns}


def risk_level(score):
    # cut-offs and band rates come from reports/risk_band_table.csv (out-of-fold, train only)
    if score >= 0.70:
        return "High", 35.2
    if score >= 0.50:
        return "Medium", 23.4
    return "Low", 7.8


explainer = load_explainer()
rng = load_ranges()


def int_input(label, col):
    lo, hi, med = rng[col]
    return st.number_input(label, int(lo), int(hi), int(med))


st.title("CHD Risk Prediction and Preventive Recommendation")
st.warning(DISCLAIMER)

with st.sidebar:
    st.header("Patient details")
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = int_input("Age (years)", "age")
    cigs = int_input("Cigarettes per day (0 = non-smoker)", "cigsPerDay")
    sbp = int_input("Systolic BP (mmHg)", "sysBP")
    dbp = int_input("Diastolic BP (mmHg)", "diaBP")
    chol = int_input("Total cholesterol (mg/dL)", "totChol")
    glucose = int_input("Glucose (mg/dL)", "glucose")
    hr = int_input("Heart rate (bpm)", "heartRate")
    lo, hi, med = rng["BMI"]
    bmi = st.number_input("BMI", float(lo), float(hi), round(med, 1), step=0.1)
    hyp = st.checkbox("Hypertensive")
    bp_meds = st.checkbox("On BP medication")
    diabetes = st.checkbox("Diabetes")
    stroke = st.checkbox("Previous stroke")

patient = {
    "male": int(sex == "Male"), "age": age, "currentSmoker": int(cigs > 0),
    "cigsPerDay": cigs, "BPMeds": int(bp_meds), "prevalentStroke": int(stroke),
    "prevalentHyp": int(hyp), "diabetes": int(diabetes), "totChol": chol,
    "sysBP": sbp, "diaBP": dbp, "BMI": bmi, "heartRate": hr, "glucose": glucose,
}

result = explainer.explain(patient)
score = result["risk"]
level, band_rate = risk_level(score)

left, right = st.columns(2)

with left:
    st.subheader("Model risk score")
    st.metric("Score (0 to 1)", f"{score:.2f}")
    message = (f"Level: {level}. In the training data, about {band_rate}% of people in this "
               "score band developed CHD within 10 years (cross-validated estimate).")
    {"Low": st.success, "Medium": st.warning, "High": st.error}[level](message)
    st.caption("This score is not a probability. The model was trained with SMOTE, which "
               "inflates scores. Current smoker status is set automatically from cigarettes per day.")

with right:
    st.subheader("What drives this score")
    top = result["contributions"].head(6).iloc[::-1]
    colors = ["tab:red" if v > 0 else "tab:blue" for v in top["shap"]]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(top["feature"], top["shap"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Effect on model score (log-odds). Red raises, blue lowers.")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Suggestions")
for r in get_recommendations(patient):
    st.markdown(f"**{r['id']}**: {r['advice']}  \n*Source: {r['source']}*")

st.divider()
st.caption(DISCLAIMER + " Based on single readings and a research dataset (Framingham, Kaggle copy).")