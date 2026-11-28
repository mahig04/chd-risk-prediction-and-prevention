"""
Streamlit dashboard for the CHD risk prototype.

The page is only drawn here. The real work is done in the src folder:
  src/explain_utils.py -> risk score and SHAP values
  src/recommend.py     -> rule-based advice
"""
import sys

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, "src")  # lets us import the files that live in the src folder
from explain_utils import RiskExplainer
from preprocess import load_xy
from recommend import DISCLAIMER, get_recommendations

st.set_page_config(page_title="CHD Risk Prototype", layout="wide")

# ---------- Colours (risk colours are used only for risk) ----------
ZONE_GREEN, ZONE_AMBER, ZONE_RED = "#9BD3B2", "#F2D18B", "#E8A092"   # meter background
BAR_UP, BAR_DOWN = "#C2412D", "#4A7FB5"                               # SHAP bars
DARK = "#1F2937"

# ---------- Plain-English names for the model's column names ----------
NAMES = {
    "age": "Age", "male": "Sex", "cigsPerDay": "Cigarettes per day",
    "sysBP": "Systolic BP", "diaBP": "Diastolic BP", "totChol": "Total cholesterol",
    "glucose": "Glucose", "heartRate": "Heart rate", "BMI": "BMI",
    "prevalentHyp": "Hypertension", "BPMeds": "BP medication",
    "diabetes": "Diabetes", "prevalentStroke": "Previous stroke",
    "currentSmoker": "Current smoker",
}
YES_NO = {"prevalentHyp", "BPMeds", "diabetes", "prevalentStroke", "currentSmoker"}

# Heading for each advice rule, and the order in which advice is shown (most urgent first)
TOPIC = {
    "R1": "Blood pressure", "R2": "Blood pressure", "R3": "Blood pressure", "R4": "Blood pressure",
    "R5": "Cholesterol", "R6": "Cholesterol", "R7": "Smoking",
    "R8a": "Weight", "R8b": "Weight", "R9a": "Blood sugar", "R9b": "Blood sugar",
    "R10": "Ongoing care", "R11": "Physical activity",
}
ORDER = ["R4", "R3", "R6", "R9b", "R8b", "R7", "R2", "R5", "R9a", "R8a", "R1", "R10", "R11"]
PRIORITY = {"R3", "R4", "R6", "R9b"}  # shown with a red "Priority" tag


# ---------- Loading (cached, so it runs once and not on every click) ----------
@st.cache_resource
def load_explainer():
    return RiskExplainer()


@st.cache_data
def load_ranges():
    """(min, max, median) of every input column in the training data."""
    X, _ = load_xy("data/processed/train.csv")
    return {c: (float(X[c].min()), float(X[c].max()), float(X[c].median())) for c in X.columns}


explainer = load_explainer()
rng = load_ranges()

# ---------- Starting values and the three example patients ----------
DEFAULTS = {
    "sex": "Female", "age": int(rng["age"][2]), "cigsPerDay": 0,
    "BMI": round(rng["BMI"][2], 1), "sysBP": int(rng["sysBP"][2]), "diaBP": int(rng["diaBP"][2]),
    "heartRate": int(rng["heartRate"][2]), "totChol": int(rng["totChol"][2]),
    "glucose": int(rng["glucose"][2]),
    "prevalentHyp": False, "BPMeds": False, "diabetes": False, "prevalentStroke": False,
}
LOWER_RISK = {**DEFAULTS, "age": 40, "sysBP": 110, "diaBP": 70, "totChol": 180, "glucose": 80, "BMI": 22.0}
HIGHER_RISK = {**DEFAULTS, "sex": "Male", "age": 67, "cigsPerDay": 40, "sysBP": 170, "diaBP": 100,
               "totChol": 260, "glucose": 105, "BMI": 29.0, "prevalentHyp": True}
PRESETS = {"Average person": DEFAULTS, "Lower-risk example": LOWER_RISK, "Higher-risk example": HIGHER_RISK}

# st.session_state remembers every input between clicks. Set each one once at the start.
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def apply_preset():
    """Runs when the 'Start from' box changes: copy the example's values into the inputs."""
    for key, value in PRESETS[st.session_state["preset"]].items():
        if key in rng:  # keep numbers inside the range seen in training
            value = type(DEFAULTS[key])(min(max(value, rng[key][0]), rng[key][1]))
        st.session_state[key] = value


def number(label, col, help_text):
    """A whole-number input whose limits come from the training data."""
    low, high, _ = rng[col]
    return st.number_input(label, min_value=int(low), max_value=int(high), key=col, help=help_text)


# ---------- Small helpers that draw things ----------
def risk_level(score):
    """Cut-offs and percentages come from reports/risk_band_table.csv (train data only)."""
    if score >= 0.70:
        return "High", "red", 35.2
    if score >= 0.50:
        return "Medium", "orange", 23.4
    return "Low", "green", 7.8


def value_text(feature, value):
    if feature == "male":
        return "Male" if value == 1 else "Female"
    if feature in YES_NO:
        return "Yes" if value == 1 else "No"
    return f"{value:g}"


def risk_meter(score):
    """A bar split into Low / Medium / High with a black marker at the person's score."""
    fig, ax = plt.subplots(figsize=(8, 1.3))
    for start, end, colour, name in [(0, 0.5, ZONE_GREEN, "Low"), (0.5, 0.7, ZONE_AMBER, "Medium"),
                                     (0.7, 1.0, ZONE_RED, "High")]:
        ax.barh(0, end - start, left=start, color=colour, height=0.6)
        ax.text((start + end) / 2, 0.38, name, ha="center", va="bottom", color=DARK, fontsize=10)
    ax.plot([score, score], [-0.4, 0.34], color="black", linewidth=3)  # the marker
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.8)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 0.7, 1.0])
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


def driver_chart(contributions):
    """Top 6 SHAP bars. Red pushes the score up, blue pulls it down."""
    top = contributions.head(6).iloc[::-1]  # reversed so the biggest bar is at the top
    labels = [f"{NAMES[f]}: {value_text(f, v)}" for f, v in zip(top["feature"], top["patient_value"])]
    colours = [BAR_UP if s > 0 else BAR_DOWN for s in top["shap"]]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(labels, top["shap"], color=colours)
    ax.axvline(0, color="#475569", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Effect on the model score")
    fig.tight_layout()
    return fig


def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)  # free memory


def show_suggestions(patient):
    advice = get_recommendations(patient)
    advice.sort(key=lambda r: ORDER.index(r["id"]) if r["id"] in ORDER else 99)
    for r in advice:
        with st.container(border=True):  # one bordered box = one piece of advice
            tag = ":red[Priority] " if r["id"] in PRIORITY else ""
            st.markdown(f"{tag}**{TOPIC.get(r['id'], 'Advice')}**")
            st.write(r["advice"])
            st.caption("Source: " + r["source"])


# ---------- Sidebar: inputs, grouped by topic ----------
with st.sidebar:
    st.header("Patient details")
    st.selectbox("Start from", list(PRESETS), key="preset", on_change=apply_preset,
                 help="Fills the form with a sample person. You can still change any value.")
    st.caption(f"Limits follow the training data (for example age {rng['age'][0]:.0f} to "
               f"{rng['age'][1]:.0f}). Outside them the model has nothing to learn from.")

    with st.expander("Person", expanded=True):
        sex = st.selectbox("Sex", ["Female", "Male"], key="sex")
        age = number("Age (years)", "age", "Completed years.")

    with st.expander("Lifestyle", expanded=True):
        cigs = number("Cigarettes per day", "cigsPerDay",
                      "0 means non-smoker. 'Current smoker' is set from this value.")
        bmi = st.number_input("BMI", min_value=float(rng["BMI"][0]), max_value=float(rng["BMI"][1]),
                              step=0.1, key="BMI", help="Weight in kg divided by height in metres squared.")

    with st.expander("Blood pressure and heart", expanded=True):
        sbp = number("Systolic BP (mmHg)", "sysBP", "The higher number of a BP reading.")
        dbp = number("Diastolic BP (mmHg)", "diaBP", "The lower number of a BP reading.")
        if dbp >= sbp:
            st.warning("Diastolic BP should be lower than systolic BP. Please re-check the values.")
        hr = number("Heart rate (bpm)", "heartRate", "Resting beats per minute.")

    with st.expander("Blood tests", expanded=True):
        chol = number("Total cholesterol (mg/dL)", "totChol", "Total cholesterol in mg/dL.")
        glucose = number("Glucose (mg/dL)", "glucose", "Blood glucose in mg/dL.")

    with st.expander("Medical history", expanded=True):
        hyp = st.checkbox("Hypertension", key="prevalentHyp")
        bp_meds = st.checkbox("Takes BP medication", key="BPMeds")
        diabetes = st.checkbox("Diabetes", key="diabetes")
        stroke = st.checkbox("Previous stroke", key="prevalentStroke")

# The model expects exactly these 14 inputs, so we collect them in one dictionary.
patient = {
    "male": int(sex == "Male"), "age": age, "currentSmoker": int(cigs > 0),
    "cigsPerDay": cigs, "BPMeds": int(bp_meds), "prevalentStroke": int(stroke),
    "prevalentHyp": int(hyp), "diabetes": int(diabetes), "totChol": chol,
    "sysBP": sbp, "diaBP": dbp, "BMI": bmi, "heartRate": hr, "glucose": glucose,
}

# ---------- Main page ----------
st.title("CHD Risk Prediction and Preventive Recommendation")
st.caption("Estimates a 10-year coronary heart disease risk score from routine health numbers "
           "and shows what drives it.")
st.info(DISCLAIMER)

result = explainer.explain(patient)  # risk score + SHAP contributions for this patient
score = result["risk"]
level, colour, band_rate = risk_level(score)

with st.container(border=True):
    left, right = st.columns([1, 2])
    with left:
        st.caption("10-year risk level (this model)")
        st.caption(f"Model score {score:.2f}. This is not a probability.")
                # A low 10-year score can still come with habits or readings that need attention
        needs_attention = [r for r in get_recommendations(patient) if r["id"] != "R11"]
        if level == "Low" and needs_attention:
            st.caption("Low 10-year score, but some inputs need attention (see Suggestions). "
                       "Age strongly lowers a short-term score for young people, "
                       "yet habits like smoking still matter over a lifetime.")
        st.markdown(f"# :{colour}[{level}]")
        st.caption(f"Model score {score:.2f}. This is not a probability.")
    with right:
        st.write(f"In the training data, about **{band_rate}%** of people in this score band "
                 "developed CHD within 10 years (cross-validated estimate).")
        show_figure(risk_meter(score))

why, advice = st.columns(2)
with why:
    st.subheader("Why this score")
    show_figure(driver_chart(result["contributions"]))
    biggest = result["contributions"].iloc[0]
    st.caption(f"Red bars push the score up and blue bars pull it down. Largest effect here: "
               f"{NAMES[biggest['feature']]} ({value_text(biggest['feature'], biggest['patient_value'])}). "
               "This shows what the model used, not what causes CHD.")
with advice:
    st.subheader("Suggestions")
    show_suggestions(patient)

st.divider()
st.caption(DISCLAIMER + " Based on single readings and a research dataset (Framingham, Kaggle copy).")