import pandas as pd
import streamlit as st

st.set_page_config(page_title="About the model", layout="wide")
st.title("About the model")
st.warning("Educational research prototype. Not clinically validated and not a diagnostic tool.")

st.subheader("What this is")
st.write(
    "A Logistic Regression model trained on the Framingham Heart Study data (Kaggle copy, "
    "4,240 people, 15.2% had CHD within 10 years). It gives a risk score, explains it with SHAP, "
    "and shows rule-based advice from public guidelines."
)

st.subheader("Held-out test results (848 people, 129 with CHD, threshold 0.50)")
test = pd.concat([pd.read_csv("reports/test_results.csv"),
                  pd.read_csv("reports/ensemble_test_result.csv")], ignore_index=True)
st.dataframe(test, hide_index=True, use_container_width=True)
st.caption(
    "A model that says 'no CHD' for everyone would score about 84.8% accuracy, so accuracy is not "
    "a useful measure here. We compare models mainly by ROC-AUC and recall. All models score "
    "within about 0.01 ROC-AUC of each other, which is within the uncertainty of a test set with "
    "only 129 CHD cases. Logistic Regression was chosen before the test set was opened."
)

st.subheader("What the risk levels mean")
bands = pd.read_csv("reports/risk_band_table.csv")
st.dataframe(bands, hide_index=True, use_container_width=True)
st.caption(
    "Computed on the training data only, using cross-validated scores. Low = score below 0.5, "
    "Medium = 0.5 to 0.7, High = 0.7 or above. The score is not a probability because the model "
    "was trained with SMOTE."
)

st.subheader("How to read the SHAP chart")
st.write(
    "Each bar shows how much one input pushed this person's score up (red) or down (blue). "
    "It explains what the model did, not what causes CHD. When two inputs carry similar "
    "information (for example diabetes and glucose), the model may give weight to only one."
)

st.subheader("Limitations")
st.markdown(
    "- Not clinically validated. Do not use for real medical decisions.\n"
    "- One dataset from one study population. It may not apply to other populations, including India.\n"
    "- Modest accuracy: ROC-AUC about 0.70 on the test set. At threshold 0.50 the model finds "
    "about 6 in 10 CHD cases (recall) and roughly 1 in 4 flagged people actually develop CHD (precision).\n"
    "- Test set has only 129 CHD cases, so small differences between models are not meaningful.\n"
    "- Blood pressure is a single reading, and it is not known whether glucose was fasting. "
    "Extreme values in the data were kept.\n"
    "- Not included in the model: family history, LDL/HDL cholesterol, diet, physical activity.\n"
    "- Advice comes from general public guidelines (ACC/AHA, WHO, CDC, NHLBI, ADA) and is not "
    "personal medical advice."
)