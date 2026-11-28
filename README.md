# CHD Risk Prediction and Preventive Recommendation

## Project Overview

This project aims to develop a machine learning-based system for predicting the risk of Coronary Heart Disease (CHD) using clinical and lifestyle parameters.

The system will also provide explainable insights about the factors contributing to the prediction and generate preventive recommendations based on identified risk factors.

##  Objectives

* Predict the risk of Coronary Heart Disease using machine learning.
* Preprocess clinical and lifestyle data.
* Compare different machine learning models.
* Explain model predictions using SHAP.
* Generate personalized preventive recommendations.
* Develop an interactive web dashboard.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Jupyter Notebook
* Git & GitHub

##  Machine Learning Models

The project will compare:

* Logistic Regression
* Random Forest
* XGBoost
* Support Vector Machine (SVM)
* Voting Ensemble (optional)


##  Evaluation Metrics

Accuracy, Precision, Recall, F1-score and ROC-AUC. Class imbalance is handled on the training data only, to prevent data leakage.


##  Dataset

The project will use the following open-source dataset:

* Framingham Heart Study dataset (primary dataset)

##  Proposed Workflow

Patient Clinical & Lifestyle Data
↓
Data Preprocessing
↓
Feature Engineering
↓
Machine Learning Models
↓
CHD Risk Prediction
↓
SHAP Explanation
↓
Preventive Recommendations
↓
Interactive Dashboard

##  Expected Output

The system will provide:

* CHD risk prediction
* Risk level
* Important contributing factors
* Explainable ML insights
* Preventive recommendations
* Interactive dashboard

##  Project Status

Research prototype complete. Educational use only, not clinically validated.

##  Disclaimer

This project is an academic machine learning prototype and is not intended to replace professional medical diagnosis or medical advice.



##  Team

* Mahi Garg
* Pooja Kaur
* Kunal Gautam
* Sunny Yadav

##  Future Scope

* Improve model performance with larger datasets.
* Add more explainability techniques.
* Improve the recommendation system.
* Deploy the application for demonstration and research purposes.
## How to Run

Tested with Python 3.13 on Windows. Key versions: scikit-learn 1.9.1, xgboost 3.4.1, shap 0.54.1, streamlit 1.67.0, pandas 3.1.1.

### 1. Set up the environment

```
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Get the dataset

The dataset is not included in this repository (the Kaggle page does not state a licence). Download **Framingham heart study dataset** (`framingham.csv`, 4,240 rows) from Kaggle and place it at:

```
data/raw/framingham.csv
```

### 3. Create the train/test split

```
python src/split_data.py
```

This writes `data/processed/train.csv` (3,392 rows) and `data/processed/test.csv` (848 rows).

### 4. Run the dashboard

```
streamlit run app.py
```

The trained model (`models/best_model.joblib`) and the result tables in `reports/` are already in the repository, so steps 1 to 4 are enough to open the dashboard.

### 5. (Optional) Reproduce every result

Run these in order. Some take 5 to 15 minutes.

| Script | What it does |
|---|---|
| `src/check_data.py` | Shape, missing values and class balance of the raw data |
| `src/eda.py`, `eda_2.py`, `eda_3.py` | EDA plots saved in `reports/figures/` |
| `src/preprocess.py` | Checks the preprocessing pipeline (fit on train only) |
| `src/feature_selection.py` | Cross-validated test of dropping correlated features |
| `src/train_models.py` | 5-fold CV comparison of the 4 models (SMOTE inside the pipeline) |
| `src/tune_models.py` | Hyperparameter tuning, writes `reports/best_params.json` |
| `src/threshold.py` | Threshold analysis on training data only |
| `src/evaluate.py` | Final test evaluation, writes `models/best_model.joblib` and `reports/test_results.csv` |
| `src/ensemble.py` | Voting ensemble (extra model from the synopsis) |
| `src/explain.py` | SHAP plots |
| `src/risk_levels.py` | Risk bands for Low / Medium / High |
| `src/test_recommend.py` | Tests for the rule-based advice |

`evaluate.py` is the only script that uses the test set. Running it again re-fits the models and rewrites the model file and result tables.

### Troubleshooting

- A scikit-learn version warning from `explain_utils.py` or the dashboard:** run `python src/evaluate.py` once to rebuild `models/best_model.joblib` on your machine.
- FileNotFoundError ... train.csv`:** run step 3 first.
- The dashboard shows a file error:** start it from the project root folder (the one containing `app.py`).