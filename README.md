# Credit Risk Analysis and Loan Default Prediction System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-EB5424.svg)](https://xgboost.ai/)
[![SQLite](https://img.shields.io/badge/SQLite-DataWarehouse-003B57.svg)](https://www.sqlite.org/)
[![Tests Passed](https://img.shields.io/badge/Tests-4%2F4%20Passing-success.svg)](https://docs.pytest.org/)

An enterprise-grade banking intelligence platform and machine learning decision engine that analyzes historical loan portfolios, uncovers default risk patterns, executes SQL business intelligence, and delivers real-time default probability predictions.

---

## 📌 Executive Summary

* **Total Portfolio Volume Analyzed:** $310,876,400 across 32,408 originated loans.
* **Portfolio Baseline Default Rate:** 21.87% ($76.93M in capital at risk).
* **Champion Model:** **XGBoost Classifier** achieving **0.9457 ROC-AUC**, **94.08% Precision**, and **93.52% Accuracy** on unseen holdout test data.
* **Architecture:** Zero-leakage stratified training, ColumnTransformer pipeline, indexed SQLite database warehouse, and an 8-page responsive Streamlit dashboard.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Data Processing & Analytics** | Python 3.12, Pandas, NumPy |
| **Data Storage & Warehouse** | SQLite 3, B-Tree Indexed Tables |
| **Machine Learning** | Scikit-Learn, XGBoost, Joblib |
| **Data Visualization** | Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib |
| **Web Application & UI** | Streamlit, Custom High-End Banking CSS |
| **Validation & Quality Assurance** | Pytest, Stratified 5-Fold Cross Validation |

---

## 📂 Project Directory Structure

```
Credit_Risk_Analysis/
├── data/
│   ├── raw/
│   │   └── credit_risk_dataset.csv     <- Real benchmark credit dataset (32,581 records)
│   └── processed/
│       ├── cleaned_loans.csv           <- Cleaned & feature-engineered dataset (32,408 records)
│       └── credit_risk.db              <- SQLite Data Warehouse with analytical indices
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py                <- Deduplication, imputation, outlier detection
│   ├── database.py                     <- SQLite connection manager & query suite
│   ├── train_models.py                 <- 4-model trainer, CV, ROC/PR curves & serializer
│   └── predict.py                      <- Real-time underwriting inference engine
├── models/
│   ├── preprocessor.joblib             <- ColumnTransformer fitted on train fold
│   ├── xgboost.joblib                  <- Serialized champion XGBoost model
│   ├── random_forest.joblib            <- Serialized Random Forest model
│   ├── decision_tree.joblib            <- Serialized Decision Tree model
│   ├── logistic_regression.joblib      <- Serialized Logistic Regression model
│   ├── model_metrics.json              <- Full test evaluation & curve coordinates
│   ├── feature_importance.json         <- Top global predictive feature rankings
│   └── feature_metadata.json           <- Categorical encoders & numerical schemas
├── sql/
│   ├── schema.sql                      <- Complete DDL schema with primary keys & indices
│   └── queries.sql                     <- Production SQL analytics suite
├── tests/
│   └── test_pipeline.py                <- Pytest automated test suite (100% pass)
├── app.py                              <- 8-Page enterprise Streamlit dashboard
├── requirements.txt                    <- Locked production dependencies
├── ARCHITECTURE.md                     <- Engineering & system architecture specifications
└── README.md                           <- Quickstart & comprehensive documentation
```

---

## ⚙️ 1. Data Cleaning & Feature Engineering

1. **Deduplication:** Stripped out 165 duplicate records.
2. **Biological & Data Entry Outlier Handling:**
   - Filtered erroneous age entries (`person_age > 90` removed anomalous ages 123 and 144).
   - Removed impossible employment durations where `person_emp_length > (person_age - 14)` or `> 50` years.
3. **Leakage-Free Imputation Strategy:**
   - Missing loan interest rates (`loan_int_rate`) imputed using the median rate by `loan_grade`, preserving authentic institutional pricing structures.
   - Missing employment lengths (`person_emp_length`) imputed using age-cohort medians.
4. **Feature Engineering:**
   - `income_group`: Brackets (`< $25k`, `$25k-$50k`, `$50k-$75k`, `$75k-$100k`, `$100k+`).
   - `age_group`: Demographic tiers (`18-25`, `26-35`, `36-50`, `50+`).
   - `debt_burden_ratio`: Loan amount divided by total annual income.
   - `application_date` / `application_month`: Deterministic time series origination timestamps across 2022–2023 for monthly trend surveillance.

---

## 📈 2. Exploratory Data Analysis (Key Findings)

* **Loan-to-Income Sensitivity:** Default rates stay below 12% when loan amount is under 20% of income, but jump to over 58% when the loan exceeds 35% of income.
* **Credit Grade Divergence:** Grade A borrowers default at only 9.9%, whereas Grade G borrowers experience an 89.2% default rate.
* **Loan Purpose Vulnerability:** Borrowing for **Debt Consolidation** (28.3% default) and **Medical emergencies** (26.7% default) presents significantly higher credit impairment than Education (17.2%) or Venture financing (16.8%).
* **Bureau Default Record:** Applicants with prior defaults on record exhibit a 2.3x higher likelihood of subsequent default.

---

## 🗄️ 3. SQL Analytics Suite

All business queries run against `data/processed/credit_risk.db`. Predefined queries include:

1. **Total Loan Applications:**
   ```sql
   SELECT COUNT(*) AS total_loan_applications FROM loan_applications;
   ```
2. **Total Loan Amount:**
   ```sql
   SELECT ROUND(SUM(loan_amnt), 2) AS total_funded_amount, ROUND(AVG(loan_amnt), 2) AS avg_ticket_size FROM loan_applications;
   ```
3. **Default Percentage & Portfolio Loss:**
   ```sql
   SELECT 
       COUNT(*) AS total_loans,
       SUM(loan_status) AS total_defaults,
       ROUND(AVG(loan_status) * 100.0, 2) AS default_percentage,
       ROUND(SUM(CASE WHEN loan_status = 1 THEN loan_amnt ELSE 0 END), 2) AS total_defaulted_amount
   FROM loan_applications;
   ```
4. **Average Loan & Income Metrics:**
   ```sql
   SELECT ROUND(AVG(loan_amnt), 2) AS average_loan_amount, ROUND(AVG(person_income), 2) AS average_applicant_income FROM loan_applications;
   ```
5. **Default Rate by Income Group:**
   ```sql
   SELECT 
       income_group,
       COUNT(*) AS total_applications,
       SUM(loan_status) AS default_count,
       ROUND(AVG(loan_status) * 100.0, 2) AS default_rate_pct
   FROM loan_applications
   GROUP BY income_group
   ORDER BY default_rate_pct DESC;
   ```
6. **Default Rate by Loan Purpose:**
   ```sql
   SELECT 
       loan_intent AS loan_purpose,
       COUNT(*) AS total_applications,
       ROUND(AVG(loan_status) * 100.0, 2) AS default_rate_pct,
       ROUND(SUM(loan_amnt), 2) AS total_funded_volume
   FROM loan_applications
   GROUP BY loan_intent
   ORDER BY default_rate_pct DESC;
   ```
7. **Monthly Loan Trends:**
   ```sql
   SELECT 
       application_month,
       COUNT(*) AS monthly_applications,
       ROUND(SUM(loan_amnt), 2) AS total_funded_volume,
       ROUND(AVG(loan_status) * 100.0, 2) AS default_rate_pct
   FROM loan_applications
   GROUP BY application_month
   ORDER BY application_month ASC;
   ```

*(See `sql/queries.sql` for all 10 optimized queries with explainability comments).*

---

## 🤖 4. Machine Learning Benchmark & Validation

### Validation Strategy:
* **Stratified 80/20 Holdout Split:** 25,926 training records / 6,482 test records.
* **Stratified 5-Fold Cross Validation:** Validated across training folds to measure model generalization variance.
* **Zero Data Leakage:** Preprocessors and scalers fitted strictly on training data.

### Test Set Performance Comparison:

| Model | Accuracy | Precision | Recall | F1-Score | Holdout ROC-AUC | 5-Fold CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **93.52%** | **94.08%** | **75.11%** | **0.8353** | **0.9457** | **0.9406 (±0.005)** |
| **Random Forest** | 93.21% | 95.87% | 72.07% | 0.8229 | 0.9275 | 0.9251 (±0.005) |
| **Decision Tree** | 88.94% | 74.09% | 76.02% | 0.7504 | 0.9024 | 0.8980 (±0.006) |
| **Logistic Regression** | 80.65% | 54.00% | 78.07% | 0.6384 | 0.8715 | 0.8705 (±0.006) |

---

## 🖥️ 5. Interactive Dashboard Pages

1. **Executive Overview:** Portfolio KPI cards ($310.8M funded, 21.87% default rate, $76.9M at risk), monthly volume vs default rate dual-axis trend chart, loan grade exposure, and loan purpose breakdown.
2. **Customer Analytics:** Income distribution, age cohort default rates, employment tenure trend lines, and home ownership cross-tabulation.
3. **Loan Portfolio Analysis:** Ticket size box plots, loan size vs interest rate scatter plots, and leverage (DTI) distribution.
4. **Credit Risk Analysis:** Prior default multiplier charts, Grade × Income risk heatmap matrix, and Pearson correlation heatmaps.
5. **SQL Analytics Console:** Live interactive query executor for all 10 predefined business queries, plus a custom read-only SQL terminal.
6. **ML Model Performance:** Side-by-side metric tables, interactive ROC curves, Precision-Recall curves, confusion matrices with financial loss interpretation, and global feature importance rankings.
7. **Individual Loan Prediction:** Real-time applicant underwriting calculator:
   - Dynamic default probability gauge meter.
   - Risk Tier categorization (`Low Risk`, `Medium Risk`, `High Risk`).
   - Granular breakdown of positive and negative risk factors.
   - Tailored underwriting actions (e.g. required collateral, co-signer, restructured principal).
8. **Business Recommendations:** Underwriting cut-off policy matrix, debt-to-income caps, risk-based pricing schedules, and concentration limits.

---

## 🚀 6. Installation & Local Execution Guide

### Prerequisites
* Python 3.10+ (Recommended: Python 3.12)
* Git

### Step-by-Step Setup

```bash
# 1. Navigate to the project root directory
cd C:\Users\Admin\Credit_Risk_Analysis

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Re-run the data cleaning & feature engineering pipeline
python src/data_cleaning.py

# 5. (Optional) Re-train and evaluate all machine learning models
python src/train_models.py

# 6. Run the automated test suite
python -m pytest tests/test_pipeline.py -v

# 7. Launch the Streamlit Interactive Dashboard
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 🧪 7. Automated Testing Suite

To execute the end-to-end integration test suite:

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected output:
```
tests/test_pipeline.py::test_cleaned_data_exists_and_valid PASSED        [ 25%]
tests/test_pipeline.py::test_sqlite_database_and_queries PASSED          [ 50%]
tests/test_pipeline.py::test_trained_models_and_metrics PASSED           [ 75%]
tests/test_pipeline.py::test_inference_engine PASSED                     [100%]
============================== 4 passed in ~6.5s ==============================
```

---

## 📋 8. Enterprise Underwriting Guidelines

| Risk Tier | Default Probability | Policy Directive | Pricing & Underwriting Action |
| :---: | :---: | :---: | :--- |
| **Low Risk** | $< 20\%$ | **Straight-Through Approval** | Prime rates (7.0%–11.0%), fast-track automated disbursement, pre-approval for credit limits. |
| **Medium Risk** | $20\% - 45\%$ | **Conditional Approval** | Mandatory paystubs/W2 verification, cap loan at 25% of income, apply +1.5% APR risk premium, ACH auto-debit required. |
| **High Risk** | $\ge 45\%$ | **Automated Decline / Committee Review** | Require creditworthy co-signer (Grade A/B), 25% cash collateral/secured lien, direct payoff to third-party creditors for debt consolidation. |
