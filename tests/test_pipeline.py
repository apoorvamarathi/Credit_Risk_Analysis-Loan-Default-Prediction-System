"""
Unit and Integration Test Suite
Validates the entire Credit Risk Analysis & Loan Default Prediction Pipeline.
"""

import os
import json
import sqlite3
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Cross-platform project directory resolution
PROJECT_DIR = Path(__file__).resolve().parent.parent


def test_cleaned_data_exists_and_valid():
    """Verify cleaned CSV exists, has no nulls, and no duplicate records."""
    csv_path = PROJECT_DIR / "data" / "processed" / "cleaned_loans.csv"
    assert csv_path.exists(), "Cleaned CSV data file does not exist!"
    
    df = pd.read_csv(csv_path)
    assert len(df) > 30000, f"Expected >30,000 rows, found {len(df)}"
    assert df.duplicated().sum() == 0, "Cleaned dataset contains duplicates!"
    assert df.isnull().sum().sum() == 0, "Cleaned dataset contains unexpected null values!"
    assert df['person_age'].max() <= 90, "Age outlier > 90 detected in cleaned data!"
    assert df['person_emp_length'].max() <= 50, "Employment length outlier > 50 detected!"


def test_sqlite_database_and_queries():
    """Verify SQLite database exists and executes all required analytical queries."""
    from src.database import CreditRiskDB
    
    db = CreditRiskDB()
    kpis = db.get_portfolio_kpis()
    assert kpis['total_applications'] > 30000
    assert kpis['total_funded'] > 100000000  # Over $100M funded
    assert 15 <= kpis['default_rate'] <= 30  # Default rate around ~21.8%
    
    predefined = db.get_predefined_queries()
    required_queries = [
        "Total Loan Applications",
        "Total Loan Amount & Sizing",
        "Default Percentage & Impairment",
        "Average Loan & Income Metrics",
        "Default Rate by Income Group",
        "Default Rate by Loan Purpose",
        "Monthly Loan Trends"
    ]
    for q_name in required_queries:
        assert q_name in predefined, f"Missing required query: {q_name}"
        res = db.execute_query(predefined[q_name])
        assert not res.empty, f"Query '{q_name}' returned empty result set!"


def test_database_auto_initialization():
    """Verify that CreditRiskDB automatically initializes from CSV if database is missing."""
    import gc
    from src.database import CreditRiskDB
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_db_path = Path(tmp_dir) / "test_auto_init.db"
        assert not temp_db_path.exists()
        
        # Instantiate DB pointing to non-existent file
        db = CreditRiskDB(db_path=temp_db_path)
        assert temp_db_path.exists(), "Database file was not auto-created!"
        
        # Verify tables and rows exist
        df_count = db.execute_query("SELECT COUNT(*) AS total FROM loan_applications;")
        assert df_count.iloc[0]['total'] > 30000, "Auto-initialized database has insufficient records!"
        
        # Clean up reference and force garbage collection before temp dir cleanup
        del db
        gc.collect()


def test_trained_models_and_metrics():
    """Verify all 4 models were trained, evaluated, and metrics persisted."""
    metrics_path = PROJECT_DIR / "models" / "model_metrics.json"
    assert metrics_path.exists(), "Model metrics JSON not found!"
    
    with open(metrics_path, "r") as f:
        data = json.load(f)
    
    metrics = data["metrics"]
    required_models = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
    for m in required_models:
        assert m in metrics, f"Model {m} missing from metrics!"
        assert metrics[m]["accuracy"] > 0.75, f"{m} accuracy below 75%"
        assert metrics[m]["roc_auc"] > 0.80, f"{m} ROC-AUC below 0.80"
        assert len(metrics[m]["confusion_matrix"]) == 2, f"{m} invalid confusion matrix"


def test_inference_engine():
    """Verify real-time prediction engine returns valid probabilities and categories."""
    from src.predict import CreditRiskPredictor
    
    predictor = CreditRiskPredictor()
    
    # Low risk applicant
    low_risk_applicant = {
        'person_age': 35,
        'person_income': 95000,
        'person_emp_length': 10,
        'loan_amnt': 8000,
        'loan_int_rate': 7.5,
        'loan_percent_income': 0.08,
        'cb_person_cred_hist_length': 12,
        'person_home_ownership': 'MORTGAGE',
        'loan_intent': 'VENTURE',
        'loan_grade': 'A',
        'cb_person_default_on_file': 'N'
    }
    low_res = predictor.predict_applicant(low_risk_applicant, model_name='XGBoost')
    assert low_res['default_probability'] < 0.25, f"Expected low default prob, got {low_res['default_probability']}"
    assert low_res['risk_tier'] in ['Low Risk', 'Medium Risk']
    assert len(low_res['key_factors']) > 0
    assert len(low_res['suggested_actions']) > 0

    # High risk applicant
    high_risk_applicant = {
        'person_age': 21,
        'person_income': 18000,
        'person_emp_length': 0.5,
        'loan_amnt': 15000,
        'loan_int_rate': 19.5,
        'loan_percent_income': 0.83,
        'cb_person_cred_hist_length': 2,
        'person_home_ownership': 'RENT',
        'loan_intent': 'DEBTCONSOLIDATION',
        'loan_grade': 'G',
        'cb_person_default_on_file': 'Y'
    }
    high_res = predictor.predict_applicant(high_risk_applicant, model_name='XGBoost')
    assert high_res['default_probability'] > 0.70, f"Expected high default prob, got {high_res['default_probability']}"
    assert high_res['risk_tier'] == 'High Risk'
    assert "Decline" in high_res['approval_recommendation']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
