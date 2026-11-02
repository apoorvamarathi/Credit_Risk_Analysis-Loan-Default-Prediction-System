"""
Real-Time Inference and Risk Decision Engine
Provides default probability, risk tiering, feature attribution, and underwriting actions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

MODELS_DIR = r"C:\Users\Admin\Credit_Risk_Analysis\models"


class CreditRiskPredictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        
        # Load preprocessor
        self.preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
        
        # Load metadata
        with open(os.path.join(models_dir, 'feature_metadata.json'), 'r') as f:
            self.metadata = json.load(f)
            
        # Load feature importance
        with open(os.path.join(models_dir, 'feature_importance.json'), 'r') as f:
            self.feature_importance = json.load(f)

        # Load models
        self.models = {
            'XGBoost': joblib.load(os.path.join(models_dir, 'xgboost.joblib')),
            'Random Forest': joblib.load(os.path.join(models_dir, 'random_forest.joblib')),
            'Decision Tree': joblib.load(os.path.join(models_dir, 'decision_tree.joblib')),
            'Logistic Regression': joblib.load(os.path.join(models_dir, 'logistic_regression.joblib'))
        }

    def predict_applicant(
        self,
        applicant_data: Dict[str, Any],
        model_name: str = 'XGBoost'
    ) -> Dict[str, Any]:
        """
        Calculates loan default probability, assigns risk tier, identifies driving factors,
        and generates personalized risk mitigation guidelines.
        """
        if model_name not in self.models:
            model_name = 'XGBoost'
        model = self.models[model_name]

        # Ensure loan_percent_income is calculated if not provided
        income = float(applicant_data.get('person_income', 50000))
        loan_amnt = float(applicant_data.get('loan_amnt', 10000))
        if 'loan_percent_income' not in applicant_data or applicant_data['loan_percent_income'] is None:
            loan_percent_income = round(loan_amnt / max(income, 1.0), 3)
            applicant_data['loan_percent_income'] = loan_percent_income
        else:
            loan_percent_income = float(applicant_data['loan_percent_income'])

        input_df = pd.DataFrame([applicant_data])
        input_proc = self.preprocessor.transform(input_df)

        prob_default = float(model.predict_proba(input_proc)[0, 1])
        prediction = int(prob_default >= 0.50)

        # Risk Tier Classification
        if prob_default < 0.20:
            risk_tier = "Low Risk"
            risk_color = "#10B981"  # Emerald green
            approval_recommendation = "Approved (Standard Underwriting)"
        elif prob_default < 0.45:
            risk_tier = "Medium Risk"
            risk_color = "#F59E0B"  # Amber
            approval_recommendation = "Conditional Approval (Enhanced Monitoring Required)"
        else:
            risk_tier = "High Risk"
            risk_color = "#EF4444"  # Crimson Red
            approval_recommendation = "Decline / Refer to Credit Committee"

        # Explainable Factors Analysis
        factors = self._analyze_risk_factors(applicant_data, prob_default)

        # Underwriting Actions
        suggested_actions = self._generate_underwriting_actions(
            risk_tier, applicant_data, prob_default
        )

        return {
            "model_used": model_name,
            "default_probability": round(prob_default, 4),
            "default_probability_pct": round(prob_default * 100, 2),
            "prediction": prediction,
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "approval_recommendation": approval_recommendation,
            "key_factors": factors,
            "suggested_actions": suggested_actions,
            "applicant_summary": {
                "loan_to_income_pct": round(loan_percent_income * 100, 1),
                "monthly_income": round(income / 12, 2),
                "interest_rate": float(applicant_data.get('loan_int_rate', 10.0)),
                "prior_default": applicant_data.get('cb_person_default_on_file', 'N')
            }
        }

    def _analyze_risk_factors(self, data: Dict[str, Any], prob: float) -> List[Dict[str, Any]]:
        """Identifies key drivers contributing towards default risk or creditworthiness."""
        factors = []
        
        # 1. Loan-to-income ratio
        lti = float(data.get('loan_percent_income', 0.2))
        if lti > 0.35:
            factors.append({
                "factor": "High Debt-to-Income Exposure",
                "impact": "Increases Risk",
                "direction": "negative",
                "detail": f"Loan represents {round(lti * 100, 1)}% of annual income (threshold: 30%)."
            })
        elif lti < 0.15:
            factors.append({
                "factor": "Conservative Debt-to-Income",
                "impact": "Lowers Risk",
                "direction": "positive",
                "detail": f"Loan amount is only {round(lti * 100, 1)}% of annual income."
            })

        # 2. Historical default record
        has_default = data.get('cb_person_default_on_file', 'N')
        if has_default == 'Y':
            factors.append({
                "factor": "Historical Credit Default on File",
                "impact": "Increases Risk",
                "direction": "negative",
                "detail": "Borrower has prior default registered with credit bureau."
            })
        else:
            factors.append({
                "factor": "Clean Credit History",
                "impact": "Lowers Risk",
                "direction": "positive",
                "detail": "No historical default record on credit bureau file."
            })

        # 3. Loan Grade
        grade = str(data.get('loan_grade', 'B'))
        if grade in ['D', 'E', 'F', 'G']:
            factors.append({
                "factor": f"Subprime Loan Grade ({grade})",
                "impact": "Increases Risk",
                "direction": "negative",
                "detail": f"Grade {grade} reflects elevated structural default probability."
            })
        elif grade in ['A', 'B']:
            factors.append({
                "factor": f"Prime Loan Grade ({grade})",
                "impact": "Lowers Risk",
                "direction": "positive",
                "detail": f"Grade {grade} indicates superior creditworthiness."
            })

        # 4. Employment stability
        emp = float(data.get('person_emp_length', 3))
        if emp >= 5:
            factors.append({
                "factor": f"Established Employment Tenure ({int(emp)} yrs)",
                "impact": "Lowers Risk",
                "direction": "positive",
                "detail": "Long-term employment tenure supports steady cash flows."
            })
        elif emp < 2:
            factors.append({
                "factor": f"Short Employment Tenure ({int(emp)} yrs)",
                "impact": "Increases Risk",
                "direction": "negative",
                "detail": "Under 2 years at current employer presents employment stability risk."
            })

        # 5. Home Ownership
        home = str(data.get('person_home_ownership', 'RENT'))
        if home == 'OWN':
            factors.append({
                "factor": "Home Ownership Asset Backing",
                "impact": "Lowers Risk",
                "direction": "positive",
                "detail": "Applicant outright owns residential property."
            })
        elif home == 'RENT':
            factors.append({
                "factor": "Renting Residential Status",
                "impact": "Moderate Risk",
                "direction": "neutral",
                "detail": "Applicant rents residential housing, recurring rental liabilities present."
            })

        return factors

    def _generate_underwriting_actions(
        self,
        risk_tier: str,
        data: Dict[str, Any],
        prob: float
    ) -> List[str]:
        """Prescribes tactical risk mitigation policies."""
        actions = []
        lti = float(data.get('loan_percent_income', 0.2))
        has_default = data.get('cb_person_default_on_file', 'N')

        if risk_tier == "Low Risk":
            actions.append("Fast-track approval: Release funds under standard prime origination terms.")
            actions.append("Offer preferential interest rate discount (-0.25% to -0.50%) to improve client retention.")
            actions.append("Qualify applicant for secondary pre-approved revolving credit products.")
        elif risk_tier == "Medium Risk":
            actions.append("Require verified income documentation: 3 consecutive months of bank statements and W-2 paystubs.")
            if lti > 0.25:
                suggested_loan = round(float(data.get('person_income', 50000)) * 0.22, -2)
                actions.append(f"Cap requested principal: Propose restructured loan size of ${suggested_loan:,.0f} to reduce debt-to-income.")
            actions.append("Apply risk-based pricing markup: Add +1.50% risk premium to base interest rate.")
            actions.append("Mandate automated recurring ACH debit repayments to minimize delinquency rates.")
        else: # High Risk
            actions.append("Decline automated disbursement: Escalate to Senior Credit Committee.")
            if has_default == 'Y':
                actions.append("Require a creditworthy co-signer / guarantor with credit grade A or B.")
            actions.append("Require cash collateral or secured lien of at least 25% of loan value.")
            if data.get('loan_intent') == 'DEBTCONSOLIDATION':
                actions.append("Direct-to-creditor disbursement: Pay off third-party creditors directly rather than releasing cash to borrower.")
            actions.append("Alternative product: Offer credit-builder secured loan with structured amortization schedule.")

        return actions


if __name__ == "__main__":
    predictor = CreditRiskPredictor()
    test_applicant = {
        'person_age': 28,
        'person_income': 65000,
        'person_emp_length': 4,
        'loan_amnt': 12000,
        'loan_int_rate': 11.2,
        'loan_percent_income': 0.18,
        'cb_person_cred_hist_length': 5,
        'person_home_ownership': 'RENT',
        'loan_intent': 'EDUCATION',
        'loan_grade': 'B',
        'cb_person_default_on_file': 'N'
    }
    res = predictor.predict_applicant(test_applicant, model_name='XGBoost')
    print("Inference Test Result:")
    print("Default Probability:", res['default_probability_pct'], "%")
    print("Risk Tier:", res['risk_tier'])
    print("Recommendation:", res['approval_recommendation'])
    print("Key Factors:")
    for f in res['key_factors']:
        print(f"  - {f['factor']}: {f['impact']} ({f['detail']})")
    print("Suggested Actions:")
    for a in res['suggested_actions']:
        print(f"  * {a}")
