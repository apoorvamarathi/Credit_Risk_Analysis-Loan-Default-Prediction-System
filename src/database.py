"""
Database Connection and Query Execution Module
Provides programmatic and interactive access to the credit risk SQLite database.
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional
import pandas as pd

DEFAULT_DB_PATH = r"C:\Users\Admin\Credit_Risk_Analysis\data\processed\credit_risk.db"


class CreditRiskDB:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}. Run data_cleaning.py first.")

    def get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database with Row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Executes a SQL SELECT query and returns the result as a pandas DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            if params:
                return pd.read_sql_query(query, conn, params=params)
            return pd.read_sql_query(query, conn)

    def get_portfolio_kpis(self) -> Dict[str, Any]:
        """Calculates executive portfolio KPIs via SQL."""
        query = """
        SELECT 
            COUNT(*) AS total_applications,
            SUM(loan_amnt) AS total_funded,
            AVG(loan_amnt) AS avg_loan_size,
            SUM(loan_status) AS total_defaults,
            AVG(loan_status) * 100.0 AS default_rate,
            AVG(loan_int_rate) AS avg_interest_rate,
            AVG(person_income) AS avg_applicant_income,
            SUM(CASE WHEN loan_status = 1 THEN loan_amnt ELSE 0 END) AS capital_at_risk
        FROM loan_applications;
        """
        df = self.execute_query(query)
        return df.to_dict(orient='records')[0]

    def get_predefined_queries(self) -> Dict[str, str]:
        """Returns the suite of pre-configured business SQL queries."""
        return {
            "Total Loan Applications": "SELECT COUNT(*) AS total_loan_applications FROM loan_applications;",
            
            "Total Loan Amount & Sizing": """
                SELECT 
                    ROUND(SUM(loan_amnt), 2) AS total_funded_amount,
                    ROUND(AVG(loan_amnt), 2) AS avg_ticket_size,
                    ROUND(MIN(loan_amnt), 2) AS min_loan_amount,
                    ROUND(MAX(loan_amnt), 2) AS max_loan_amount
                FROM loan_applications;
            """,
            
            "Default Percentage & Impairment": """
                SELECT 
                    COUNT(*) AS total_loans,
                    SUM(loan_status) AS total_defaults,
                    COUNT(*) - SUM(loan_status) AS total_performing_loans,
                    ROUND(AVG(loan_status) * 100.0, 2) AS default_percentage,
                    ROUND(SUM(CASE WHEN loan_status = 1 THEN loan_amnt ELSE 0 END), 2) AS total_defaulted_amount,
                    ROUND(SUM(CASE WHEN loan_status = 1 THEN loan_amnt ELSE 0 END) * 100.0 / SUM(loan_amnt), 2) AS capital_at_loss_pct
                FROM loan_applications;
            """,
            
            "Average Loan & Income Metrics": """
                SELECT 
                    ROUND(AVG(loan_amnt), 2) AS average_loan_amount,
                    ROUND(AVG(person_income), 2) AS average_applicant_income,
                    ROUND(AVG(loan_percent_income) * 100.0, 2) AS avg_loan_to_income_pct
                FROM loan_applications;
            """,
            
            "Default Rate by Income Group": """
                SELECT 
                    income_group,
                    COUNT(*) AS total_applications,
                    SUM(loan_status) AS default_count,
                    ROUND(AVG(loan_status) * 100.0, 2) AS default_rate_pct,
                    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
                    ROUND(AVG(loan_int_rate), 2) AS avg_interest_rate,
                    ROUND(SUM(loan_amnt), 2) AS total_loan_volume
                FROM loan_applications
                GROUP BY income_group
                ORDER BY 
                    CASE income_group
                        WHEN '< $25k' THEN 1
                        WHEN '$25k-$50k' THEN 2
                        WHEN '$50k-$75k' THEN 3
                        WHEN '$75k-$100k' THEN 4
                        ELSE 5
                    END;
            """,
            
            "Default Rate by Loan Purpose": """
                SELECT 
                    loan_intent AS loan_purpose,
                    COUNT(*) AS total_applications,
                    SUM(loan_status) AS default_count,
                    ROUND(AVG(loan_status) * 100.0, 2) AS default_rate_pct,
                    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
                    ROUND(AVG(loan_int_rate), 2) AS avg_interest_rate,
                    ROUND(SUM(loan_amnt), 2) AS total_funded_volume
                FROM loan_applications
                GROUP BY loan_intent
                ORDER BY default_rate_pct DESC;
            """,
            
            "Monthly Loan Trends": """
                SELECT 
                    application_month,
                    COUNT(*) AS monthly_applications,
                    ROUND(SUM(loan_amnt), 2) AS total_funded_volume,
                    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
                    SUM(loan_status) AS default_count,
                    ROUND(AVG(loan_status) * 100.0, 2) AS default_rate_pct
                FROM loan_applications
                GROUP BY application_month
                ORDER BY application_month ASC;
            """,
            
            "Default Rate by Loan Grade": """
                SELECT 
                    loan_grade,
                    COUNT(*) AS total_loans,
                    SUM(loan_status) AS defaults,
                    ROUND(AVG(loan_status) * 100.0, 2) AS default_rate_pct,
                    ROUND(AVG(loan_int_rate), 2) AS avg_interest_rate,
                    ROUND(AVG(loan_amnt), 2) AS avg_loan_size,
                    ROUND(SUM(loan_amnt), 2) AS total_exposure
                FROM loan_applications
                GROUP BY loan_grade
                ORDER BY loan_grade ASC;
            """,
            
            "Default Rate by Home Ownership": """
                SELECT 
                    person_home_ownership,
                    COUNT(*) AS total_applicants,
                    SUM(loan_status) AS defaults,
                    ROUND(AVG(loan_status) * 100.0, 2) AS default_rate_pct,
                    ROUND(AVG(person_income), 2) AS avg_income,
                    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount
                FROM loan_applications
                GROUP BY person_home_ownership
                ORDER BY default_rate_pct DESC;
            """,
            
            "High Risk Borrowers (Prior Default + DTI > 30%)": """
                SELECT 
                    loan_id,
                    person_age,
                    person_income,
                    loan_amnt,
                    ROUND(loan_percent_income * 100, 1) AS loan_pct_income,
                    loan_intent,
                    loan_grade,
                    cb_person_default_on_file,
                    loan_status
                FROM loan_applications
                WHERE cb_person_default_on_file = 'Y' 
                  AND loan_percent_income > 0.30
                ORDER BY loan_amnt DESC
                LIMIT 20;
            """
        }


if __name__ == "__main__":
    db = CreditRiskDB()
    print("Testing DB connection and sample queries...")
    kpis = db.get_portfolio_kpis()
    print("Portfolio KPIs:")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
    
    print("\nExecuting Default Rate by Loan Purpose:")
    df_purpose = db.execute_query(db.get_predefined_queries()["Default Rate by Loan Purpose"])
    print(df_purpose)
