"""
Data Cleaning and Feature Engineering Module
Part of Credit Risk Analysis and Loan Default Prediction System
"""

import os
import sqlite3
import numpy as np
import pandas as pd


def get_income_group(income: float) -> str:
    """Categorize annual income into standard banking income tiers."""
    if income < 25000:
        return '< $25k'
    elif income < 50000:
        return '$25k-$50k'
    elif income < 75000:
        return '$50k-$75k'
    elif income < 100000:
        return '$75k-$100k'
    else:
        return '$100k+'


def get_age_group(age: int) -> str:
    """Categorize age into demographic groups."""
    if age <= 25:
        return '18-25'
    elif age <= 35:
        return '26-35'
    elif age <= 50:
        return '36-50'
    else:
        return '50+'


def clean_and_prepare_data(
    raw_csv_path: str = r"C:\Users\Admin\Credit_Risk_Analysis\data\raw\credit_risk_dataset.csv",
    output_csv_path: str = r"C:\Users\Admin\Credit_Risk_Analysis\data\processed\cleaned_loans.csv",
    db_path: str = r"C:\Users\Admin\Credit_Risk_Analysis\data\processed\credit_risk.db"
) -> pd.DataFrame:
    """
    Executes full data cleaning, outlier resolution, imputation, 
    feature engineering, and SQLite database loading.
    """
    print(f"Loading raw dataset from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    initial_rows = len(df)
    print(f"Initial record count: {initial_rows:,}")

    # 1. Remove duplicate records
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Deduplication: removed {dup_count:,} duplicate records.")

    # 2. Handle demographic and employment outliers
    # Biological / realistic constraints: Age <= 90, Employment Length <= 50 and <= (age - 14)
    valid_mask = (
        (df['person_age'] >= 18) &
        (df['person_age'] <= 90) &
        (df['person_income'] > 0) &
        (df['person_emp_length'].isna() | (
            (df['person_emp_length'] <= 50) & 
            (df['person_emp_length'] <= (df['person_age'] - 14))
        ))
    )
    outliers_removed = len(df) - valid_mask.sum()
    df = df[valid_mask].copy().reset_index(drop=True)
    print(f"Outlier removal: removed {outliers_removed:,} anomalous records.")

    # 3. Handle Missing Values
    # Impute missing loan interest rate using median interest rate per loan grade
    grade_median_int = df.groupby('loan_grade')['loan_int_rate'].median().to_dict()
    df['loan_int_rate'] = df.apply(
        lambda row: grade_median_int.get(row['loan_grade'], 11.0)
        if pd.isna(row['loan_int_rate']) else row['loan_int_rate'],
        axis=1
    )

    # Impute missing employment length using median by age group
    df['temp_age_bin'] = pd.cut(df['person_age'], bins=[17, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
    emp_median_map = df.groupby('temp_age_bin', observed=False)['person_emp_length'].median().to_dict()
    df['person_emp_length'] = df.apply(
        lambda row: emp_median_map.get(row['temp_age_bin'], 4.0)
        if pd.isna(row['person_emp_length']) else row['person_emp_length'],
        axis=1
    )
    df.drop(columns=['temp_age_bin'], inplace=True)

    # 4. Correct Data Types
    df['person_age'] = df['person_age'].astype(int)
    df['person_income'] = df['person_income'].astype(float)
    df['person_emp_length'] = df['person_emp_length'].round(1)
    df['loan_amnt'] = df['loan_amnt'].astype(float)
    df['loan_int_rate'] = df['loan_int_rate'].round(2)
    df['loan_status'] = df['loan_status'].astype(int)
    df['loan_percent_income'] = df['loan_percent_income'].round(3)
    df['cb_person_cred_hist_length'] = df['cb_person_cred_hist_length'].astype(int)

    # 5. Feature Engineering
    df['loan_id'] = [f"LN-{10001 + i}" for i in range(len(df))]
    df['income_group'] = df['person_income'].apply(get_income_group)
    df['age_group'] = df['person_age'].apply(get_age_group)
    df['debt_burden_ratio'] = (df['loan_amnt'] / df['person_income']).round(4)
    
    # Generate realistic application dates distributed across 2022-2023 for monthly trend analysis
    np.random.seed(42)
    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2023-12-31')
    days_range = (end_date - start_date).days
    random_days = np.random.randint(0, days_range + 1, size=len(df))
    app_dates = start_date + pd.to_timedelta(random_days, unit='D')
    
    df['application_date'] = app_dates.strftime('%Y-%m-%d')
    df['application_year'] = app_dates.year
    df['application_month'] = app_dates.strftime('%Y-%m')

    # Status labels for dashboards
    df['default_label'] = df['loan_status'].map({0: 'Non-Default (Current/Paid)', 1: 'Defaulted'})

    # 6. Save Cleaned Dataset to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Cleaned data successfully saved to: {output_csv_path} ({len(df):,} records)")

    # 7. Export to SQLite Database
    export_to_sqlite(df, db_path)

    return df


def export_to_sqlite(df: pd.DataFrame, db_path: str):
    """Stores the cleaned data into an indexed SQLite database."""
    print(f"Exporting data to SQLite database at {db_path}...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop existing table if needed
    cursor.execute("DROP TABLE IF EXISTS loan_applications;")

    # Write dataframe to table
    df.to_sql('loan_applications', conn, if_exists='replace', index=False)

    # Create strategic indices for high-performance SQL analytical queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_loan_id ON loan_applications(loan_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_loan_status ON loan_applications(loan_status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_loan_intent ON loan_applications(loan_intent);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_loan_grade ON loan_applications(loan_grade);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_income_group ON loan_applications(income_group);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_app_month ON loan_applications(application_month);")

    conn.commit()
    conn.close()
    print("SQLite database created and indexed successfully.")


if __name__ == "__main__":
    clean_and_prepare_data()
