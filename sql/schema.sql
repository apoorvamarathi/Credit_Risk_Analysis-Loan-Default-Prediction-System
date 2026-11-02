-- ====================================================================
-- CREDIT RISK & LOAN DEFAULT PREDICTION SYSTEM
-- DATABASE SCHEMA DEFINITION (SQLite)
-- Table: loan_applications
-- ====================================================================

CREATE TABLE IF NOT EXISTS loan_applications (
    loan_id                     TEXT PRIMARY KEY,
    person_age                  INTEGER NOT NULL,
    person_income               REAL NOT NULL,
    income_group                TEXT NOT NULL,
    age_group                   TEXT NOT NULL,
    person_home_ownership       TEXT NOT NULL,     -- 'RENT', 'OWN', 'MORTGAGE', 'OTHER'
    person_emp_length           REAL NOT NULL,     -- In years
    loan_intent                 TEXT NOT NULL,     -- Purpose: 'PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'
    loan_grade                  TEXT NOT NULL,     -- Credit Risk Tier: 'A', 'B', 'C', 'D', 'E', 'F', 'G'
    loan_amnt                   REAL NOT NULL,     -- Loan principal amount ($)
    loan_int_rate               REAL NOT NULL,     -- Annual interest rate (%)
    loan_status                 INTEGER NOT NULL,  -- 0: Non-default, 1: Default
    loan_percent_income         REAL NOT NULL,     -- Ratio of loan amount to applicant income
    debt_burden_ratio           REAL NOT NULL,     -- Total loan debt divided by income
    cb_person_default_on_file   TEXT NOT NULL,     -- Historical default recorded: 'Y' or 'N'
    cb_person_cred_hist_length  INTEGER NOT NULL,  -- Credit history duration in years
    application_date            TEXT NOT NULL,     -- 'YYYY-MM-DD'
    application_year            INTEGER NOT NULL,  -- e.g. 2022, 2023
    application_month           TEXT NOT NULL,     -- 'YYYY-MM'
    default_label               TEXT NOT NULL      -- Human-readable outcome
);

-- Strategic analytical indices for query acceleration
CREATE INDEX IF NOT EXISTS idx_loan_id ON loan_applications(loan_id);
CREATE INDEX IF NOT EXISTS idx_loan_status ON loan_applications(loan_status);
CREATE INDEX IF NOT EXISTS idx_loan_intent ON loan_applications(loan_intent);
CREATE INDEX IF NOT EXISTS idx_loan_grade ON loan_applications(loan_grade);
CREATE INDEX IF NOT EXISTS idx_income_group ON loan_applications(income_group);
CREATE INDEX IF NOT EXISTS idx_app_month ON loan_applications(application_month);
CREATE INDEX IF NOT EXISTS idx_home_ownership ON loan_applications(person_home_ownership);
