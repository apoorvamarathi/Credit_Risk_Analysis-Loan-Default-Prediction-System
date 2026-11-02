-- ====================================================================
-- CREDIT RISK & LOAN DEFAULT PREDICTION SYSTEM
-- PRODUCTION SQL ANALYTICS SUITE
-- Database: credit_risk.db
-- Table: loan_applications
-- ====================================================================

-- --------------------------------------------------------------------
-- QUERY 1: TOTAL LOAN APPLICATIONS
-- Purpose: Calculate total volume of processed loan applications.
-- --------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_loan_applications
FROM loan_applications;


-- --------------------------------------------------------------------
-- QUERY 2: TOTAL LOAN AMOUNT (FUNDED CAPITAL)
-- Purpose: Aggregate total dollar capital disbursed across portfolio.
-- --------------------------------------------------------------------
SELECT 
    ROUND(SUM(loan_amnt), 2) AS total_funded_amount,
    ROUND(AVG(loan_amnt), 2) AS avg_ticket_size,
    ROUND(MIN(loan_amnt), 2) AS min_loan_amount,
    ROUND(MAX(loan_amnt), 2) AS max_loan_amount
FROM loan_applications;


-- --------------------------------------------------------------------
-- QUERY 3: DEFAULT PERCENTAGE (PORTFOLIO DEFAULT RATE)
-- Purpose: Measure high-level credit impairment across the book.
-- --------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_loans,
    SUM(loan_status) AS total_defaults,
    COUNT(*) - SUM(loan_status) AS total_performing_loans,
    ROUND(AVG(loan_status) * 100.0, 2) AS default_percentage,
    ROUND(SUM(CASE WHEN loan_status = 1 THEN loan_amnt ELSE 0 END), 2) AS total_defaulted_amount,
    ROUND(SUM(CASE WHEN loan_status = 1 THEN loan_amnt ELSE 0 END) * 100.0 / SUM(loan_amnt), 2) AS capital_at_loss_pct
FROM loan_applications;


-- --------------------------------------------------------------------
-- QUERY 4: AVERAGE LOAN AMOUNT
-- Purpose: Baseline sizing metric for credit committee evaluations.
-- --------------------------------------------------------------------
SELECT 
    ROUND(AVG(loan_amnt), 2) AS average_loan_amount,
    ROUND(AVG(person_income), 2) AS average_applicant_income,
    ROUND(AVG(loan_percent_income) * 100.0, 2) AS avg_loan_to_income_pct
FROM loan_applications;


-- --------------------------------------------------------------------
-- QUERY 5: DEFAULT RATE BY INCOME GROUP
-- Purpose: Assess vulnerability of lower-income cohorts vs high-earners.
-- --------------------------------------------------------------------
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


-- --------------------------------------------------------------------
-- QUERY 6: DEFAULT RATE BY LOAN PURPOSE (LOAN INTENT)
-- Purpose: Identify high-risk borrowing categories (e.g., Medical, Debt Consolidation).
-- --------------------------------------------------------------------
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


-- --------------------------------------------------------------------
-- QUERY 7: MONTHLY LOAN TRENDS
-- Purpose: Time-series analysis of origination volume, disbursed capital, and default rate.
-- --------------------------------------------------------------------
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


-- --------------------------------------------------------------------
-- BONUS QUERY 8: RISK MATRIX BY CREDIT GRADE
-- Purpose: Benchmark credit bureau grades against realized default frequency.
-- --------------------------------------------------------------------
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


-- --------------------------------------------------------------------
-- BONUS QUERY 9: HOME OWNERSHIP RISK PROFILING
-- Purpose: Contrast default probability between renters, mortgagees, and owners.
-- --------------------------------------------------------------------
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


-- --------------------------------------------------------------------
-- BONUS QUERY 10: HIGH RISK SEGMENT DETECTION
-- Purpose: Surface borrowers with both prior default flag and >30% loan-to-income.
-- --------------------------------------------------------------------
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
