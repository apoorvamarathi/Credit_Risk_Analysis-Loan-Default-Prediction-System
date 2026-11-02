"""
=============================================================================
CREDIT RISK ANALYSIS AND LOAN DEFAULT PREDICTION SYSTEM
Enterprise Banking Analytics Platform & Real-Time Decisioning Engine
=============================================================================
"""

import os
import sys
import json
import joblib
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pathlib import Path

# Add project root to sys.path (cross-platform)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "credit_risk_dataset.csv"
CLEANED_DATA_PATH = DATA_DIR / "processed" / "cleaned_loans.csv"
DB_PATH = DATA_DIR / "processed" / "credit_risk.db"
MODELS_DIR = PROJECT_ROOT / "models"

from src.data_cleaning import ensure_database_ready
from src.database import CreditRiskDB
from src.predict import CreditRiskPredictor

# ---------------------------------------------------------------------------
# STREAMLIT CONFIGURATION & CUSTOM STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ApexCredit | Credit Risk & Default Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Banking CSS
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Banner */
    .brand-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F284B 100%);
        border-radius: 12px;
        padding: 24px 30px;
        color: #F8FAFC;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    
    .brand-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .brand-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 6px;
        font-weight: 400;
    }
    
    /* KPI Card Component */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 18px 22px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 12px;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    
    .kpi-label {
        font-size: 12px;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    
    .kpi-subtext {
        font-size: 12px;
        color: #64748B;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .kpi-badge-danger {
        color: #DC2626;
        font-weight: 600;
    }
    
    .kpi-badge-success {
        color: #16A34A;
        font-weight: 600;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 20px;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Risk Badges */
    .badge-low {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #86EFAC;
    }
    
    .badge-medium {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #FCD34D;
    }
    
    .badge-high {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        border: 1px solid #FCA5A5;
    }
    
    /* Prediction Container */
    .decision-box {
        border-radius: 12px;
        padding: 24px;
        margin-top: 15px;
        border-left: 6px solid;
    }
    
    /* Code/SQL block styling */
    .stCodeBlock {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DATA & CACHING HELPERS (With Self-Healing Auto-Initialization)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cleaned_data():
    """Loads cleaned dataset, automatically building SQLite database or running pipeline if needed."""
    if not CLEANED_DATA_PATH.exists():
        try:
            ensure_database_ready(db_path=DB_PATH, cleaned_csv_path=CLEANED_DATA_PATH, raw_csv_path=RAW_DATA_PATH)
        except Exception as e:
            pass
            
    if CLEANED_DATA_PATH.exists():
        return pd.read_csv(CLEANED_DATA_PATH)
    elif RAW_DATA_PATH.exists():
        from src.data_cleaning import clean_and_prepare_data
        return clean_and_prepare_data(RAW_DATA_PATH, CLEANED_DATA_PATH, DB_PATH)
    else:
        st.error(
            "⚠️ **Required Dataset Not Found**\n\n"
            "Neither `data/processed/cleaned_loans.csv` nor `data/raw/credit_risk_dataset.csv` could be located.\n\n"
            "**How to resolve on Streamlit Cloud:**\n"
            "1. Make sure your `data/` directory is committed to GitHub.\n"
            "2. Verify that `.gitignore` does not exclude `cleaned_loans.csv` or `credit_risk_dataset.csv`.\n"
            "3. Push your repository and click 'Reboot App' in the Streamlit Cloud management console."
        )
        st.stop()

@st.cache_resource(show_spinner=False)
def get_db():
    return CreditRiskDB(db_path=DB_PATH)

@st.cache_resource(show_spinner=False)
def get_predictor():
    return CreditRiskPredictor(models_dir=MODELS_DIR)

@st.cache_data(show_spinner=False)
def load_ml_metrics():
    metrics_path = MODELS_DIR / "model_metrics.json"
    with open(metrics_path, "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_feature_importance():
    path = MODELS_DIR / "feature_importance.json"
    with open(path, "r") as f:
        return json.load(f)

# Load data with automatic database and dataset readiness
raw_df = load_cleaned_data()
db = get_db()
predictor = get_predictor()
ml_metrics = load_ml_metrics()
feat_importance = load_feature_importance()


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION & DYNAMIC FILTERS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0; text-align: left;">
        <h2 style="color: #0F172A; font-size: 20px; font-weight: 700; margin: 0; display: flex; align-items: center; gap: 8px;">
            🏦 ApexCredit AI
        </h2>
        <span style="font-size: 12px; color: #64748B; font-weight: 500;">Credit Risk & Default Decisioning</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧭 Navigation")
    menu_options = [
        "Executive Overview",
        "Customer Analytics",
        "Loan Portfolio Analysis",
        "Credit Risk Analysis",
        "SQL Analytics Console",
        "ML Model Performance",
        "Individual Loan Prediction",
        "Business Recommendations"
    ]
    
    page = st.radio("Go to:", menu_options, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ⚙️ Global Portfolio Filters")
    st.caption("Filters apply across Executive, Customer, Portfolio & Risk dashboards.")
    
    # Filter 1: Loan Purpose
    all_purposes = ["All"] + sorted(raw_df['loan_intent'].unique().tolist())
    selected_purpose = st.selectbox("Loan Purpose", all_purposes)
    
    # Filter 2: Loan Grade
    all_grades = ["All"] + sorted(raw_df['loan_grade'].unique().tolist())
    selected_grade = st.selectbox("Credit Grade", all_grades)
    
    # Filter 3: Home Ownership
    all_homes = ["All"] + sorted(raw_df['person_home_ownership'].unique().tolist())
    selected_home = st.selectbox("Home Ownership", all_homes)
    
    # Filter 4: Income Tier
    income_order = ['< $25k', '$25k-$50k', '$50k-$75k', '$75k-$100k', '$100k+']
    all_income_groups = ["All"] + income_order
    selected_income = st.selectbox("Income Bracket", all_income_groups)
    
    # Apply Filtering
    filtered_df = raw_df.copy()
    if selected_purpose != "All":
        filtered_df = filtered_df[filtered_df['loan_intent'] == selected_purpose]
    if selected_grade != "All":
        filtered_df = filtered_df[filtered_df['loan_grade'] == selected_grade]
    if selected_home != "All":
        filtered_df = filtered_df[filtered_df['person_home_ownership'] == selected_home]
    if selected_income != "All":
        filtered_df = filtered_df[filtered_df['income_group'] == selected_income]
        
    st.markdown(f"""
    <div style="background-color: #F1F5F9; border-radius: 8px; padding: 10px 14px; margin-top: 15px; font-size: 12px; color: #334155;">
        <strong>Active Cohort:</strong> {len(filtered_df):,} of {len(raw_df):,} loans ({round(len(filtered_df)/len(raw_df)*100, 1)}%)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 11px; color: #94A3B8; line-height: 1.4;">
        <strong>System Version:</strong> v2.4 Enterprise<br>
        <strong>Validation Strategy:</strong> Stratified 5-Fold CV<br>
        <strong>Model Engine:</strong> XGBoost & Random Forest
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# APP HEADER BANNER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="brand-header">
    <div class="brand-title">
        <span>🏦</span> CREDIT RISK ANALYSIS & LOAN DEFAULT PREDICTION SYSTEM
    </div>
    <div class="brand-subtitle">
        Enterprise banking intelligence platform for credit underwriting, portfolio exposure surveillance, and machine learning default forecasting.
    </div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ===========================================================================
if page == "Executive Overview":
    st.markdown("### 📊 Executive Portfolio Overview")
    
    # Calculate filtered KPIs
    total_apps = len(filtered_df)
    total_volume = filtered_df['loan_amnt'].sum()
    avg_ticket = filtered_df['loan_amnt'].mean() if total_apps > 0 else 0
    total_defaults = filtered_df['loan_status'].sum()
    default_rate = (filtered_df['loan_status'].mean() * 100) if total_apps > 0 else 0
    capital_at_risk = filtered_df[filtered_df['loan_status'] == 1]['loan_amnt'].sum()
    avg_int_rate = filtered_df['loan_int_rate'].mean() if total_apps > 0 else 0
    avg_lti = filtered_df['loan_percent_income'].mean() * 100 if total_apps > 0 else 0

    # 4-Column KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Loan Applications</div>
            <div class="kpi-value">{total_apps:,}</div>
            <div class="kpi-subtext">Total Origination Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Funded Capital</div>
            <div class="kpi-value">${total_volume/1e6:.2f}M</div>
            <div class="kpi-subtext">Avg Ticket: <strong>${avg_ticket:,.0f}</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        badge_cls = "kpi-badge-danger" if default_rate > 20 else "kpi-badge-success"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Portfolio Default Rate</div>
            <div class="kpi-value <span class='{badge_cls}'>{default_rate:.2f}%</span></div>
            <div class="kpi-subtext">{total_defaults:,} defaulted accounts</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Capital at Loss Exposure</div>
            <div class="kpi-value">${capital_at_risk/1e6:.2f}M</div>
            <div class="kpi-subtext">Avg Interest Rate: <strong>{avg_int_rate:.2f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Monthly Trends & Donut Distribution
    col_chart1, col_chart2 = st.columns([2, 1])

    with col_chart1:
        st.markdown("##### 📈 Monthly Loan Origination & Default Trajectory")
        monthly_df = filtered_df.groupby('application_month').agg(
            total_loans=('loan_id', 'count'),
            disbursed_volume=('loan_amnt', 'sum'),
            defaults=('loan_status', 'sum')
        ).reset_index()
        monthly_df['default_rate'] = (monthly_df['defaults'] / monthly_df['total_loans']) * 100

        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_trend.add_trace(
            go.Bar(
                x=monthly_df['application_month'],
                y=monthly_df['disbursed_volume'] / 1e6,
                name="Funded Capital ($M)",
                marker_color="#3B82F6",
                opacity=0.75
            ),
            secondary_y=False
        )
        
        fig_trend.add_trace(
            go.Scatter(
                x=monthly_df['application_month'],
                y=monthly_df['default_rate'],
                name="Default Rate (%)",
                line=dict(color="#EF4444", width=3, dash="solid"),
                mode="lines+markers"
            ),
            secondary_y=True
        )
        
        fig_trend.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            height=340
        )
        fig_trend.update_yaxes(title_text="Disbursed ($M)", secondary_y=False, showgrid=True)
        fig_trend.update_yaxes(title_text="Default Rate (%)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_chart2:
        st.markdown("##### 🎯 Loan Status Breakdown")
        status_counts = filtered_df['default_label'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig_donut = px.pie(
            status_counts,
            names='Status',
            values='Count',
            hole=0.55,
            color='Status',
            color_discrete_map={
                'Non-Default (Current/Paid)': '#10B981',
                'Defaulted': '#EF4444'
            }
        )
        fig_donut.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            height=340
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Row 2: Portfolio Exposure by Credit Grade & Loan Intent
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("##### 🏷️ Loan Grade Risk Profile & Exposure")
        grade_analysis = filtered_df.groupby('loan_grade').agg(
            total_loans=('loan_id', 'count'),
            total_volume=('loan_amnt', 'sum'),
            default_rate=('loan_status', lambda x: round(x.mean() * 100, 2))
        ).reset_index()
        
        fig_grade = px.bar(
            grade_analysis,
            x='loan_grade',
            y='total_volume',
            color='default_rate',
            color_continuous_scale="Reds",
            labels={'loan_grade': 'Credit Grade', 'total_volume': 'Total Capital ($)', 'default_rate': 'Default %'},
            text='default_rate'
        )
        fig_grade.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_grade.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20), height=320)
        st.plotly_chart(fig_grade, use_container_width=True)

    with row2_col2:
        st.markdown("##### 🎯 Capital Allocation by Loan Intent")
        intent_analysis = filtered_df.groupby('loan_intent').agg(
            total_volume=('loan_amnt', 'sum'),
            default_rate=('loan_status', lambda x: round(x.mean() * 100, 2))
        ).reset_index().sort_values(by='total_volume', ascending=False)
        
        fig_intent = px.bar(
            intent_analysis,
            x='total_volume',
            y='loan_intent',
            orientation='h',
            color='default_rate',
            color_continuous_scale="Viridis",
            labels={'loan_intent': 'Purpose', 'total_volume': 'Capital Funded ($)', 'default_rate': 'Default Rate (%)'}
        )
        fig_intent.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=20, b=20), height=320)
        st.plotly_chart(fig_intent, use_container_width=True)

    # Senior Analyst Summary Box
    st.markdown("""
    <div style="background-color: #EFF6FF; border-left: 5px solid #3B82F6; padding: 16px 20px; border-radius: 6px; margin-top: 15px;">
        <strong style="color: #1E40AF; font-size: 14px;">Senior Portfolio Analyst Key Findings:</strong>
        <ul style="color: #1E3A8A; font-size: 13px; margin-top: 6px; margin-bottom: 0;">
            <li><strong>Credit Grade Steepening:</strong> Grades A and B exhibit pristine default rates under 10%, while Grades D, E, F, and G exceed 55% default risk.</li>
            <li><strong>Loan Purpose Vulnerabilities:</strong> Debt Consolidation and Medical borrowing hold highest loss propensities, representing non-discretionary borrowing under financial strain.</li>
            <li><strong>Leverage Sensitivity:</strong> Applicants whose requested loan exceeds 35% of annual income experience a 3.4x higher default frequency.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 2: CUSTOMER ANALYTICS
# ===========================================================================
elif page == "Customer Analytics":
    st.markdown("### 👥 Customer Demographics & Financial Health")
    st.caption("Deep behavioral and demographic analysis of credit applicants.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 💵 Applicant Income Distribution vs Default Risk")
        fig_inc = px.histogram(
            filtered_df[filtered_df['person_income'] <= 200000],
            x="person_income",
            color="default_label",
            nbins=40,
            barmode="overlay",
            color_discrete_map={'Non-Default (Current/Paid)': '#10B981', 'Defaulted': '#EF4444'},
            labels={"person_income": "Annual Income ($)", "default_label": "Loan Status"},
            opacity=0.7
        )
        fig_inc.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_inc, use_container_width=True)

    with col2:
        st.markdown("##### 🎂 Applicant Age Distribution & Default Rate")
        age_summary = filtered_df.groupby('age_group').agg(
            total=('loan_id', 'count'),
            default_rate=('loan_status', lambda x: round(x.mean() * 100, 2))
        ).reindex(['18-25', '26-35', '36-50', '50+']).reset_index()

        fig_age = px.bar(
            age_summary,
            x='age_group',
            y='total',
            color='default_rate',
            color_continuous_scale='Oranges',
            text='default_rate',
            labels={'age_group': 'Age Bracket', 'total': 'Applications', 'default_rate': 'Default %'}
        )
        fig_age.update_traces(texttemplate='%{text}% Default', textposition='outside')
        fig_age.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_age, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("##### 💼 Employment Duration vs Default Rate")
        emp_summary = filtered_df.groupby('person_emp_length').agg(
            total=('loan_id', 'count'),
            default_rate=('loan_status', lambda x: round(x.mean() * 100, 2))
        ).reset_index()
        emp_summary = emp_summary[emp_summary['person_emp_length'] <= 25]

        fig_emp = px.line(
            emp_summary,
            x='person_emp_length',
            y='default_rate',
            markers=True,
            line_shape='spline',
            labels={'person_emp_length': 'Employment Length (Years)', 'default_rate': 'Default Rate (%)'}
        )
        fig_emp.update_traces(line_color="#2563EB", marker=dict(size=8, color="#1D4ED8"))
        fig_emp.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_emp, use_container_width=True)

    with col4:
        st.markdown("##### 🏠 Home Ownership Structure & Risk")
        home_df = filtered_df.groupby('person_home_ownership').agg(
            total=('loan_id', 'count'),
            default_rate=('loan_status', lambda x: round(x.mean() * 100, 2)),
            avg_income=('person_income', 'mean')
        ).reset_index().sort_values(by='default_rate', ascending=False)

        fig_home = px.bar(
            home_df,
            x='person_home_ownership',
            y='default_rate',
            color='person_home_ownership',
            text='default_rate',
            labels={'person_home_ownership': 'Residential Status', 'default_rate': 'Default %'}
        )
        fig_home.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_home.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_home, use_container_width=True)

    st.markdown("##### 📑 Cross-Tabulation Matrix: Age Group × Income Tier")
    cross_tab = pd.crosstab(
        filtered_df['income_group'],
        filtered_df['age_group'],
        values=filtered_df['loan_status'],
        aggfunc=lambda x: f"{round(x.mean()*100, 1)}% ({len(x):,})"
    ).reindex(['< $25k', '$25k-$50k', '$50k-$75k', '$75k-$100k', '$100k+'])
    st.dataframe(cross_tab, use_container_width=True)


# ===========================================================================
# PAGE 3: LOAN PORTFOLIO ANALYSIS
# ===========================================================================
elif page == "Loan Portfolio Analysis":
    st.markdown("### 📊 Loan Portfolio & Capital Sizing")
    st.caption("Distribution of loan amounts, interest rates, and leverage ratios.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 💰 Loan Amount Distribution by Purpose")
        fig_box = px.box(
            filtered_df,
            x="loan_intent",
            y="loan_amnt",
            color="loan_intent",
            labels={"loan_intent": "Purpose", "loan_amnt": "Loan Amount ($)"}
        )
        fig_box.update_layout(template="plotly_white", height=350, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        st.markdown("##### 📈 Loan Amount vs Interest Rate by Grade")
        sample_df = filtered_df.sample(min(2500, len(filtered_df)), random_state=42)
        fig_scatter = px.scatter(
            sample_df,
            x="loan_amnt",
            y="loan_int_rate",
            color="loan_grade",
            category_orders={"loan_grade": ['A', 'B', 'C', 'D', 'E', 'F', 'G']},
            labels={"loan_amnt": "Loan Principal ($)", "loan_int_rate": "Interest Rate (%)", "loan_grade": "Grade"},
            opacity=0.65
        )
        fig_scatter.update_layout(template="plotly_white", height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("##### ⚖️ Debt-to-Income (Loan % Income) Distribution")
        fig_lti = px.histogram(
            filtered_df,
            x="loan_percent_income",
            color="default_label",
            nbins=35,
            barmode="overlay",
            color_discrete_map={'Non-Default (Current/Paid)': '#10B981', 'Defaulted': '#EF4444'},
            labels={"loan_percent_income": "Loan as % of Annual Income", "default_label": "Status"},
            opacity=0.7
        )
        fig_lti.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_lti, use_container_width=True)

    with col4:
        st.markdown("##### 🛡️ Credit History Length vs Default Rate")
        cred_summary = filtered_df.groupby('cb_person_cred_hist_length').agg(
            total=('loan_id', 'count'),
            default_rate=('loan_status', lambda x: round(x.mean() * 100, 2))
        ).reset_index()
        cred_summary = cred_summary[cred_summary['cb_person_cred_hist_length'] <= 20]

        fig_cred = px.bar(
            cred_summary,
            x='cb_person_cred_hist_length',
            y='default_rate',
            color='default_rate',
            color_continuous_scale="Teal",
            labels={'cb_person_cred_hist_length': 'Credit History (Years)', 'default_rate': 'Default Rate (%)'}
        )
        fig_cred.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_cred, use_container_width=True)


# ===========================================================================
# PAGE 4: CREDIT RISK ANALYSIS
# ===========================================================================
elif page == "Credit Risk Analysis":
    st.markdown("### ⚠️ Credit Risk Drivers & Default Patterns")
    st.caption("Statistical isolation of root-cause risk indicators and default correlations.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### ⚠️ Impact of Historical Default Record")
        hist_df = filtered_df.groupby('cb_person_default_on_file').agg(
            total=('loan_id', 'count'),
            defaults=('loan_status', 'sum'),
            default_rate=('loan_status', lambda x: round(x.mean() * 100, 2))
        ).reset_index()
        hist_df['Label'] = hist_df['cb_person_default_on_file'].map({'Y': 'Prior Default on File', 'N': 'No Prior Default'})

        fig_hist = px.bar(
            hist_df,
            x='Label',
            y='default_rate',
            color='default_rate',
            color_continuous_scale='Reds',
            text='default_rate',
            labels={'Label': 'Credit Bureau Flag', 'default_rate': 'Default Rate (%)'}
        )
        fig_hist.update_traces(texttemplate='%{text}% Default', textposition='outside')
        fig_hist.update_layout(template="plotly_white", height=330, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.markdown("##### 🗺️ Risk Heatmap: Credit Grade vs Income Bracket")
        heatmap_data = pd.crosstab(
            filtered_df['loan_grade'],
            filtered_df['income_group'],
            values=filtered_df['loan_status'],
            aggfunc=lambda x: round(x.mean() * 100, 1)
        ).reindex(index=['A', 'B', 'C', 'D', 'E', 'F', 'G'], columns=['< $25k', '$25k-$50k', '$50k-$75k', '$75k-$100k', '$100k+'])

        fig_heat = px.imshow(
            heatmap_data,
            labels=dict(x="Income Group", y="Loan Grade", color="Default %"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale="Reds",
            text_auto=True
        )
        fig_heat.update_layout(template="plotly_white", height=330, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    # Correlation Matrix
    st.markdown("##### 🔗 Pearson Correlation of Continuous Risk Indicators")
    num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'loan_status']
    corr_df = filtered_df[num_cols].corr().round(3)
    
    fig_corr = px.imshow(
        corr_df,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    fig_corr.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)


# ===========================================================================
# PAGE 5: SQL ANALYTICS CONSOLE
# ===========================================================================
elif page == "SQL Analytics Console":
    st.markdown("### 🔍 SQL Analytics & Ad-Hoc Query Engine")
    st.caption("Live execution against the production SQLite database (`credit_risk.db`).")

    predefined_queries = db.get_predefined_queries()
    
    tab1, tab2 = st.tabs(["⚡ Predefined Business Queries", "💻 Custom SQL Terminal"])

    with tab1:
        query_choice = st.selectbox("Select Business Query:", list(predefined_queries.keys()))
        sql_to_run = predefined_queries[query_choice]

        st.markdown("##### SQL Implementation:")
        st.code(sql_to_run.strip(), language="sql")

        if st.button("▶️ Execute Query", key="btn_run_predefined"):
            try:
                result_df = db.execute_query(sql_to_run)
                st.success(f"Query returned {len(result_df)} records successfully.")
                st.dataframe(result_df, use_container_width=True)

                # Auto-visualize if appropriate
                num_cols_res = result_df.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols_res = result_df.select_dtypes(include=['object']).columns.tolist()

                if len(cat_cols_res) >= 1 and len(num_cols_res) >= 1:
                    st.markdown("##### 📊 Automated SQL Visualization")
                    fig_sql = px.bar(
                        result_df,
                        x=cat_cols_res[0],
                        y=num_cols_res[0],
                        color=num_cols_res[-1] if len(num_cols_res) > 1 else None,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_sql, use_container_width=True)
            except Exception as e:
                st.error(f"SQL Error: {e}")

    with tab2:
        st.markdown("Enter custom `SELECT` SQL statements to query `loan_applications`:")
        user_sql = st.text_area(
            "SQL Query Editor",
            value="SELECT loan_grade, COUNT(*) AS total_loans, ROUND(AVG(loan_amnt), 2) AS avg_amount, ROUND(AVG(loan_status)*100, 2) AS default_pct FROM loan_applications GROUP BY loan_grade ORDER BY default_pct DESC;",
            height=120
        )
        if st.button("▶️ Run Custom SQL", key="btn_run_custom"):
            if not user_sql.strip().lower().startswith("select"):
                st.error("Security restriction: Only read-only SELECT queries are permitted.")
            else:
                try:
                    custom_df = db.execute_query(user_sql)
                    st.success(f"Execution complete: {len(custom_df)} rows returned.")
                    st.dataframe(custom_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Database error: {e}")


# ===========================================================================
# PAGE 6: ML MODEL PERFORMANCE
# ===========================================================================
elif page == "ML Model Performance":
    st.markdown("### 🤖 Machine Learning Model Benchmarking")
    st.caption("Rigorous evaluation on unseen holdout test set (20% split) with Stratified 5-Fold Cross Validation.")

    metrics = ml_metrics['metrics']
    curves = ml_metrics['curves']

    # Metrics Summary Table
    metrics_table = []
    for model_name, m in metrics.items():
        metrics_table.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "Precision": f"{m['precision']*100:.2f}%",
            "Recall": f"{m['recall']*100:.2f}%",
            "F1-Score": f"{m['f1_score']:.4f}",
            "Holdout ROC-AUC": f"{m['roc_auc']:.4f}",
            "5-Fold CV ROC-AUC": f"{m['cv_mean_roc_auc']:.4f} (±{m['cv_std_roc_auc']:.3f})"
        })
    df_metrics = pd.DataFrame(metrics_table)
    st.dataframe(df_metrics, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📉 ROC Curves Comparison (Holdout Test Set)")
        fig_roc = go.Figure()
        colors = {'Logistic Regression': '#94A3B8', 'Decision Tree': '#F59E0B', 'Random Forest': '#3B82F6', 'XGBoost': '#10B981'}
        
        for model_name, c in curves.items():
            fig_roc.add_trace(go.Scatter(
                x=c['fpr'],
                y=c['tpr'],
                name=f"{model_name} (AUC = {metrics[model_name]['roc_auc']:.3f})",
                line=dict(color=colors.get(model_name, '#000'), width=2.5)
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name="Random Guess (AUC = 0.50)",
            line=dict(dash="dash", color="#CBD5E1")
        ))
        fig_roc.update_layout(
            template="plotly_white",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.markdown("##### 🎯 Precision-Recall Curves")
        fig_pr = go.Figure()
        for model_name, c in curves.items():
            fig_pr.add_trace(go.Scatter(
                x=c['pr_recall'],
                y=c['pr_precision'],
                name=f"{model_name}",
                line=dict(color=colors.get(model_name, '#000'), width=2)
            ))
        fig_pr.update_layout(
            template="plotly_white",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35)
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # Confusion Matrix Explorer
    st.markdown("##### 🧮 Confusion Matrix & Financial Trade-Off Explorer")
    selected_eval_model = st.selectbox("Select Model for Confusion Matrix:", list(metrics.keys()))
    cm_data = metrics[selected_eval_model]
    
    cm_col1, cm_col2 = st.columns([1, 1])
    
    with cm_col1:
        cm_matrix = np.array([[cm_data['tn'], cm_data['fp']], [cm_data['fn'], cm_data['tp']]])
        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Condition", y="Actual Condition", color="Count"),
            x=['Predicted Non-Default', 'Predicted Default'],
            y=['Actual Non-Default', 'Actual Default'],
            color_continuous_scale="Blues",
            text_auto=True
        )
        fig_cm.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    with cm_col2:
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px;">
            <h4 style="margin: 0 0 10px 0; color: #0F172A; font-size: 15px;">Banking Financial Interpretation ({selected_eval_model}):</h4>
            <ul style="font-size: 13px; color: #475569; line-height: 1.6;">
                <li><strong>True Negatives (TN = {cm_data['tn']:,}):</strong> Creditworthy borrowers correctly approved. Primary source of interest revenue.</li>
                <li><strong>True Positives (TP = {cm_data['tp']:,}):</strong> Defaulting borrowers correctly rejected or collateralized. Loss averted!</li>
                <li><strong>False Positives (FP = {cm_data['fp']:,}):</strong> Good borrowers denied credit (Opportunity cost / lost customer).</li>
                <li><strong>False Negatives (FN = {cm_data['fn']:,}):</strong> Critical default risk! Defaulting applicants mistakenly approved, causing direct credit loss.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Feature Importance
    st.markdown("##### 🏆 Global Feature Importance (Top Risk Drivers)")
    selected_fi_model = st.selectbox("Feature Importance from:", ["XGBoost", "Random Forest", "Logistic Regression"])
    fi_list = feat_importance.get(selected_fi_model, [])[:10]
    df_fi = pd.DataFrame(fi_list).sort_values(by='importance', ascending=True)

    fig_fi = px.bar(
        df_fi,
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale='Teal',
        labels={'importance': 'Relative Importance', 'feature': 'Predictor Variable'}
    )
    fig_fi.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_fi, use_container_width=True)


# ===========================================================================
# PAGE 7: INDIVIDUAL LOAN DEFAULT PREDICTION
# ===========================================================================
elif page == "Individual Loan Prediction":
    st.markdown("### 🔮 Individual Loan Underwriting & Default Predictor")
    st.caption("Input loan applicant particulars to receive real-time default probability, risk tiering, and tailored mitigation steps.")

    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Borrower Profile")
            age = st.slider("Borrower Age", min_value=18, max_value=85, value=29)
            income = st.number_input("Annual Income ($)", min_value=5000, max_value=2000000, value=65000, step=5000)
            emp_length = st.slider("Employment Length (Years)", min_value=0.0, max_value=45.0, value=4.0, step=0.5)
            home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

        with col2:
            st.markdown("##### 💳 Loan Specifications")
            loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=50000, value=12000, step=1000)
            loan_intent = st.selectbox(
                "Loan Purpose",
                ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
            )
            loan_grade = st.selectbox("Credit Grade (A = Best, G = Riskiest)", ["A", "B", "C", "D", "E", "F", "G"], index=1)
            
            # Smart default interest rate based on loan grade
            default_rate_map = {'A': 7.5, 'B': 11.0, 'C': 13.5, 'D': 15.3, 'E': 16.8, 'F': 18.5, 'G': 20.2}
            rec_rate = default_rate_map.get(loan_grade, 11.0)
            int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=25.0, value=float(rec_rate), step=0.1)

        with col3:
            st.markdown("##### 🛡️ Credit Bureau Record")
            default_history = st.radio("Historical Default on Record?", ["N", "Y"], format_func=lambda x: "No Previous Default" if x == "N" else "Yes, Has Default History")
            cred_hist_len = st.slider("Credit Bureau History (Years)", min_value=1, max_value=30, value=5)
            selected_model = st.selectbox("Predictive Model", ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"])

        submitted = st.form_submit_button("⚡ Run Real-Time Credit Assessment", use_container_width=True)

    if submitted:
        calc_lti = round(loan_amnt / max(income, 1), 3)
        applicant_payload = {
            'person_age': int(age),
            'person_income': float(income),
            'person_emp_length': float(emp_length),
            'loan_amnt': float(loan_amnt),
            'loan_int_rate': float(int_rate),
            'loan_percent_income': calc_lti,
            'cb_person_cred_hist_length': int(cred_hist_len),
            'person_home_ownership': home_ownership,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'cb_person_default_on_file': default_history
        }

        # Predict
        res = predictor.predict_applicant(applicant_payload, model_name=selected_model)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📋 Credit Decision & Risk Scorecard")

        res_col1, res_col2 = st.columns([1, 1.3])

        with res_col1:
            # Risk Gauge Meter
            prob = res['default_probability_pct']
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Default Probability", 'font': {'size': 20}},
                number={'suffix': "%", 'font': {'size': 36, 'color': res['risk_color']}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': res['risk_color'], 'thickness': 0.3},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 20], 'color': '#DCFCE7'},
                        {'range': [20, 45], 'color': '#FEF3C7'},
                        {'range': [45, 100], 'color': '#FEE2E2'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.8,
                        'value': prob
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk Category Badge
            badge_map = {'Low Risk': 'badge-low', 'Medium Risk': 'badge-medium', 'High Risk': 'badge-high'}
            st.markdown(f"""
            <div style="text-align: center; margin-top: -10px;">
                <span class="{badge_map.get(res['risk_tier'], 'badge-medium')}">
                    {res['risk_tier'].upper()}
                </span>
                <p style="font-size: 15px; font-weight: 600; color: #1E293B; margin-top: 10px;">
                    Underwriting Verdict: {res['approval_recommendation']}
                </p>
                <p style="font-size: 12px; color: #64748B;">
                    Loan-to-Income Exposure: <strong>{round(calc_lti*100, 1)}%</strong> | Debt Monthly Burden: <strong>${round(loan_amnt/36, 0):,.0f}/mo (est. 3yr)</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("##### 🔍 Key Risk Drivers for Applicant")
            for f in res['key_factors']:
                if f['direction'] == 'positive':
                    icon = "✅"
                    badge_style = "color: #15803D; background: #DCFCE7; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;"
                elif f['direction'] == 'negative':
                    icon = "⚠️"
                    badge_style = "color: #B91C1C; background: #FEE2E2; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;"
                else:
                    icon = "ℹ️"
                    badge_style = "color: #475569; background: #F1F5F9; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;"

                st.markdown(f"""
                <div style="margin-bottom: 8px; font-size: 13px; line-height: 1.4;">
                    {icon} <strong>{f['factor']}</strong> <span style="{badge_style}">{f['impact']}</span><br>
                    <span style="color: #64748B; font-size: 12px; padding-left: 20px;">{f['detail']}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("##### 🛡️ Prescribed Underwriting Actions")
            for action in res['suggested_actions']:
                st.markdown(f"""
                <div style="background-color: #F8FAFC; border-left: 3px solid #3B82F6; padding: 6px 12px; margin-bottom: 6px; font-size: 12px; color: #1E293B;">
                    👉 {action}
                </div>
                """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 8: BUSINESS RECOMMENDATIONS
# ===========================================================================
elif page == "Business Recommendations":
    st.markdown("### 📋 Executive Credit Policy & Portfolio Strategy")
    st.caption("Strategic underwriting guidelines, risk-based pricing rules, and concentration safeguards.")

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown("""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
            <h4 style="color: #0F172A; margin-top: 0;">1. Automated Decisioning Cut-Off Matrix</h4>
            <table style="width: 100%; font-size: 13px; border-collapse: collapse; margin-top: 10px;">
                <tr style="background: #F1F5F9; border-bottom: 1px solid #CBD5E1;">
                    <th style="padding: 8px; text-align: left;">Risk Tier</th>
                    <th style="padding: 8px; text-align: left;">Default Prob</th>
                    <th style="padding: 8px; text-align: left;">Policy Directive</th>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0;">
                    <td style="padding: 8px;"><span class="badge-low">Tier 1: Low</span></td>
                    <td style="padding: 8px;">&lt; 20%</td>
                    <td style="padding: 8px;"><strong>Instant Straight-Through Approval</strong> with prime pricing.</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0;">
                    <td style="padding: 8px;"><span class="badge-medium">Tier 2: Medium</span></td>
                    <td style="padding: 8px;">20% - 45%</td>
                    <td style="padding: 8px;"><strong>Conditional Review</strong>: Require income verification, debt cap.</td>
                </tr>
                <tr>
                    <td style="padding: 8px;"><span class="badge-high">Tier 3: High</span></td>
                    <td style="padding: 8px;">&gt; 45%</td>
                    <td style="padding: 8px;"><strong>Automated Decline</strong> or require 100% co-signer/collateral.</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 20px;">
            <h4 style="color: #0F172A; margin-top: 0;">2. Debt-to-Income (DTI) Guardrails</h4>
            <p style="font-size: 13px; color: #475569;">
                The empirical data proves default probability rises exponentially when loan amount exceeds <strong>30% of annual income</strong>.
            </p>
            <ul style="font-size: 13px; color: #334155;">
                <li><strong>Strict Cap at 35% LTI:</strong> Reject unsecured personal/medical loans where loan exceeds 35% of income.</li>
                <li><strong>Dynamic Tenor Structuring:</strong> Extend loan amortization from 36 to 60 months for borderline applicants to decrease monthly burden.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with rec_col2:
        st.markdown("""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
            <h4 style="color: #0F172A; margin-top: 0;">3. Risk-Based Pricing Framework</h4>
            <p style="font-size: 13px; color: #475569;">
                Implement dynamic risk premia to protect expected loss provisions ($EL = PD \times EAD \times LGD$):
            </p>
            <ul style="font-size: 13px; color: #334155;">
                <li><strong>Prime Tier (Grades A-B):</strong> 7.0% - 11.0% APR to maintain competitive market volume.</li>
                <li><strong>Near-Prime (Grade C):</strong> 13.0% - 14.5% APR + 1.0% origination fee.</li>
                <li><strong>Subprime (Grades D-G):</strong> Minimum 16.5% - 21.0% APR with compulsory collateral coverage.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 20px;">
            <h4 style="color: #0F172A; margin-top: 0;">4. Portfolio Concentration Limits</h4>
            <ul style="font-size: 13px; color: #334155;">
                <li><strong>Subprime Exposure Ceiling:</strong> Cap combined portfolio exposure in Grades E, F, and G to under <strong>12%</strong> of total funded book.</li>
                <li><strong>Medical & Debt Consolidation Limits:</strong> Institute secondary underwriting covenants on debt consolidation loans to disburse payoff funds directly to third-party lenders.</li>
                <li><strong>Early Warning Telemetry:</strong> Deploy monthly re-scoring on active borrowers using XGBoost to detect pre-delinquency signs early.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
