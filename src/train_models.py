"""
Machine Learning Model Training, Comparison, and Evaluation Module
Trains and compares Logistic Regression, Decision Tree, Random Forest, and XGBoost.
Evaluates Precision, Recall, F1, ROC-AUC, Confusion Matrix, and saves serialised models.
"""

import os
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLEANED_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_loans.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def train_and_evaluate_models(
    csv_path: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None
):
    csv_path = Path(csv_path) if csv_path else DEFAULT_CLEANED_CSV
    models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR

    models_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        from src.data_cleaning import ensure_database_ready
        ensure_database_ready()

    print(f"Loading cleaned dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Core predictor features
    num_features = [
        'person_age',
        'person_income',
        'person_emp_length',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length'
    ]
    cat_features = [
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ]
    target = 'loan_status'

    X = df[num_features + cat_features].copy()
    y = df[target].values

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Dataset split: Train set = {len(X_train):,} samples, Test set = {len(X_test):,} samples")

    # ColumnTransformer preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ]
    )

    # Fit preprocessor strictly on training data
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, str(models_dir / 'preprocessor.joblib'))
    print("Preprocessor fitted and saved.")

    # Get one-hot feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_encoded_names = list(cat_encoder.get_feature_names_out(cat_features))
    all_feature_names = num_features + cat_encoded_names

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Save feature metadata
    with open(models_dir / 'feature_metadata.json', 'w') as f:
        json.dump({
            'num_features': num_features,
            'cat_features': cat_features,
            'all_feature_names': all_feature_names,
            'cat_categories': {
                'person_home_ownership': sorted(df['person_home_ownership'].unique().tolist()),
                'loan_intent': sorted(df['loan_intent'].unique().tolist()),
                'loan_grade': sorted(df['loan_grade'].unique().tolist()),
                'cb_person_default_on_file': ['N', 'Y']
            }
        }, f, indent=4)

    # Define Candidate Models
    model_candidates = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=1.5,
            eval_metric='logloss',
            random_state=42
        )
    }

    metrics_results = {}
    curve_data = {}
    feature_importance_dict = {}
    test_eval_store = {
        'y_test': y_test.tolist(),
        'predictions': {},
        'probabilities': {}
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in model_candidates.items():
        print(f"\n---> Training and evaluating {name}...")
        
        cv_scores = cross_val_score(model, X_train_proc, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        mean_cv_auc = float(np.mean(cv_scores))
        std_cv_auc = float(np.std(cv_scores))
        print(f"     5-Fold CV ROC-AUC: {mean_cv_auc:.4f} (+/- {std_cv_auc:.4f})")

        model.fit(X_train_proc, y_train)

        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)

        metrics_results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4),
            'cv_mean_roc_auc': round(mean_cv_auc, 4),
            'cv_std_roc_auc': round(std_cv_auc, 4),
            'confusion_matrix': cm,
            'tn': int(cm[0][0]),
            'fp': int(cm[0][1]),
            'fn': int(cm[1][0]),
            'tp': int(cm[1][1])
        }

        step_roc = max(1, len(fpr) // 100)
        step_pr = max(1, len(pr_rec) // 100)
        curve_data[name] = {
            'fpr': [round(x, 4) for x in fpr[::step_roc].tolist()] + [1.0],
            'tpr': [round(x, 4) for x in tpr[::step_roc].tolist()] + [1.0],
            'pr_recall': [round(x, 4) for x in pr_rec[::step_pr].tolist()],
            'pr_precision': [round(x, 4) for x in pr_prec[::step_pr].tolist()]
        }

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.zeros(len(all_feature_names))

        feat_rank = sorted(
            zip(all_feature_names, [float(x) for x in importances]),
            key=lambda x: x[1],
            reverse=True
        )
        feature_importance_dict[name] = [
            {'feature': f, 'importance': round(imp, 5)} for f, imp in feat_rank
        ]

        slug = name.lower().replace(' ', '_')
        model_filename = f"{slug}.joblib"
        joblib.dump(model, str(models_dir / model_filename))
        print(f"     Saved model to {model_filename}")

        test_eval_store['predictions'][name] = y_pred.tolist()
        test_eval_store['probabilities'][name] = [round(float(p), 4) for p in y_prob]

    with open(models_dir / 'model_metrics.json', 'w') as f:
        json.dump({
            'metrics': metrics_results,
            'curves': curve_data
        }, f, indent=4)

    with open(models_dir / 'feature_importance.json', 'w') as f:
        json.dump(feature_importance_dict, f, indent=4)

    joblib.dump(test_eval_store, str(models_dir / 'test_eval_data.joblib'))
    print("All models and metrics saved successfully.")


if __name__ == "__main__":
    train_and_evaluate_models()
