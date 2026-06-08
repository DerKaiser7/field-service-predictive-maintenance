"""
Logistic Regression Baseline Model

Purpose:
  - Train a simple, interpretable baseline for comparison
  - Validate the feature engineering pipeline works end-to-end
  - Serve as a meta-learner base for stacking (Phase 4)

Strategy:
  - Load features from PostgreSQL (model_input_features table)
  - Handle missing values with mean imputation
  - 5-fold cross-validation using PR-AUC metric
  - Train on full train+val, evaluate on test
  - Log CV scores and model coefficients
  - Save model artifact for later use

Output:
  - models/baseline_lr.pkl (trained model)
  - models/baseline_metrics.json (CV scores + test performance)
  - models/baseline_feature_importance.csv (coefficients for interpretation)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.data_operations.load import get_engine

load_dotenv()

# =====================================================================
# CONFIG
# =====================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
CV_FOLDS = 5

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# =====================================================================
# LOAD DATA FROM DATABASE
# =====================================================================

def load_features_from_db() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load model_input_features from PostgreSQL.
    
    Returns:
      X: Feature matrix (all columns except label)
      y: Target vector (label column)
    """
    engine = get_engine()
    
    query = """
    SELECT * FROM model_input_features
    WHERE label IS NOT NULL
    ORDER BY machineid, observation_time
    """
    
    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df):,} rows from model_input_features")
    print(f"Columns: {df.shape[1]}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Separate features and target
    X = df.drop(columns=['feature_id', 'machineid', 'observation_time', 'label'])
    y = df['label']
    
    return X, y


# =====================================================================
# PREPROCESSING
# =====================================================================

def preprocess_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Preprocess features:
    1. Impute missing values (mean strategy)
    2. Standardize numeric features
    
    Returns:
      X_train_scaled, X_val_scaled, X_test_scaled, feature_names
    """
    
    # Separate categorical and numeric features
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    # Impute missing values (using only train statistics)
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = X_train.copy()
    X_train_imputed[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_val_imputed = X_val.copy()
    X_val_imputed[numeric_cols] = imputer.transform(X_val[numeric_cols])
    X_test_imputed = X_test.copy()
    X_test_imputed[numeric_cols] = imputer.transform(X_test[numeric_cols])
    
    # Standardize numeric features
    scaler = StandardScaler()
    X_train_scaled = X_train_imputed.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_imputed[numeric_cols])
    X_val_scaled = X_val_imputed.copy()
    X_val_scaled[numeric_cols] = scaler.transform(X_val_imputed[numeric_cols])
    X_test_scaled = X_test_imputed.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test_imputed[numeric_cols])
    
    # One-hot encode categorical features if needed
    if categorical_cols:
        X_train_scaled = pd.get_dummies(X_train_scaled, columns=categorical_cols, drop_first=True)
        X_val_scaled = pd.get_dummies(X_val_scaled, columns=categorical_cols, drop_first=True)
        X_test_scaled = pd.get_dummies(X_test_scaled, columns=categorical_cols, drop_first=True)
        
        # Align columns
        all_cols = X_train_scaled.columns
        X_val_scaled = X_val_scaled.reindex(columns=all_cols, fill_value=0)
        X_test_scaled = X_test_scaled.reindex(columns=all_cols, fill_value=0)
    
    print(f"After preprocessing: {X_train_scaled.shape[1]} features")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, X_train_scaled.columns.tolist()


# =====================================================================
# CROSS-VALIDATION
# =====================================================================

def evaluate_with_cv(X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
    """
    5-fold cross-validation using PR-AUC metric.
    """
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver='lbfgs'
    )
    
    # PR-AUC scoring (area under precision-recall curve)
    cv_scores = cross_val_score(
        model, X, y,
        cv=cv_folds,
        scoring='average_precision',  # PR-AUC
        n_jobs=-1
    )
    
    print(f"\n{'='*60}")
    print(f"5-Fold Cross-Validation Results (Logistic Regression)")
    print(f"{'='*60}")
    print(f"PR-AUC Scores: {cv_scores}")
    print(f"Mean PR-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"{'='*60}\n")
    
    return {
        'cv_scores': cv_scores.tolist(),
        'mean_pr_auc': float(cv_scores.mean()),
        'std_pr_auc': float(cv_scores.std()),
    }


# =====================================================================
# TRAIN FINAL MODEL
# =====================================================================

def train_final_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series
) -> tuple[LogisticRegression, dict]:
    """
    Train final model on train+val, evaluate on test.
    """
    
    # Combine train and val for final training
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)
    
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_combined, y_combined)
    
    # Evaluate on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Compute PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Additional metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        'test_pr_auc': float(pr_auc),
        'test_precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
        'test_recall': float(sensitivity),
        'test_specificity': float(specificity),
        'test_f1': float(f1),
        'test_tp': int(tp),
        'test_fp': int(fp),
        'test_tn': int(tn),
        'test_fn': int(fn),
    }
    
    print(f"\n{'='*60}")
    print(f"Test Set Performance (Logistic Regression)")
    print(f"{'='*60}")
    print(f"PR-AUC: {metrics['test_pr_auc']:.4f}")
    print(f"Precision: {metrics['test_precision']:.4f}")
    print(f"Recall: {metrics['test_recall']:.4f}")
    print(f"Specificity: {metrics['test_specificity']:.4f}")
    print(f"F1-Score: {metrics['test_f1']:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"{'='*60}\n")
    
    return model, metrics


# =====================================================================
# SAVE ARTIFACTS
# =====================================================================

def save_model_and_metrics(model: LogisticRegression, cv_results: dict, test_metrics: dict, feature_names: list):
    """Save model and metrics to disk."""
    
    # Save model
    model_path = MODEL_DIR / "baseline_lr.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {model_path}")
    
    # Save metrics
    all_metrics = {
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'model_type': 'LogisticRegression',
        'n_features': model.n_features_in_,
    }
    metrics_path = MODEL_DIR / "baseline_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    
    # Save feature importance (coefficients)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0],
    }).sort_values('coefficient', ascending=False)
    
    importance_path = MODEL_DIR / "baseline_feature_importance.csv"
    feature_importance_df.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved: {importance_path}")
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("="*60)
    print("LOGISTIC REGRESSION BASELINE MODEL")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading data from PostgreSQL...")
    X, y = load_features_from_db()
    
    # Train-Val-Test split
    print(f"\n[2/5] Splitting data (60-20-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE, stratify=y_train
    )
    
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Preprocess
    print(f"\n[3/5] Preprocessing features (imputation + standardization)...")
    X_train_scaled, X_val_scaled, X_test_scaled, feature_names = preprocess_features(
        X_train, X_val, X_test
    )
    
    # Cross-validation
    print(f"\n[4/5] Cross-validation (5-fold)...")
    cv_results = evaluate_with_cv(X_train_scaled, y_train, CV_FOLDS)
    
    # Train final model
    print(f"\n[5/5] Training final model on train+val...")
    model, test_metrics = train_final_model(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test
    )
    
    # Save artifacts
    print("\n[SAVE] Saving model and metrics...")
    save_model_and_metrics(model, cv_results, test_metrics, feature_names)
    
    print("\n" + "="*60)
    print("✓ BASELINE MODEL TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
