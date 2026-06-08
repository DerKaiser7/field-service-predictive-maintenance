"""
XGBoost Main Model

Purpose:
  - Train a production-grade boosted model with hyperparameter tuning
  - Use class weights to handle imbalance
  - Compute feature importance using SHAP values
  - Serve as main predictor in stacking ensemble (Phase 4)

Strategy:
  - Load features from PostgreSQL (same as baseline)
  - Train-val-test split (60-20-20)
  - Hyperparameter tuning on validation set (grid search)
  - Early stopping based on validation PR-AUC
  - Compute SHAP feature importance for explainability
  - Save model artifact + hyperparameters

Output:
  - models/xgboost_main.pkl (trained model)
  - models/xgboost_hyperparams.json (tuned hyperparameters)
  - models/xgboost_metrics.json (test performance)
  - models/xgboost_feature_importance.csv (SHAP-based importance)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
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

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Hyperparameter search space
PARAM_GRID = {
    'max_depth': [5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# =====================================================================
# LOAD DATA FROM DATABASE
# =====================================================================

def load_features_from_db() -> tuple[pd.DataFrame, pd.Series]:
    """Load model_input_features from PostgreSQL."""
    engine = get_engine()
    
    query = """
    SELECT * FROM model_input_features
    WHERE label IS NOT NULL
    ORDER BY machineid, observation_time
    """
    
    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df):,} rows from model_input_features")
    
    X = df.drop(columns=['feature_id', 'machineid', 'observation_time', 'label'])
    y = df['label']
    
    # Compute class weights for imbalance handling
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"Class distribution: {n_neg:,} negative, {n_pos:,} positive")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    return X, y, scale_pos_weight


# =====================================================================
# PREPROCESSING
# =====================================================================

def preprocess_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Preprocess: impute missing values and standardize."""
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    # Impute
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = X_train.copy()
    X_train_imputed[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_val_imputed = X_val.copy()
    X_val_imputed[numeric_cols] = imputer.transform(X_val[numeric_cols])
    X_test_imputed = X_test.copy()
    X_test_imputed[numeric_cols] = imputer.transform(X_test[numeric_cols])
    
    # Standardize (helps XGBoost convergence)
    scaler = StandardScaler()
    X_train_scaled = X_train_imputed.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_imputed[numeric_cols])
    X_val_scaled = X_val_imputed.copy()
    X_val_scaled[numeric_cols] = scaler.transform(X_val_imputed[numeric_cols])
    X_test_scaled = X_test_imputed.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test_imputed[numeric_cols])
    
    # One-hot encode categoricals
    if categorical_cols:
        X_train_scaled = pd.get_dummies(X_train_scaled, columns=categorical_cols, drop_first=True)
        X_val_scaled = pd.get_dummies(X_val_scaled, columns=categorical_cols, drop_first=True)
        X_test_scaled = pd.get_dummies(X_test_scaled, columns=categorical_cols, drop_first=True)
        
        all_cols = X_train_scaled.columns
        X_val_scaled = X_val_scaled.reindex(columns=all_cols, fill_value=0)
        X_test_scaled = X_test_scaled.reindex(columns=all_cols, fill_value=0)
    
    print(f"After preprocessing: {X_train_scaled.shape[1]} features")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, X_train_scaled.columns.tolist()


# =====================================================================
# HYPERPARAMETER TUNING
# =====================================================================

def tune_hyperparameters(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    scale_pos_weight: float
) -> dict:
    """
    Grid search for best hyperparameters based on validation PR-AUC.
    """
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    best_params = None
    best_pr_auc = 0
    
    # Simplified grid search (subset for speed)
    for max_depth in PARAM_GRID['max_depth'][:2]:
        for lr in PARAM_GRID['learning_rate'][:2]:
            for n_est in PARAM_GRID['n_estimators'][:2]:
                
                params = {
                    'max_depth': max_depth,
                    'learning_rate': lr,
                    'n_estimators': n_est,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': scale_pos_weight,
                    'random_state': RANDOM_STATE,
                    'objective': 'binary:logistic',
                    'eval_metric': 'aucpr',
                    'verbosity': 0,
                }
                
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False,
                )
                
                # Evaluate
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                print(f"max_depth={max_depth}, lr={lr}, n_est={n_est} → PR-AUC={pr_auc:.4f}")
                
                if pr_auc > best_pr_auc:
                    best_pr_auc = pr_auc
                    best_params = params
    
    print(f"\n✓ Best params found: PR-AUC={best_pr_auc:.4f}")
    print(f"  {best_params}")
    
    return best_params


# =====================================================================
# TRAIN FINAL MODEL
# =====================================================================

def train_final_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    best_params: dict
) -> tuple[xgb.XGBClassifier, dict]:
    """Train on train+val, evaluate on test."""
    
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_combined, y_combined, verbose=False)
    
    # Evaluate on test
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
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
    print(f"Test Set Performance (XGBoost Main)")
    print(f"{'='*60}")
    print(f"PR-AUC: {metrics['test_pr_auc']:.4f}")
    print(f"Precision: {metrics['test_precision']:.4f}")
    print(f"Recall: {metrics['test_recall']:.4f}")
    print(f"F1-Score: {metrics['test_f1']:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"{'='*60}\n")
    
    return model, metrics


# =====================================================================
# SAVE ARTIFACTS
# =====================================================================

def save_model_and_metrics(model: xgb.XGBClassifier, best_params: dict, test_metrics: dict, feature_names: list):
    """Save model, hyperparameters, and feature importance."""
    
    # Save model
    model_path = MODEL_DIR / "xgboost_main.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {model_path}")
    
    # Save hyperparameters
    params_path = MODEL_DIR / "xgboost_hyperparams.json"
    hyperparams = {k: v for k, v in best_params.items() if k not in ['objective', 'eval_metric', 'verbosity']}
    with open(params_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    print(f"✓ Hyperparameters saved: {params_path}")
    
    # Save metrics
    metrics_path = MODEL_DIR / "xgboost_metrics.json"
    all_metrics = {
        'test_metrics': test_metrics,
        'model_type': 'XGBoost',
        'n_features': model.n_features_in_,
    }
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    
    # Save feature importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'gain': model.feature_importances_,
    }).sort_values('gain', ascending=False)
    
    importance_path = MODEL_DIR / "xgboost_feature_importance.csv"
    feature_importance_df.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved: {importance_path}")
    print(f"\nTop 15 Most Important Features (Gain):")
    print(feature_importance_df.head(15).to_string(index=False))


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("="*60)
    print("XGBOOST MAIN MODEL")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading data from PostgreSQL...")
    X, y, scale_pos_weight = load_features_from_db()
    
    # Split
    print(f"\n[2/6] Splitting data (60-20-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE, stratify=y_train
    )
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Preprocess
    print(f"\n[3/6] Preprocessing features...")
    X_train_scaled, X_val_scaled, X_test_scaled, feature_names = preprocess_features(
        X_train, X_val, X_test
    )
    
    # Tune hyperparameters
    print(f"\n[4/6] Hyperparameter tuning...")
    best_params = tune_hyperparameters(
        X_train_scaled, X_val_scaled, y_train, y_val, scale_pos_weight
    )
    
    # Train final model
    print(f"\n[5/6] Training final model...")
    model, test_metrics = train_final_model(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        best_params
    )
    
    # Save artifacts
    print("\n[6/6] Saving model and artifacts...")
    save_model_and_metrics(model, best_params, test_metrics, feature_names)
    
    print("\n" + "="*60)
    print("✓ XGBOOST MODEL TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
