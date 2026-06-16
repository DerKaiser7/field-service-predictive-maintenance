"""
Train Baseline Logistic Regression Model

Uses the MachineLogisticRegression class to train and evaluate
a Logistic Regression baseline on predictive maintenance features.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data_operations.load import get_engine
from src.models.MachineLogisticRegression import MachineLogisticRegression


def main():
    print("="*60)
    print("BASELINE LOGISTIC REGRESSION MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\n[1/4] Loading features from PostgreSQL...")
    engine = get_engine()
    query = """
    SELECT * FROM model_input_features
    WHERE label IS NOT NULL
    ORDER BY machineid, observation_time
    """
    df = pd.read_sql(query, engine)
    print(f"✓ Loaded {len(df):,} rows")
    
    X = df.drop(columns=['feature_id', 'machineid', 'observation_time', 'label'])
    y = df['label']
    print(f"  Class distribution: {(y==0).sum():,} negative, {(y==1).sum():,} positive")
    
    # Split data
    print("\n[2/4] Train-Val-Test split (60-20-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    print(f"✓ Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Initialize and cross-validate
    print("\n[3/4] Cross-validation (5-fold, PR-AUC)...")
    model = MachineLogisticRegression()
    cv_results = model.cross_validate(X_train, y_train, cv_folds=5)
    print(f"✓ Mean PR-AUC: {cv_results['mean_pr_auc']:.4f} +/- {cv_results['std_pr_auc']:.4f}")
    
    # Fit on train+val
    print("\n[4/4] Training final model on train+val...")
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)
    model.fit(X_combined, y_combined)
    
    # Evaluate on test
    test_metrics = model.evaluate(X_test, y_test)
    print(f"✓ Test PR-AUC: {test_metrics['test_pr_auc']:.4f}")
    print(f"  Precision: {test_metrics['test_precision']:.4f}")
    print(f"  Recall: {test_metrics['test_recall']:.4f}")
    print(f"  F1: {test_metrics['test_f1']:.4f}")
    
    # Save
    print("\nSaving artifacts...")
    model.save(Path("model_artifacts"))
    print("✓ Model saved to models/")
    
    print("\n" + "="*60)
    print("✓ BASELINE TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

