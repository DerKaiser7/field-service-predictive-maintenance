"""
Train XGBoost Main Model

Uses the MachineXGBoost class to train and evaluate
an XGBoost model with hyperparameter tuning on predictive maintenance features.
"""

import mlflow
import mlflow.xgboost
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data_operations.load import get_engine
from src.mlops.mlflow_utils import configure_mlflow
from src.models.MachineXGBoost import MachineXGBoost


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute scale_pos_weight for class imbalance handling."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return n_neg / n_pos if n_pos > 0 else 1.0


def main():
    configure_mlflow()
    print("="*60)
    print("XGBOOST MAIN MODEL TRAINING")
    print("="*60)

    with mlflow.start_run(run_name="xgboost_main"):
        # Load data
        print("\n[1/5] Loading features from PostgreSQL...")
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
        n_neg = (y==0).sum()
        n_pos = (y==1).sum()
        print(f"  Class distribution: {n_neg:,} negative, {n_pos:,} positive")
        scale_pos_weight = compute_scale_pos_weight(y)
        print(f"  Scale pos weight: {scale_pos_weight:.2f}")
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        # Split data
        print("\n[2/5] Train-Val-Test split (60-20-20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )
        print(f"✓ Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

        # Initialize with scale_pos_weight
        print("\n[3/5] Hyperparameter tuning (grid search)...")
        model = MachineXGBoost(scale_pos_weight=scale_pos_weight)

        param_grid = {
            'max_depth': [5, 6, 7],
            'learning_rate': [0.01, 0.05],
            'n_estimators': [100, 200],
        }

        best_params = model.tune_hyperparameters(
            X_train, X_val, y_train, y_val, param_grid
        )
        print(f"✓ Best params: depth={best_params['max_depth']} lr={best_params['learning_rate']} n_est={best_params['n_estimators']}")
        mlflow.log_params({
            "max_depth": best_params["max_depth"],
            "learning_rate": best_params["learning_rate"],
            "n_estimators": best_params["n_estimators"],
        })

        # Fit on train+val
        print("\n[4/5] Training final model on train+val...")
        X_combined = pd.concat([X_train, X_val], ignore_index=True)
        y_combined = pd.concat([y_train, y_val], ignore_index=True)
        model.fit(X_combined, y_combined)

        # Evaluate on test
        test_metrics = model.evaluate(X_test, y_test)
        mlflow.log_metrics(test_metrics)
        print(f"✓ Test PR-AUC: {test_metrics['test_pr_auc']:.4f}")
        print(f"  Precision: {test_metrics['test_precision']:.4f}")
        print(f"  Recall: {test_metrics['test_recall']:.4f}")
        print(f"  F1: {test_metrics['test_f1']:.4f}")

        input_example = model.preprocess_features(X_combined.head(5), fit=False)
        mlflow.xgboost.log_model(model.model, artifact_path="model", input_example=input_example)

        # Save
        print("\n[5/5] Saving artifacts...")
        model.save(Path("model_artifacts"))
        print("✓ Model saved to models/")

    print("\n" + "="*60)
    print("✓ XGBOOST TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
