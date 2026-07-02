"""
Train Ensemble Model

Uses EnsembleModel to stack LR + XGBoost, optimise the decision threshold
against a business cost matrix, and evaluate on the held-out test set.
"""

from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_operations.load import get_engine
from src.mlops.ensemble_pyfunc import EnsemblePyfuncModel
from src.mlops.mlflow_utils import configure_mlflow
from src.models.EnsembleModel import EnsembleModel
from src.models.MachineLogisticRegression import MachineLogisticRegression
from src.models.MachineXGBoost import MachineXGBoost

ARTIFACTS = Path("model_artifacts")
REGISTERED_MODEL_NAME = "predictive-maintenance-ensemble"


def main() -> None:
    configure_mlflow()
    print("=" * 60)
    print("ENSEMBLE: STACKED LR-OVER-XGBOOST + THRESHOLD OPTIMISATION")
    print("=" * 60)

    with mlflow.start_run(run_name="stacked_ensemble"):
        print("\n[1/5] Loading data...")
        engine = get_engine()
        df = pd.read_sql(
            "SELECT * FROM model_input_features WHERE label IS NOT NULL ORDER BY machineid, observation_time",
            engine,
        )
        X = df.drop(columns=["feature_id", "machineid", "observation_time", "label"])
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )
        print(f"✓ Val: {len(X_val):,} | Test: {len(X_test):,}")

        print("\n[2/5] Loading base models...")
        ensemble = EnsembleModel()
        ensemble.lr = MachineLogisticRegression.load(ARTIFACTS)
        ensemble.xgb = MachineXGBoost.load(ARTIFACTS)
        print("✓ LR and XGBoost loaded from model_artifacts/")

        print("\n[3/5] Training meta-learner on val-set stacked predictions...")
        ensemble.fit_meta_learner(X_val, y_val)
        print("✓ Meta-learner trained")

        print("\n[4/5] Optimising decision threshold (cost: FN=5, FP=1)...")
        optimal_threshold = ensemble.optimise_threshold(X_val, y_val)
        print(f"✓ Optimal threshold: {optimal_threshold:.3f}")
        mlflow.log_params({
            "cost_fn": EnsembleModel.COST_FN,
            "cost_fp": EnsembleModel.COST_FP,
            "optimal_threshold": optimal_threshold,
        })

        print("\n[5/5] Evaluating on test set...")
        metrics = ensemble.evaluate(X_test, y_test)
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "threshold"})
        print(f"✓ Test PR-AUC:  {metrics['pr_auc']:.4f}")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  F1:           {metrics['f1']:.4f}")
        print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
        print(f"  Business cost at threshold: {metrics['business_cost']:,}")

        print("\nSaving artifacts...")
        ensemble.save(ARTIFACTS)
        print("✓ Saved: ensemble_meta_learner.pkl, optimal_threshold.json, ensemble_metrics.json")

        print("\nRegistering full ensemble (LR + XGBoost + meta-learner) in MLflow...")
        mlflow.pyfunc.log_model(
            artifact_path="ensemble",
            python_model=EnsemblePyfuncModel(),
            artifacts={"model_artifacts": str(ARTIFACTS)},
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_test.head(5),
        )
        print(f"✓ Registered as '{REGISTERED_MODEL_NAME}'")

    print("\n" + "=" * 60)
    print("✓ ENSEMBLE TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
