"""
Leakage ablation check.

Retrains XGBoost with the failure-history features removed
(total_prior_failures, days_since_last_failure, distinct_failure_types)
and compares test PR-AUC against the full feature set, using identical
hyperparameters (model_artifacts/xgboost_hyperparams.json) and the same
train/val/test split as the committed model.

Purpose: the committed model's 0.9999 test PR-AUC is high enough to be a
leakage red flag on its own. If PR-AUC barely moves once failure-history
features are removed, the score is coming from telemetry/error signal
rather than from those features encoding the label. If it collapses,
that's evidence worth investigating further.

This is a diagnostic script, not part of the training pipeline — it needs
the full 876k-row Postgres table and is run manually, not in CI.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_operations.load import get_engine
from src.models.MachineXGBoost import MachineXGBoost

FAILURE_HISTORY_COLS = [
    "total_prior_failures",
    "days_since_last_failure",
    "distinct_failure_types",
]


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    engine = get_engine()
    query = """
    SELECT * FROM model_input_features
    WHERE label IS NOT NULL
    ORDER BY machineid, observation_time
    """
    df = pd.read_sql(query, engine)
    X = df.drop(columns=["feature_id", "machineid", "observation_time", "label"])
    y = df["label"]
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_evaluate(X_train, X_val, y_train, y_val, X_test, y_test, hyperparams: dict) -> dict:
    model = MachineXGBoost(
        random_state=hyperparams["random_state"],
        max_depth=hyperparams["max_depth"],
        learning_rate=hyperparams["learning_rate"],
        n_estimators=hyperparams["n_estimators"],
        scale_pos_weight=hyperparams["scale_pos_weight"],
    )
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)
    model.fit(X_combined, y_combined)
    return model.evaluate(X_test, y_test)


def main() -> None:
    with open("model_artifacts/xgboost_hyperparams.json") as f:
        hyperparams = json.load(f)

    print("Loading features from PostgreSQL...")
    X, y = load_features()
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)

    print("\n[1/2] Full feature set (matches committed model)...")
    full_metrics = train_and_evaluate(X_train, X_val, y_train, y_val, X_test, y_test, hyperparams)
    print(f"  Test PR-AUC: {full_metrics['test_pr_auc']:.4f}  |  Recall: {full_metrics['test_recall']:.4f}  |  FN: {int(full_metrics['test_fn'])}")

    print("\n[2/2] Failure-history features removed...")
    drop_cols = [c for c in FAILURE_HISTORY_COLS if c in X_train.columns]
    ablated_metrics = train_and_evaluate(
        X_train.drop(columns=drop_cols), X_val.drop(columns=drop_cols), y_train, y_val,
        X_test.drop(columns=drop_cols), y_test, hyperparams,
    )
    print(f"  Test PR-AUC: {ablated_metrics['test_pr_auc']:.4f}  |  Recall: {ablated_metrics['test_recall']:.4f}  |  FN: {int(ablated_metrics['test_fn'])}")

    result = {
        "dropped_features": FAILURE_HISTORY_COLS,
        "full_feature_set": full_metrics,
        "failure_history_removed": ablated_metrics,
        "pr_auc_drop": round(full_metrics["test_pr_auc"] - ablated_metrics["test_pr_auc"], 4),
    }
    out_path = Path("model_artifacts/leakage_ablation.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")
    print(f"PR-AUC drop from removing failure-history features: {result['pr_auc_drop']:.4f}")


if __name__ == "__main__":
    main()
