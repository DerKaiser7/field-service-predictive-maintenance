"""
Feature drift report — Evidently.

Compares the feature distributions the model was trained on (a sample of
model_input_features from Postgres) against what's actually been sent to
/predict in production (prediction_logs.features, logged as JSONB per
prediction since the schema change in sql/schema.sql).

Prediction-output monitoring (mean/std of failure_probability) can't catch
this: a model can keep producing confident-looking probabilities on inputs
that have drifted far from its training distribution. Comparing the raw
feature distributions is the only way to catch that directly.

Run manually (not part of CI — needs a live Postgres with real traffic):

    python -m src.monitoring.drift_report
"""

import json
from pathlib import Path

import pandas as pd
import sqlalchemy
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.data_operations.load import get_engine
from src.models.EnsembleModel import EnsembleModel

MIN_CURRENT_ROWS = 30
REPORT_PATH = Path("monitoring/reports/drift_report.html")
MODEL_ARTIFACTS = Path("model_artifacts")


def get_feature_columns() -> list[str]:
    """Raw (pre-one-hot) feature names — matches what's sent in a /predict
    request body and what model_input_features stores, unlike the post-OHE
    dummy columns in xgboost_feature_importance.csv."""
    ensemble = EnsembleModel.load(MODEL_ARTIFACTS)
    numeric_cols = list(ensemble.lr.imputer.feature_names_in_)
    return numeric_cols + ["model", "age_category"]


def load_reference(engine, feature_cols: list[str], sample_size: int = 5000) -> pd.DataFrame:
    query = f"""
        SELECT {', '.join(feature_cols)}
        FROM model_input_features
        WHERE label IS NOT NULL
        ORDER BY random()
        LIMIT {sample_size}
    """
    return pd.read_sql(query, engine)


def load_current(engine, feature_cols: list[str]) -> pd.DataFrame:
    query = "SELECT features FROM prediction_logs WHERE features IS NOT NULL"
    rows = pd.read_sql(query, engine)
    if rows.empty:
        return pd.DataFrame(columns=feature_cols)

    parsed = pd.json_normalize(rows["features"].apply(
        lambda f: f if isinstance(f, dict) else json.loads(f)
    ))
    return parsed.reindex(columns=feature_cols)


def persist_drift_metrics(engine, result: dict) -> None:
    """Write one row per checked feature into model_monitoring — this table
    was already defined in sql/schema.sql (evidently unused, from before this
    drift check existed) so results are queryable without parsing the HTML
    report."""
    drift_by_column = result["metrics"][1]["result"]["drift_by_columns"]
    rows = [
        {
            "feature_name": name,
            "drift_metric": round(float(col["drift_score"]), 4) if col["drift_score"] is not None else None,
            "alert_triggered": bool(col["drift_detected"]),
            "alert_message": f"{col['stattest_name']} vs training baseline" if col["drift_detected"] else None,
        }
        for name, col in drift_by_column.items()
    ]
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text(
                """
                INSERT INTO model_monitoring (feature_name, drift_metric, alert_triggered, alert_message)
                VALUES (:feature_name, :drift_metric, :alert_triggered, :alert_message)
                """
            ),
            rows,
        )
    print(f"Logged {len(rows)} per-feature drift checks to model_monitoring "
          f"({sum(r['alert_triggered'] for r in rows)} flagged).")


def main() -> None:
    engine = get_engine()
    feature_cols = get_feature_columns()

    print(f"Comparing {len(feature_cols)} features against the training baseline...")

    reference = load_reference(engine, feature_cols)
    current = load_current(engine, feature_cols)

    print(f"Reference (training) rows: {len(reference):,}")
    print(f"Current (production) rows: {len(current):,}")

    if len(current) < MIN_CURRENT_ROWS:
        print(
            f"\nOnly {len(current)} logged predictions with feature data — need at least "
            f"{MIN_CURRENT_ROWS} for a meaningful drift report. Generate some traffic first, "
            f"e.g.: python -m src.monitoring.generate_synthetic_load"
        )
        return

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))
    print(f"\nWrote {REPORT_PATH}")

    result = report.as_dict()
    drift_summary = result["metrics"][0]["result"]
    print(f"Dataset drift detected: {drift_summary['dataset_drift']}")
    print(f"Drifted features: {drift_summary['number_of_drifted_columns']} / {drift_summary['number_of_columns']}")

    persist_drift_metrics(engine, result)


if __name__ == "__main__":
    main()
