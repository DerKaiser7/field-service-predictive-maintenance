"""API contract tests for src/api/main.py.

Run against the real committed model artifacts (model_artifacts/) rather than
a mock — the artifacts are small and versioned in git specifically so the
API layer can be exercised without a database or a training run. Tests must
be run from the repository root (as `pytest` / `make test` already do) since
the API resolves MODEL_ARTIFACTS as a relative path.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


VALID_FEATURES = {
    "model": "model3",
    "age_category": "aged",
    "age": 15,
    "voltage_mean_3h": 169.5, "voltage_std_3h": 2.4, "voltage_min_3h": 165.0, "voltage_max_3h": 173.0,
    "rotation_mean_3h": 441.0, "rotation_std_3h": 5.5, "rotation_min_3h": 430.0, "rotation_max_3h": 452.0,
    "pressure_mean_3h": 98.0, "pressure_std_3h": 1.6, "pressure_min_3h": 95.0, "pressure_max_3h": 101.0,
    "vibration_mean_3h": 45.0, "vibration_std_3h": 3.8, "vibration_min_3h": 40.0, "vibration_max_3h": 50.0,
    "voltage_mean_12h": 169.8, "voltage_std_12h": 2.3,
    "rotation_mean_12h": 442.0, "rotation_std_12h": 5.2,
    "pressure_mean_12h": 98.2, "pressure_std_12h": 1.5,
    "vibration_mean_12h": 44.5, "vibration_std_12h": 3.7,
    "voltage_mean_24h": 170.0, "voltage_std_24h": 2.2,
    "rotation_mean_24h": 443.0, "rotation_std_24h": 5.0,
    "pressure_mean_24h": 98.5, "pressure_std_24h": 1.4,
    "vibration_mean_24h": 44.0, "vibration_std_24h": 3.6,
    "error_count_24h": 3,
    "hours_since_last_error": 12.0,
    "distinct_error_types": 2,
    "days_since_last_maintenance": 140.0,
    "component_diversity": 3,
    "total_prior_failures": 3,
    "days_since_last_failure": 90.0,
    "distinct_failure_types": 2,
}


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json={
        "machine_id": "5",
        "observation_time": "2026-01-01T00:00:00",
        "features": VALID_FEATURES,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["predicted_label"] in (0, 1)
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert body["model_info"]["ensemble_type"] == "stacked-lr-xgboost"
    assert len(body["top_features"]) > 0


def test_predict_label_matches_threshold(client):
    """predicted_label must be the threshold-applied version of failure_probability
    — a contract the dashboard and any downstream consumer relies on."""
    resp = client.post("/predict", json={
        "machine_id": "5",
        "observation_time": "2026-01-01T00:00:00",
        "features": VALID_FEATURES,
    })
    body = resp.json()
    threshold = body["model_info"]["threshold"]
    expected_label = int(body["failure_probability"] >= threshold)
    assert body["predicted_label"] == expected_label


def test_predict_missing_optional_features_are_imputed(client):
    partial = {k: v for k, v in VALID_FEATURES.items() if k not in ("days_since_last_failure", "component_diversity")}
    resp = client.post("/predict", json={
        "machine_id": "7",
        "observation_time": "2026-01-01T00:00:00",
        "features": partial,
    })
    assert resp.status_code == 200


def test_predict_missing_required_field_is_422(client):
    resp = client.post("/predict", json={
        "observation_time": "2026-01-01T00:00:00",
        "features": VALID_FEATURES,
    })
    assert resp.status_code == 422


def test_batch_predict(client):
    resp = client.post("/batch", json={
        "items": [
            {"machine_id": "5", "observation_time": "2026-01-01T00:00:00", "features": VALID_FEATURES},
            {"machine_id": "7", "observation_time": "2026-01-01T00:00:00", "features": VALID_FEATURES},
        ]
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2


def test_batch_predict_empty_items(client):
    resp = client.post("/batch", json={"items": []})
    assert resp.status_code == 200
    assert resp.json() == {"results": [], "count": 0}


def test_metrics_endpoint_reports_test_metrics(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "test_metrics" in body
    assert "threshold_info" in body
    assert "base_models" in body
