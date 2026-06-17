"""
FastAPI Prediction Service — Phase 5

Endpoints:
  POST /predict  — single-machine failure prediction with SHAP explanations
  POST /batch    — bulk predictions (no SHAP) for dashboard use
  GET  /metrics  — ensemble model metadata and test-set performance
  GET  /health   — liveness check
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data_operations.load import get_engine
from src.models.EnsembleModel import EnsembleModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ARTIFACTS = Path("model_artifacts")
MODEL_VERSION = "1.0-ensemble"
TOP_K_FEATURES = 5

_ensemble: EnsembleModel = None  # type: ignore[assignment]  # set in lifespan
_explainer: shap.TreeExplainer = None  # type: ignore[assignment]  # set in lifespan
_xgb_feature_names: list[str] = []
_numeric_col_order: list[str] = []   # imputer's expected column order, set in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ensemble, _explainer, _xgb_feature_names, _numeric_col_order

    logger.info("Loading EnsembleModel from %s", MODEL_ARTIFACTS)
    _ensemble = EnsembleModel.load(MODEL_ARTIFACTS)

    # Column order the imputer was fitted on (numeric cols only, pre-OHE)
    _numeric_col_order = list(_ensemble.lr.imputer.feature_names_in_)  # type: ignore[union-attr]

    # XGBoost 2.x stores feature names used during training on the booster
    booster = _ensemble.xgb.model.get_booster()
    _xgb_feature_names = booster.feature_names or []

    logger.info("Building SHAP TreeExplainer on XGBoost base model")
    _explainer = shap.TreeExplainer(_ensemble.xgb.model)

    logger.info("API ready — threshold=%.3f", _ensemble.threshold)
    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Kaiser Intelligence — Predictive Maintenance API",
    description="Binary failure classifier predicting machine failures 24 h in advance.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    machine_id: str = Field(..., description="Machine identifier (e.g. '1')")
    observation_time: datetime = Field(..., description="ISO-8601 timestamp of the observation")
    features: dict[str, Any] = Field(
        ...,
        description=(
            "Feature values keyed by feature name. "
            "Missing numeric features are imputed with the training-set mean. "
            "Categorical features: 'model' (A–D), 'age_category' (new/mid-life/aged)."
        ),
    )


class FeatureImpact(BaseModel):
    name: str
    impact: float = Field(..., description="SHAP value: positive=increases failure risk")
    value: float | str | None = Field(None, description="Raw feature value")


class ModelInfo(BaseModel):
    version: str
    ensemble_type: str
    threshold: float
    threshold_rationale: str


class PredictionResponse(BaseModel):
    machine_id: str
    observation_time: datetime
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_label: int = Field(..., description="1 = failure expected within 24 h")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH")
    top_features: list[FeatureImpact]
    model_info: ModelInfo


class BatchItem(BaseModel):
    machine_id: str
    observation_time: datetime
    features: dict[str, Any]


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., max_length=1000)


class BatchResultItem(BaseModel):
    machine_id: str
    observation_time: datetime
    failure_probability: float
    predicted_label: int
    risk_level: str


class BatchResponse(BaseModel):
    results: list[BatchResultItem]
    count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_input_df(features: dict[str, Any]) -> pd.DataFrame:
    """
    Build a model-ready DataFrame from a raw feature dict.

    Columns are emitted in the order the imputer was fitted on: numeric
    columns first (in training order), then categorical columns. This
    prevents sklearn's column-order validation from raising on inference.
    Python None → np.nan keeps numeric columns as float64.
    """
    categorical_keys = {"model", "age_category"}
    row: dict[str, Any] = {}

    # Numeric columns in training order (imputer-expected)
    for col in _numeric_col_order:
        v = features.get(col)
        row[col] = np.nan if v is None else v

    # Categorical columns
    for k in categorical_keys:
        if k in features:
            row[k] = features[k]

    return pd.DataFrame([row])


def _risk_level(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "HIGH"
    if prob >= threshold * 0.5:
        return "MEDIUM"
    return "LOW"


def _shap_top_k(X_raw: pd.DataFrame, k: int = TOP_K_FEATURES) -> list[FeatureImpact]:
    """
    Compute SHAP values via the XGBoost base model.

    Preprocessing (impute + scale) is applied before SHAP so that the
    explainer sees the same feature space the model was trained on.
    Categorical one-hot columns produced by get_dummies are included;
    the top-k most impactful original feature values are returned.
    """
    X_proc = _ensemble.xgb.preprocess_features(X_raw, fit=False)  # type: ignore[union-attr]

    # Align columns to what XGBoost was trained on (fill any missing dummy columns with 0)
    if _xgb_feature_names:
        for col in _xgb_feature_names:
            if col not in X_proc.columns:
                X_proc[col] = 0.0
        X_proc = X_proc[_xgb_feature_names]

    sv = _explainer(X_proc)  # type: ignore[misc]
    # sv.values shape: (1, n_features) for binary classification in SHAP 0.46+
    shap_vals = np.abs(sv.values[0])
    feature_names = X_proc.columns.tolist()

    top_indices = np.argsort(shap_vals)[::-1][:k]
    result: list[FeatureImpact] = []
    for idx in top_indices:
        fname = feature_names[idx]
        raw_val = X_raw.iloc[0].get(fname)
        result.append(
            FeatureImpact(
                name=fname,
                impact=float(sv.values[0][idx]),
                value=float(raw_val) if isinstance(raw_val, (int, float, np.number)) else raw_val,
            )
        )
    return result


def _log_prediction(machine_id: str, obs_time: datetime, prob: float, label: int) -> None:
    """Insert a row into prediction_logs (runs as a FastAPI background task)."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(
                __import__("sqlalchemy").text(
                    """
                    INSERT INTO prediction_logs
                        (machineid, observation_time, failure_probability, predicted_label, model_version)
                    VALUES (:mid, :obs, :prob, :label, :ver)
                    """
                ),
                {
                    "mid": machine_id,
                    "obs": obs_time,
                    "prob": prob,
                    "label": label,
                    "ver": MODEL_VERSION,
                },
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Prediction log write failed: %s", exc)


def _read_threshold_rationale() -> str:
    path = MODEL_ARTIFACTS / "optimal_threshold.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get("rationale", "")
    return ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["infra"])
def health() -> dict[str, str]:
    if _ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(req: PredictRequest, background_tasks: BackgroundTasks) -> PredictionResponse:
    if _ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = _build_input_df(req.features)

    try:
        prob = float(_ensemble.predict_proba(X))
    except Exception as exc:
        logger.error("predict_proba failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    label = int(prob >= _ensemble.threshold)
    risk = _risk_level(prob, _ensemble.threshold)

    try:
        top_features = _shap_top_k(X)
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        top_features = []

    background_tasks.add_task(
        _log_prediction, req.machine_id, req.observation_time, prob, label
    )

    return PredictionResponse(
        machine_id=req.machine_id,
        observation_time=req.observation_time,
        failure_probability=round(prob, 6),
        predicted_label=label,
        risk_level=risk,
        top_features=top_features,
        model_info=ModelInfo(
            version=MODEL_VERSION,
            ensemble_type="stacked-lr-xgboost",
            threshold=_ensemble.threshold,
            threshold_rationale=_read_threshold_rationale(),
        ),
    )


@app.post("/batch", response_model=BatchResponse, tags=["prediction"])
def batch_predict(req: BatchRequest) -> BatchResponse:
    if _ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.items:
        return BatchResponse(results=[], count=0)

    # Build each row via _build_input_df so None→nan and column order are consistent
    X_all = pd.concat([_build_input_df(item.features) for item in req.items], ignore_index=True)

    try:
        probs = _ensemble.predict_proba(X_all)
    except Exception as exc:
        logger.error("batch predict_proba failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Batch prediction failed: {exc}") from exc

    results: list[BatchResultItem] = []
    for item, prob in zip(req.items, probs):
        label = int(float(prob) >= _ensemble.threshold)
        results.append(
            BatchResultItem(
                machine_id=item.machine_id,
                observation_time=item.observation_time,
                failure_probability=round(float(prob), 6),
                predicted_label=label,
                risk_level=_risk_level(float(prob), _ensemble.threshold),
            )
        )

    return BatchResponse(results=results, count=len(results))


@app.get("/metrics", tags=["model"])
def metrics() -> dict[str, Any]:
    """Return ensemble test-set metrics and model metadata."""
    payload: dict[str, Any] = {"version": MODEL_VERSION}

    metrics_path = MODEL_ARTIFACTS / "ensemble_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            payload["test_metrics"] = json.load(f)

    threshold_path = MODEL_ARTIFACTS / "optimal_threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            payload["threshold_info"] = json.load(f)

    if _ensemble is not None:
        payload["base_models"] = {
            "logistic_regression": _ensemble.lr.metrics if _ensemble.lr else {},
            "xgboost": _ensemble.xgb.metrics if _ensemble.xgb else {},
        }

    return payload
