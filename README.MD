# Predictive Maintenance System — Azure PdM Dataset

![Tests](https://github.com/DerKaiser7/field-service-predictive-maintenance/actions/workflows/tests.yml/badge.svg)

Binary ensemble classifier that predicts whether an industrial machine will fail within the next 24 hours, enabling proactive maintenance intervention and reducing unplanned downtime.

**[Live Demo](https://field-service-maintenance.streamlit.app) · [Dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)**

> The dashboard's "Live Demo" tab runs real ensemble inference, but on 20 hand-authored machine profiles rather than live database rows — Streamlit Community Cloud has no database access. Telemetry sparklines are simulated waveforms, not sampled sensor data. See [Design Notes](#design-notes) below.

---

## Problem

Unplanned machine failures are significantly more costly than scheduled maintenance — both in repair cost and lost production time. This system uses historical sensor readings, error logs, and maintenance records to predict failures 24 hours in advance.

**ML Task:** Binary classification — predict `label=1` if a failure occurs within 24 hours of a given observation, `label=0` otherwise.

**Dataset:** Microsoft Azure Predictive Maintenance — 100 machines, 876,100 hourly telemetry observations over one year.

---

## Architecture

Raw CSVs → PostgreSQL staging → base tables → SQL feature build → LR + XGBoost base models → stacked ensemble → committed `model_artifacts/` → FastAPI service and Streamlit demo.

Full pipeline diagram and deployment topology: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Data

| Table | Rows | Description |
| --- | --- | --- |
| `machines` | 100 | Machine metadata (model, age) |
| `telemetry` | 876,100 | Hourly sensor readings (voltage, rotation, pressure, vibration) |
| `errors` | 3,919 | Error events per machine |
| `maintenance` | 3,286 | Maintenance history by component |
| `failures` | 761 | Failure events with failure type |

**Class balance:** 17,184 positive labels (1.96%) — handled via `scale_pos_weight=49.98` in XGBoost and `class_weight='balanced'` in Logistic Regression.

---

## Features

42 engineered features per observation, all computed strictly from data available at or before `observation_time` (no leakage):

| Category | Features |
| --- | --- |
| Static | `model`, `age`, `age_category` |
| Telemetry 3h | mean/std/min/max for voltage, rotation, pressure, vibration (16 features) |
| Telemetry 12h | mean/std for all four sensors (8 features) |
| Telemetry 24h | mean/std for all four sensors (8 features) |
| Errors | `error_count_24h`, `hours_since_last_error`, `distinct_error_types` |
| Maintenance | `days_since_last_maintenance`, `component_diversity` |
| Failure history | `total_prior_failures`, `days_since_last_failure`, `distinct_failure_types` |

**Label:** `1` if failure occurs in `(t, t+24h]`, else `0`.

See [docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md) for full rationale.

---

## Modelling

**Three-stage pipeline:**

1. **Baseline — Logistic Regression** (`class_weight='balanced'`, 5-fold CV): Sanity check and meta-learner input.

2. **Main — XGBoost** (`scale_pos_weight=49.98`, grid search over depth/lr/estimators): boosted model tuned on PR-AUC.

3. **Ensemble — Stacked meta-learner**: Logistic Regression trained on val-set predictions from LR + XGBoost. Threshold optimised against cost matrix: missed failure (FN) penalised 5× over false alarm (FP).

**Primary metric:** PR-AUC (preferred over ROC-AUC for imbalanced data).

---

## Results

Train/val/test split: 60/20/20 (stratified). All metrics on held-out test set.

| Model | PR-AUC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Logistic Regression (baseline) | 0.8285 | 0.4735 | 0.9895 | 0.6405 |
| XGBoost | 0.9999 | 0.9840 | 1.0000 | 0.9919 |
| Ensemble (threshold=0.729) | 0.9992 | 0.9888 | 0.9985 | 0.9936 |

**Ensemble at optimal threshold (cost: 5×FN + 1×FP):**

- True Positives: 3,432 — True Negatives: 171,744
- False Positives: 39 — False Negatives: 5
- Business cost: 64

---

## MLOps Monitoring

Three pieces, each addressing a different gap between "a model that scores well offline" and "a model someone can operate":

- **MLflow — experiment tracking + model registry.** `train_baseline.py`, `train_xgboost.py`, and `ensemble.py` each log params/metrics/model artifacts to a dedicated `mlflow` service (Postgres-backed, proxied artifact storage). The final ensemble is registered as `predictive-maintenance-ensemble` via `src/mlops/ensemble_pyfunc.py` — a pyfunc wrapper, since the "model" is really three coupled artifacts (LR, XGBoost, meta-learner) that only make sense registered together, not as a single sklearn object.
- **Prometheus + Grafana — service monitoring.** The FastAPI service is instrumented with `prometheus-fastapi-instrumentator`, exposing request rate/latency/status metrics at `/metrics/prometheus` (kept separate from the existing `/metrics` endpoint, which returns model metadata, not service metrics). Grafana ships with a provisioned dashboard reading from Prometheus.
- **Evidently — feature drift detection.** `prediction_logs` now stores the input feature vector per prediction (`features JSONB` column), not just the output. `src/monitoring/drift_report.py` compares logged production features against a sample of the training baseline, writes an HTML drift report, and logs one row per feature to `model_monitoring` (a table that existed in `sql/schema.sql` from early on but had nothing writing to it until now). Output-only monitoring (mean/std of `failure_probability`) can't catch a model that's still confidently wrong on inputs that have drifted away from what it was trained on.

Neither the API nor a real machine fleet has production traffic to show in a demo, so `src/monitoring/generate_synthetic_load.py` fires simulated `/predict` requests — with a deliberate drift ramp partway through, simulating a fleet aging over time — so Grafana and the drift report have real data to display rather than empty charts.

```bash
# with the full stack running (docker compose up -d):
python -m src.monitoring.generate_synthetic_load --requests 200
python -m src.monitoring.drift_report
```

| Service | URL |
| --- | --- |
| MLflow tracking UI | <http://localhost:5001> |
| Prometheus | <http://localhost:9090> |
| Grafana (anonymous viewer access) | <http://localhost:3000> |

---

## Repository Structure

```text
.
├── dashboard/
│   ├── app.py                       # Streamlit demo
│   └── .streamlit/config.toml       # Theme config
├── docs/
│   ├── ARCHITECTURE.md
│   └── FEATURE_ENGINEERING.md
├── model_artifacts/                 # Trained model files (committed)
│   ├── baseline_lr.pkl
│   ├── xgboost_main.pkl
│   ├── ensemble_meta_learner.pkl
│   ├── optimal_threshold.json
│   └── leakage_ablation.json
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/provisioning/       # datasource + dashboard, auto-loaded on container start
├── mlflow/
│   └── Dockerfile                   # mlflow server image (mlflow + psycopg2-binary)
├── sql/
│   ├── schema.sql
│   ├── eda/                         # Exploratory queries (00–06)
│   └── features/build_features.sql
├── src/
│   ├── data_operations/
│   │   ├── load.py
│   │   └── promote_to_base.py
│   ├── models/
│   │   ├── BaseMachineModel.py
│   │   ├── MachineLogisticRegression.py
│   │   ├── MachineXGBoost.py
│   │   ├── EnsembleModel.py
│   │   ├── train_baseline.py
│   │   ├── train_xgboost.py
│   │   ├── ensemble.py
│   │   └── ablation_leakage_check.py
│   ├── mlops/
│   │   ├── mlflow_utils.py
│   │   └── ensemble_pyfunc.py       # registers the LR+XGBoost+meta-learner trio as one MLflow model
│   ├── monitoring/
│   │   ├── drift_report.py
│   │   └── generate_synthetic_load.py
│   └── api/
│       ├── main.py                  # FastAPI serving layer
│       ├── Dockerfile
│       └── requirements.txt         # API-only dependency set (no streamlit/mlflow/etc.)
├── docker-compose.yml                # Postgres, API, MLflow, Prometheus, Grafana
├── requirements.txt
└── README.MD
```

---

## Design Notes

Trade-offs made deliberately for a demo context:

- **Model artifacts are committed to git** (`model_artifacts/`, ~624KB total) so the Streamlit demo and API can run without training anything first. MLflow now provides a real model registry for anyone running the full `docker compose` stack locally — the two aren't in tension: git holds what's needed to *run* the demo, MLflow holds run history and versioned models for anyone iterating on training.
- **Live Demo profiles are hardcoded**, not queried from a database. Streamlit Community Cloud's free tier has no database access, so `dashboard/app.py` ships 20 machine profiles designed to span the LOW/MEDIUM/HIGH risk spectrum. The ensemble inference and SHAP explanations on those profiles are real; the input data is not.
- **CORS is fully open** (`allow_origins=["*"]`) in `src/api/main.py`. Demo API with no auth surface — would be scoped to specific origins before handling real traffic.
- **XGBoost's near-perfect test PR-AUC (0.9999)** was treated as a leakage red flag during development, not just a good result. `src/models/ablation_leakage_check.py` retrains the model with `total_prior_failures`, `days_since_last_failure`, and `distinct_failure_types` removed — these are the features most structurally similar to the label, since both derive from the `failures` table. Test PR-AUC drops by only 0.0003 (0.9999 → 0.9995, FN: 0 → 4), meaning the score is driven by telemetry/error signal, not by those features leaking future failure information. `tests/test_feature_leakage.py` adds a static regression guard against the most likely leakage mistake (a feature CTE picking up the label's forward-looking window).
- **MLflow's host port is 5001, not the default 5000.** Port 5000 is claimed by macOS's AirPlay Receiver (ControlCenter) by default on most Macs; the container's internal port is still 5000.
- **Monitoring runs on synthetic traffic, not real usage.** Neither the API nor a real fleet has production traffic — `src/monitoring/generate_synthetic_load.py` exists specifically so Grafana and the Evidently drift report have real data to display rather than empty charts. The drift it detects (simulated fleet aging) is manufactured, not observed.

## Testing

```bash
pytest tests/
```

29 tests cover: preprocessing/imputation/one-hot alignment (`BaseMachineModel`), fit → predict → save → load roundtrips for each model class, ensemble stacking and cost-threshold optimisation, a static leakage-regression guard on `sql/features/build_features.sql`, and FastAPI contract tests run against the real committed model artifacts (no mocking, no database required). Runs on every push/PR via `.github/workflows/tests.yml`.

## Running Locally

**Prerequisites:** Docker, Python 3.11, `.env` file with DB credentials.

```bash
# 1. Start the full stack: Postgres, API, MLflow, Prometheus, Grafana
docker compose up -d

# 2. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Load data
python -m src.data_operations.load
python -m src.data_operations.promote_to_base

# 4. Build features
psql -h localhost -U $POSTGRES_USER -d $POSTGRES_DB -f sql/features/build_features.sql

# 5. Train models — logs params/metrics/artifacts to the mlflow service (http://localhost:5001)
export MLFLOW_TRACKING_URI=http://localhost:5001
python -m src.models.train_baseline
python -m src.models.train_xgboost
python -m src.models.ensemble   # also registers 'predictive-maintenance-ensemble' in the MLflow Model Registry

# 6. Run Streamlit demo (uses committed model artifacts — no DB required)
streamlit run dashboard/app.py

# 7. Run FastAPI (optional) — locally...
uvicorn src.api.main:app --port 8000

# ...or as a container alongside the rest of the stack (already started in step 1)
docker compose up -d --build api

# 8. Generate synthetic traffic + a drift report (optional — see MLOps Monitoring below)
python -m src.monitoring.generate_synthetic_load --requests 200
python -m src.monitoring.drift_report
```
