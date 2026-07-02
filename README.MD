# Predictive Maintenance System — Azure PdM Dataset

![Tests](https://github.com/DerKaiser7/field-service-predictive-maintenance/actions/workflows/tests.yml/badge.svg)

Binary ensemble classifier that predicts whether an industrial machine will fail within the next 24 hours, enabling proactive maintenance intervention and reducing unplanned downtime.

**[Live Demo](https://field-service-maintenance.streamlit.app) · [Dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)**

---

## Problem

Unplanned machine failures are significantly more costly than scheduled maintenance — both in repair cost and lost production time. This system uses historical sensor readings, error logs, and maintenance records to predict failures 24 hours in advance.

**ML Task:** Binary classification — predict `label=1` if a failure occurs within 24 hours of a given observation, `label=0` otherwise.

**Dataset:** Microsoft Azure Predictive Maintenance — 100 machines, 876,100 hourly telemetry observations over one year.

---

## Architecture

```text
Raw CSVs (telemetry, errors, maintenance, failures, machines)
    ↓  src/data_operations/load.py
Staging Tables (PostgreSQL)
    ↓  src/data_operations/promote_to_base.py
Base Tables (machines, telemetry, errors, maintenance, failures)
    ↓  sql/features/build_features.sql
model_input_features (876,100 rows, 42 features + label)
    ↓
Model Training
    ├── src/models/train_baseline.py  →  Logistic Regression
    ├── src/models/train_xgboost.py   →  XGBoost
    └── src/models/ensemble.py        →  Stacked ensemble + threshold optimisation
    ↓  model_artifacts/
FastAPI Prediction Service  (src/api/main.py)
    ↓
Streamlit Demo  (dashboard/app.py)
```

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

2. **Main — XGBoost** (`scale_pos_weight=49.98`, grid search over depth/lr/estimators): Production-grade boosted model tuned on PR-AUC.

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

## Repository Structure

```text
.
├── dashboard/
│   ├── app.py                       # Streamlit demo
│   └── .streamlit/config.toml       # Theme config
├── docs/
│   └── FEATURE_ENGINEERING.md
├── model_artifacts/                 # Trained model files (committed)
│   ├── baseline_lr.pkl
│   ├── xgboost_main.pkl
│   ├── ensemble_meta_learner.pkl
│   └── optimal_threshold.json
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
│   │   └── ensemble.py
│   └── api/main.py                  # FastAPI serving layer
├── requirements.txt
└── README.MD
```

---

## Testing

```bash
pytest tests/
```

29 tests cover: preprocessing/imputation/one-hot alignment (`BaseMachineModel`), fit → predict → save → load roundtrips for each model class, ensemble stacking and cost-threshold optimisation, a static leakage-regression guard on `sql/features/build_features.sql`, and FastAPI contract tests run against the real committed model artifacts (no mocking, no database required). Runs on every push/PR via `.github/workflows/tests.yml`.

## Running Locally

**Prerequisites:** Docker, Python 3.11, `.env` file with DB credentials.

```bash
# 1. Start PostgreSQL
docker compose up -d

# 2. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Load data
python -m src.data_operations.load
python -m src.data_operations.promote_to_base

# 4. Build features
psql -h localhost -U $POSTGRES_USER -d $POSTGRES_DB -f sql/features/build_features.sql

# 5. Train models
python -m src.models.train_baseline
python -m src.models.train_xgboost
python -m src.models.ensemble

# 6. Run Streamlit demo (uses committed model artifacts — no DB required)
streamlit run dashboard/app.py

# 7. Run FastAPI (optional)
uvicorn src.api.main:app --port 8000
```
