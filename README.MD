# Predictive Maintenance System

Binary classifier that predicts whether a machine will fail within the next 24 hours, enabling proactive intervention and reducing unplanned downtime.

---

## Problem

[Brief description of the business context and why reactive maintenance is costly]

**ML Task:** Binary classification — predict failure within a 24-hour horizon per machine observation.

---

## Architecture

[Architecture diagram — to be added]

**Pipeline overview:**
1. Raw telemetry → PostgreSQL (staging)
2. SQL feature engineering → `model_input_features`
3. Offline model training (XGBoost/LightGBM)
4. FastAPI serving — real-time predictions
5. Prediction logging → Postgres
6. Monitoring — basic drift detection

---

## Data

| Table | Description |
|---|---|
| `machines` | Machine metadata (model, age) |
| `telemetry` | Sensor readings (voltage, rotation, pressure, vibration) |
| `errors` | Error events per machine |
| `maintenance` | Maintenance history by component |
| `failures` | Failure events with failure type |

---

## Features

- **Static:** machine model, age
- **Telemetry:** rolling mean/std/min/max over 3h, 12h, 24h windows
- **Error:** counts, time since last error, distinct error types
- **Maintenance:** days since last maintenance, component history
- **Failure history:** total prior failures, days since last failure

**Label:** `1` if failure occurs in `(t, t+24h]`, else `0`. Features strictly limited to data available at time `t`.

---

## Modelling

| Stage | Model |
|---|---|
| Baseline | Logistic Regression |
| Main | XGBoost / LightGBM |

**Primary metric:** PR AUC  
**Threshold:** tuned against cost of missed failures vs. unnecessary maintenance

---

## Results

[To be completed]

| Metric | Score |
|---|---|
| PR AUC | — |
| Recall | — |
| Precision | — |
| F1 | — |

---

## Repository Structure

