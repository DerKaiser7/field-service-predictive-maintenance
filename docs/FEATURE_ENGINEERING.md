# Feature Engineering Strategy

## Overview

This document explains the feature engineering pipeline for the predictive maintenance system. The goal is to create a robust set of features that:

1. **Prevent data leakage** — use only information available at the observation time
2. **Capture temporal patterns** — rolling aggregations at multiple timescales
3. **Enable interpretability** — features correspond to meaningful machine diagnostics

**Key Design Principle:** Every feature uses data strictly <= observation_time. No future data is used in feature creation.

---

## Data Flow

```text
Raw Data (staging tables)
    ↓
Base Tables (machines, telemetry, errors, maintenance, failures)
    ↓
Feature Engineering (build_features.sql)
    ↓
model_input_features table (876,100 rows, 42 features + label)
```

---

## Feature Categories

### 1. Static Features (Non-temporal)

**Source:** `machines` table

| Feature | Type | Rationale |
| --- | --- | --- |
| `model` | VARCHAR | Machine type affects failure modes (e.g., model A fails earlier than model B) |
| `age` | INTEGER | Older machines more likely to fail |
| `age_category` | VARCHAR | Binned age into 3 categories: new (0-3y), mid-life (3-7y), aged (7+y). Captures non-linear age effects |

**Implementation:** Simple CASE statement on age.

---

### 2. Telemetry Features (Time-Series Aggregations)

**Source:** `telemetry` table (voltage, rotation, pressure, vibration)

**Concept:** For each observation time `t`, compute rolling aggregations of sensor readings from past time windows.

**Windows:** 3-hour, 12-hour, 24-hour lookback periods

**Aggregations per window:**

- Mean (central tendency)
- Standard deviation (variability)
- Min/Max (extremes) — *only for 3h window to reduce feature count*

**Why these windows?**

- **3h:** Captures short-term anomalies (e.g., vibration spike this morning)
- **12h:** Medium-term trends (e.g., gradual pressure increase)
- **24h:** Long-term patterns and daily cycles

**Why these aggregations?**

- **Mean:** Overall sensor operating point
- **Std:** Variability indicates instability (high variance = problem)
- **Min/Max:** Extremes identify transient faults

**Telemetry Features Created:**

For each sensor (`voltage`, `rotation`, `pressure`, `vibration`):

- 3h: `{sensor}_mean_3h`, `{sensor}_std_3h`, `{sensor}_min_3h`, `{sensor}_max_3h` (4 features × 4 sensors = 16 features)
- 12h: `{sensor}_mean_12h`, `{sensor}_std_12h` (2 features × 4 sensors = 8 features)
- 24h: `{sensor}_mean_24h`, `{sensor}_std_24h` (2 features × 4 sensors = 8 features)

**Total: 32 telemetry features**

**Implementation Detail:**

```sql
AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' 
         AND t2.datetime <= t.datetime 
    THEN t2.volt END) AS voltage_mean_3h
```

This uses a self-join on the telemetry table, filtering readings to the 3-hour window **ending at** observation_time.

---

### 3. Error Features (Event History)

**Source:** `errors` table

| Feature | Rationale |
| --- | --- |
| `error_count_24h` | # errors in past 24h. Frequent errors = higher failure risk |
| `hours_since_last_error` | Time decay: recent errors more predictive than old ones |
| `distinct_error_types` | Error diversity. Multiple error types = complex issues |

**Implementation:**

```sql
COUNT(CASE WHEN e.datetime > t.datetime - INTERVAL '24 hours' 
           AND e.datetime <= t.datetime THEN 1 END) AS error_count_24h,

EXTRACT(EPOCH FROM (t.datetime - MAX(e.datetime))) / 3600.0 AS hours_since_last_error,

COUNT(DISTINCT e.errorID) AS distinct_error_types
```

**Edge Case:** If no errors in history, `hours_since_last_error` is NULL (handled in model training).

---

### 4. Maintenance Features (Service History)

**Source:** `maintenance` table

| Feature | Rationale |
| --- | --- |
| `days_since_last_maintenance` | Time since last maintenance. Longer intervals = degradation |
| `component_diversity` | # distinct components serviced. Indicates system complexity or repeated failures |

**Implementation:**

```sql
EXTRACT(EPOCH FROM (t.datetime - MAX(m.datetime))) / 86400.0 AS days_since_last_maintenance,

COUNT(DISTINCT m.comp) AS component_diversity
```

**Business Context:** If machine never had maintenance, `days_since_last_maintenance` is NULL (indicates data gap or new installation).

---

### 5. Failure History Features (Past Failures)

**Source:** `failures` table

| Feature | Rationale |
| --- | --- |
| `total_prior_failures` | # past failures. Repeated failures = unreliable machine |
| `days_since_last_failure` | Time since last failure. Recent failures = current vulnerability |
| `distinct_failure_types` | Failure diversity. Multiple failure types = systemic issues |

**Implementation:**

```sql
COUNT(CASE WHEN f.datetime <= t.datetime THEN 1 END) AS total_prior_failures,

EXTRACT(EPOCH FROM (t.datetime - MAX(f.datetime))) / 86400.0 AS days_since_last_failure,

COUNT(DISTINCT f.failure) AS distinct_failure_types
```

**Temporal Cutoff:** Only count failures **at or before** `observation_time`, never future failures.

---

## Label Engineering (Binary Target)

**Question:** Will this machine fail in the next 24 hours?

**Implementation:**

```sql
CASE WHEN EXISTS (
    SELECT 1 FROM failures fut
    WHERE fut.machineid = t.machineid
      AND fut.datetime > t.observation_time
      AND fut.datetime <= t.observation_time + INTERVAL '24 hours'
) THEN 1 ELSE 0 END AS label
```

**Key Properties:**

1. **Strictly future-looking:** Checks for failures AFTER observation_time
2. **24-hour window:** Matches the business requirement (predictive maintenance horizon)
3. **Leakage prevention:** The EXISTS subquery only executes after all features are computed, ensuring no feature "sees" the label

**Interpretation:**

- `label = 1`: Failure occurred within 24h after observation_time (positive example)
- `label = 0`: No failure in 24h window (negative example)

**Class Imbalance:** 1.96% positive class (17,184 failures out of 876,100 observations). Handled via `scale_pos_weight=49.98` in XGBoost and `class_weight='balanced'` in Logistic Regression.

---

## Temporal Cutoff Strategy (No Leakage)

**Critical Principle:** All features use data available at observation_time.

**Example Walkthrough:**

Observation: Machine A at 2024-01-15 10:00 UTC

Features computed:

- `voltage_mean_3h`: Average voltage from 2024-01-15 07:00 to 10:00 ✓ (available)
- `error_count_24h`: Errors from 2024-01-15 10:00 - 24h to 10:00 ✓ (available)
- `days_since_last_failure`: Days since last failure <= 2024-01-15 10:00 ✓ (available)

Label (target) computed:

- `label`: Does failure occur 2024-01-15 10:00 < datetime <= 2024-01-16 10:00? ✓ (not used for feature computation)

---

## SQL Optimization

### Cardinality & Join Strategy

**Telemetry Features CTE:**

```sql
FROM telemetry t
LEFT JOIN telemetry t2 ON t.machineid = t2.machineid 
    AND t2.datetime <= t.datetime 
    AND t2.datetime > t.datetime - INTERVAL '24 hours'
GROUP BY t.machineid, t.datetime
```

- **Outer query:** One row per telemetry timestamp per machine
- **Join condition:** Self-join to look back 24 hours
- **Risk:** For dense telemetry (e.g., 1 reading/minute), this is a cross-product: O(1440 × 1440) per observation
- **Mitigation:** Indexes on `(machineid, datetime)` speed range lookups; consider data downsampling if cardinality explodes

### Feature Table Clustering

**Index:**

```sql
CREATE INDEX idx_model_input_features_machine_time 
  ON model_input_features(machineid, observation_time);
```

Enables fast retrieval during:

1. Model training (SELECT all features for a machine)
2. Batch scoring (SELECT recent features for a machine)
3. Monitoring (SELECT features for drift detection)

---

## Handling Missing Data

**NULLs can arise from:**

1. **No sensor readings in window** → `voltage_mean_3h = NULL`
   - Imputed with column mean via `SimpleImputer(strategy='mean')` during training

2. **No errors ever recorded** → `hours_since_last_error = NULL`
   - Imputed with column mean; effectively assigns the average time-since-error across machines

3. **No maintenance records** → `days_since_last_maintenance = NULL`
   - Imputed with column mean; signals machine has not been recently serviced

All imputation is fit on training data only and applied to val/test sets to prevent leakage.

---

## Observation Time Selection

The feature pipeline builds one row per telemetry reading across the full historical dataset (no date filter). This gives 876,100 training rows covering one year of operations across 100 machines, which is sufficient for stable model training.

For production retraining, the trailing window should be adjusted based on data retention policy and concept drift monitoring results.

---

## Validation Queries

After running `build_features.sql`, verify:

```sql
-- Check row count
SELECT COUNT(*) as total_rows FROM model_input_features;

-- Check class balance
SELECT label, COUNT(*) as count FROM model_input_features 
GROUP BY label;

-- Check for excessive NULLs
SELECT 
  COUNT(*) as total,
  COUNT(voltage_mean_3h) as non_null_voltage_3h,
  ROUND(100.0 * COUNT(voltage_mean_3h) / COUNT(*), 2) as pct_non_null
FROM model_input_features;

-- Check time range
SELECT 
  MIN(observation_time) as earliest,
  MAX(observation_time) as latest,
  COUNT(DISTINCT machineid) as unique_machines
FROM model_input_features;
```

---

## Model Training Summary

Feature engineering is complete. Models trained on this feature set:

| Model | PR-AUC | Recall | F1 |
| --- | --- | --- | --- |
| Logistic Regression (baseline) | 0.8285 | 0.9895 | 0.6405 |
| XGBoost | 0.9999 | 1.0000 | 0.9919 |
| Ensemble (threshold=0.729) | 0.9992 | 0.9985 | 0.9936 |

See [README.MD](../README.MD) for full results and confusion matrix breakdown.
