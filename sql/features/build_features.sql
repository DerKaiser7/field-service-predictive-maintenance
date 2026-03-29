-- =============================================================
-- build_features.sql
-- Purpose:    Production feature engineering — populates the
--             model_input_features table used for training and
--             scoring. This is the finalised version of the logic
--             prototyped in sql/eda/05_feature_candidates.sql.
-- Business Q: N/A — this is a pipeline query, not an analytical one.
-- Concepts:   Multi-step CTEs, rolling window aggregations,
--             time-bounded joins, leakage-safe label engineering.
-- Output:     One row per (machine_id, observation_time) with all
--             engineered features + binary failure label.
--             Inserted into model_input_features table.
-- Depends on: machines, telemetry, errors, maintenance, failures
-- Run after:  schema.sql, staging/ loads
-- Note:       Features must use ONLY data available at or before
--             observation_time. No future leakage.
-- =============================================================


-- ---------------------------------------------------------------
-- CTE 1: Static machine features
-- ---------------------------------------------------------------

-- TODO: Copy finalised logic from 05_feature_candidates.sql BLOCK 1


-- ---------------------------------------------------------------
-- CTE 2: Telemetry rolling features (3h, 12h, 24h windows)
-- ---------------------------------------------------------------

-- TODO: Copy finalised logic from 05_feature_candidates.sql BLOCK 2
--       Extend to 12h and 24h windows once 3h is validated


-- ---------------------------------------------------------------
-- CTE 3: Error features
-- ---------------------------------------------------------------

-- TODO: Copy finalised logic from 05_feature_candidates.sql BLOCK 3


-- ---------------------------------------------------------------
-- CTE 4: Maintenance features
-- ---------------------------------------------------------------

-- TODO: Copy finalised logic from 05_feature_candidates.sql BLOCK 4


-- ---------------------------------------------------------------
-- CTE 5: Failure history features
-- ---------------------------------------------------------------

-- TODO: Copy finalised logic from 05_feature_candidates.sql BLOCK 5


-- ---------------------------------------------------------------
-- FINAL: Assemble + label + insert into model_input_features
-- ---------------------------------------------------------------

INSERT INTO model_input_features (
    machine_id,
    observation_time,
    -- static
    model,
    age,
    age_category,
    -- telemetry
    voltage_mean_3h,
    voltage_std_3h,
    rotation_mean_3h,
    rotation_std_3h,
    pressure_mean_3h,
    pressure_std_3h,
    vibration_mean_3h,
    vibration_std_3h,
    -- errors
    error_count_24h,
    hours_since_last_error,
    -- maintenance
    days_since_last_maintenance,
    -- failure history
    total_prior_failures,
    days_since_last_failure,
    -- label
    label
)

-- TODO: Final SELECT joining all CTEs + label column
-- Label:
--   CASE WHEN EXISTS (
--       SELECT 1 FROM failures f
--       WHERE f.machine_id = t.machine_id
--         AND f.datetime > t.observation_time
--         AND f.datetime <= t.observation_time + INTERVAL '24 hours'
--   ) THEN 1 ELSE 0 END AS label
;
