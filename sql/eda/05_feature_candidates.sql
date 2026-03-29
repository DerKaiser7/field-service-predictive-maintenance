-- =============================================================
-- 05_feature_candidates.sql
-- Purpose:    Prototype the ML feature set in SQL before Python
-- Business Q: What signals should we engineer for the model?
--             Hours since last failure, error count in last 24h,
--             rolling sensor averages — do they show predictive
--             signal when inspected at the row level?
-- Concepts:   Multi-step CTEs building a feature table row by row,
--             combining window functions with joins,
--             time-bounded aggregation via WHERE + interval filter.
--             This is the most architecturally complex query in the
--             EDA set — it is the direct SQL blueprint for the
--             Python feature engineering pipeline. The SQL and the
--             pandas code should tell the same story.
-- Output:     A prototype feature row per (machine_id, observation_time)
--             that maps 1:1 to model_input_features table columns.
--             Validates feature logic before committing to Python.
-- Tables:     machines, telemetry, errors, maintenance, failures
-- Note:       No future leakage — all features must use only data
--             available at or before observation_time.
-- =============================================================


-- ---------------------------------------------------------------
-- BLOCK 1: Static features — machine age and model
-- The simplest feature block. Establishes the base CTE that all
-- subsequent CTEs join onto.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- CTE: machine_static
-- Columns: machine_id, model, age, age_category (CASE WHEN)


-- ---------------------------------------------------------------
-- BLOCK 2: Telemetry rolling features (3h window)
-- For each telemetry reading, compute rolling mean and std
-- for all four sensors using ROWS BETWEEN 2 PRECEDING AND CURRENT ROW.
-- This CTE becomes the spine of the feature table.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- CTE: telemetry_rolling
-- Columns: machine_id, datetime as observation_time,
--          voltage_mean_3h, voltage_std_3h,
--          rotation_mean_3h, rotation_std_3h,
--          pressure_mean_3h, pressure_std_3h,
--          vibration_mean_3h, vibration_std_3h


-- ---------------------------------------------------------------
-- BLOCK 3: Error features — count in last 24h, time since last error
-- For each observation_time, count errors in the preceding 24h
-- and compute hours since the most recent error.
-- Demonstrates: correlated-style logic via CTE + time filter join
-- ---------------------------------------------------------------

-- TODO: Write query here
-- CTE: error_features
-- Columns: machine_id, observation_time,
--          error_count_24h, hours_since_last_error


-- ---------------------------------------------------------------
-- BLOCK 4: Maintenance features — days since last maintenance
-- For each observation, how many days since the machine last had
-- a maintenance event? Uses MAX with a time filter.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- CTE: maintenance_features
-- Columns: machine_id, observation_time, days_since_last_maintenance


-- ---------------------------------------------------------------
-- BLOCK 5: Failure history features
-- Total prior failures and days since last failure, both computed
-- strictly before observation_time to prevent label leakage.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- CTE: failure_history
-- Columns: machine_id, observation_time,
--          total_prior_failures, days_since_last_failure


-- ---------------------------------------------------------------
-- BLOCK 6: Assemble full feature row + label
-- Join all CTEs on (machine_id, observation_time).
-- Label = 1 if a failure occurs in (observation_time, observation_time + 24h].
-- This is the model_input_features table prototype.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Final SELECT joining all CTEs
-- Add label column:
--   CASE WHEN EXISTS (
--       SELECT 1 FROM failures f
--       WHERE f.machine_id = t.machine_id
--         AND f.datetime > t.observation_time
--         AND f.datetime <= t.observation_time + INTERVAL '24 hours'
--   ) THEN 1 ELSE 0 END AS label
