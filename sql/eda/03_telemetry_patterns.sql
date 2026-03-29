-- =============================================================
-- 03_telemetry_patterns.sql
-- Purpose:    Identify sensor anomalies that precede failures
-- Business Q: What do sensor readings look like in the hours
--             before a failure vs during normal operation?
--             Are rolling average anomalies detectable in advance?
-- Concepts:   Rolling window aggregations:
--               AVG() OVER (PARTITION BY ... ORDER BY ...
--                           ROWS BETWEEN N PRECEDING AND CURRENT ROW)
--             Multi-step CTEs to build a before/after comparison.
--             This is the most advanced SQL in the EDA set —
--             demonstrates ability to chain logic rather than
--             write one-shot queries.
-- Output:     Quantifies sensor divergence in the pre-failure window.
--             Directly justifies the 3h / 12h / 24h rolling feature
--             windows in the feature engineering plan.
-- Tables:     telemetry, failures
-- =============================================================


-- ---------------------------------------------------------------
-- BLOCK 1: Sensor summary statistics — overall baseline
-- What is the normal operating range for each sensor?
-- Establishes the baseline before looking at anomalies.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: sensor (voltage/rotation/pressure/vibration),
--          min, max, avg, stddev


-- ---------------------------------------------------------------
-- BLOCK 2: Rolling 3-hour average per machine per sensor
-- Core window function pattern: ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
-- (assuming hourly telemetry = 3 rows = 3h window)
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: machine_id, datetime, voltage, rotation, pressure, vibration,
--          voltage_3h_avg, rotation_3h_avg, pressure_3h_avg, vibration_3h_avg


-- ---------------------------------------------------------------
-- BLOCK 3: Label pre-failure vs normal windows
-- CTE 1: get rolling averages (from block 2)
-- CTE 2: join failures to flag readings within 24h before a failure
-- Final: compare avg sensor readings in pre-failure vs normal windows
-- Demonstrates: multi-step CTE chaining — the key senior-level pattern
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Structure:
--   WITH rolling AS (... rolling averages ...),
--        labelled AS (... join failures, flag is_pre_failure ...)
--   SELECT is_pre_failure,
--          AVG(voltage_3h_avg), AVG(rotation_3h_avg), ...
--   FROM labelled
--   GROUP BY is_pre_failure


-- ---------------------------------------------------------------
-- BLOCK 4: Sensor volatility (stddev) in pre-failure window
-- Does sensor variance increase before failures, even if the mean
-- doesn't shift? Tests whether std features are worth engineering.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: is_pre_failure, stddev per sensor
-- Compare stddev values between the two groups


-- ---------------------------------------------------------------
-- BLOCK 5: Machines with most anomalous pre-failure sensor readings
-- Which machines show the largest delta between normal and
-- pre-failure rolling averages? Useful for model explainability.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: machine_id, avg_voltage_delta, avg_rotation_delta, ...
-- Order by total sensor deviation DESC
