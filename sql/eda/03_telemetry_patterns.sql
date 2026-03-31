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
-- Columns: sensor (voltage/rotation/pressure/vibration),
--          min, max, avg, stddev
-- ---------------------------------------------------------------
-- Wide Format 
SELECT
    ROUND(MIN(volt)::NUMERIC, 2) AS min_voltage,
    ROUND(MAX(volt)::NUMERIC, 2) AS max_voltage,
    ROUND(AVG(volt)::NUMERIC, 2) AS avg_voltage,
    ROUND(STDDEV(volt)::NUMERIC, 2) AS voltage_stddev,
    ROUND(MIN(rotate)::NUMERIC, 2) AS min_rotation, 
    ROUND(MAX(rotate)::NUMERIC, 2) AS max_rotation,
    ROUND(AVG(rotate)::NUMERIC, 2) AS avg_rotation,
    ROUND(STDDEV(rotate)::NUMERIC, 2) AS rotation_stddev,
    ROUND(MIN(pressure)::NUMERIC, 2) AS min_pressure,
    ROUND(MAX(pressure)::NUMERIC, 2) AS max_pressure,
    ROUND(AVG(pressure)::NUMERIC, 2) AS avg_pressure,
    ROUND(STDDEV(pressure)::NUMERIC, 2) AS pressure_stddev,
    ROUND(MIN(vibration)::NUMERIC, 2) AS min_vibration,
    ROUND(MAX(vibration)::NUMERIC, 2) AS max_vibration,
    ROUND(AVG(vibration)::NUMERIC, 2) AS avg_vibration,
    ROUND(STDDEV(vibration)::NUMERIC, 2) AS vibration_stddev
FROM telemetry;

-- Long Format
SELECT 'voltage' AS sensor, ROUND(MIN(volt)::NUMERIC, 2) AS min, ROUND(MAX(volt)::NUMERIC, 2) AS max, ROUND(AVG(volt)::NUMERIC, 2) AS avg, ROUND(STDDEV(volt)::NUMERIC, 2) AS stddev FROM telemetry
UNION ALL
SELECT 'rotation', ROUND(MIN(rotate)::NUMERIC, 2), ROUND(MAX(rotate)::NUMERIC, 2), ROUND(AVG(rotate)::NUMERIC, 2), ROUND(STDDEV(rotate)::NUMERIC, 2) FROM telemetry
UNION ALL
SELECT 'pressure', ROUND(MIN(pressure)::NUMERIC, 2), ROUND(MAX(pressure)::NUMERIC, 2), ROUND(AVG(pressure)::NUMERIC, 2), ROUND(STDDEV(pressure)::NUMERIC, 2) FROM telemetry
UNION ALL
SELECT 'vibration', ROUND(MIN(vibration)::NUMERIC, 2), ROUND(MAX(vibration)::NUMERIC, 2), ROUND(AVG(vibration)::NUMERIC, 2), ROUND(STDDEV(vibration)::NUMERIC, 2) FROM telemetry


-- ---------------------------------------------------------------
-- BLOCK 2: Rolling 3-hour average per machine per sensor
-- Core window function pattern: ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
-- (assuming hourly telemetry = 3 rows = 3h window)
-- Columns: machine_id, datetime, voltage, rotation, pressure, vibration,
--          voltage_3h_avg, rotation_3h_avg, pressure_3h_avg, vibration_3h_avg
-- ---------------------------------------------------------------
SELECT 
    machineid,
    datetime,
    volt,
    rotate,
    pressure,
    vibration,
    ROUND(AVG(volt) OVER (PARTITION BY machineid ORDER BY datetime
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::NUMERIC, 2) AS volt_3h_avg,
    ROUND(AVG(rotate) OVER (PARTITION BY machineid ORDER BY datetime
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::NUMERIC, 2) AS rotate_3h_avg,
    ROUND(AVG(pressure) OVER (PARTITION BY machineid ORDER BY datetime
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::NUMERIC, 2) AS pressure_3h_avg,
    ROUND(AVG(vibration) OVER (PARTITION BY machineid ORDER BY datetime
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::NUMERIC, 2) AS vibration_3h_avg
FROM telemetry
ORDER BY machineid, datetime;


-- ---------------------------------------------------------------
-- BLOCK 3: Label pre-failure vs normal windows
-- CTE 1: get rolling averages (from block 2)
-- CTE 2: join failures to flag readings within 24h before a failure
-- Final: compare avg sensor readings in pre-failure vs normal windows
-- Demonstrates: multi-step CTE chaining — the key senior-level pattern
-- ---------------------------------------------------------------
WITH rolling AS (
    SELECT 
        machineid,
        datetime,
        ROUND(AVG(volt) OVER (PARTITION BY machineid ORDER BY datetime
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::NUMERIC, 2) AS volt_3h_avg,
        ROUND(AVG(rotate) OVER (PARTITION BY machineid ORDER BY datetime
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::NUMERIC, 2) AS rotate_3h_avg,
        ROUND(AVG(pressure) OVER (PARTITION BY machineid ORDER BY datetime
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::NUMERIC, 2) AS pressure_3h_avg,
        ROUND(AVG(vibration) OVER (PARTITION BY machineid ORDER BY datetime
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::NUMERIC, 2) AS vibration_3h_avg
    FROM telemetry
), labelled AS (
    SELECT 
        r.*,
        EXISTS (
            SELECT 1 FROM failures f WHERE f.machineid = r.machineid
            AND f.datetime > r.datetime
            AND f.datetime <= r.datetime + INTERVAL '24 hours'
        ) AS is_pre_failure
    FROM rolling r 
)
SELECT 
    is_pre_failure,
    ROUND(AVG(volt_3h_avg)::NUMERIC, 2) AS avg_volt_3h,
    ROUND(AVG(rotate_3h_avg)::NUMERIC, 2) AS avg_rotate_3h,
    ROUND(AVG(pressure_3h_avg)::NUMERIC, 2) AS avg_pressure_3h,
    ROUND(AVG(vibration_3h_avg)::NUMERIC, 2) AS avg_vibration_3h
FROM labelled
GROUP BY is_pre_failure
ORDER BY is_pre_failure;
    


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
