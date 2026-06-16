-- =============================================================
-- build_features.sql
-- Purpose:    Production feature engineering — populates the
--             model_input_features table used for training and
--             scoring.
-- Concepts:   Multi-step CTEs, rolling window aggregations,
--             time-bounded joins, leakage-safe label engineering.
-- Output:     One row per (machine_id, observation_time) with all
--             engineered features + binary failure label.
-- Note:       Features strictly use ONLY data available at or 
--             before observation_time. NO FUTURE LEAKAGE.
-- =============================================================

-- Clear existing data for reproducibility
DELETE FROM model_input_features;

-- ---------------------------------------------------------------
-- CTE 1: Static machine features (model, age, age_category)
-- ---------------------------------------------------------------
WITH static_features AS (
    SELECT
        machineid,
        model,
        age,
        CASE
            WHEN age < 3 THEN 'new'
            WHEN age < 7 THEN 'mid_life'
            ELSE 'aged'
        END AS age_category
    FROM machines
),

-- ---------------------------------------------------------------
-- CTE 2: Telemetry rolling features (3h, 12h, 24h windows)
-- ---------------------------------------------------------------
-- For each machine at each observation time, compute rolling
-- aggregations of sensor readings from specified lookback windows
telemetry_features AS (
    SELECT
        t.machineid,
        t.datetime AS observation_time,
        
        -- 3h window (3 hours before observation_time)
        AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.volt END) AS voltage_mean_3h,
        STDDEV_POP(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.volt END) AS voltage_std_3h,
        MIN(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.volt END) AS voltage_min_3h,
        MAX(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.volt END) AS voltage_max_3h,
        
        AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.rotate END) AS rotation_mean_3h,
        STDDEV_POP(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.rotate END) AS rotation_std_3h,
        MIN(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.rotate END) AS rotation_min_3h,
        MAX(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.rotate END) AS rotation_max_3h,
        
        AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.pressure END) AS pressure_mean_3h,
        STDDEV_POP(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.pressure END) AS pressure_std_3h,
        MIN(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.pressure END) AS pressure_min_3h,
        MAX(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.pressure END) AS pressure_max_3h,
        
        AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.vibration END) AS vibration_mean_3h,
        STDDEV_POP(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.vibration END) AS vibration_std_3h,
        MIN(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.vibration END) AS vibration_min_3h,
        MAX(CASE WHEN t2.datetime > t.datetime - INTERVAL '3 hours' AND t2.datetime <= t.datetime THEN t2.vibration END) AS vibration_max_3h,
        
        -- 12h window
        AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '12 hours' AND t2.datetime <= t.datetime THEN t2.volt END) AS voltage_mean_12h,
        STDDEV_POP(CASE WHEN t2.datetime > t.datetime - INTERVAL '12 hours' AND t2.datetime <= t.datetime THEN t2.volt END) AS voltage_std_12h,
        AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '12 hours' AND t2.datetime <= t.datetime THEN t2.rotate END) AS rotation_mean_12h,
        STDDEV_POP(CASE WHEN t2.datetime > t.datetime - INTERVAL '12 hours' AND t2.datetime <= t.datetime THEN t2.rotate END) AS rotation_std_12h,
        AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '12 hours' AND t2.datetime <= t.datetime THEN t2.pressure END) AS pressure_mean_12h,
        STDDEV_POP(CASE WHEN t2.datetime > t.datetime - INTERVAL '12 hours' AND t2.datetime <= t.datetime THEN t2.pressure END) AS pressure_std_12h,
        AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '12 hours' AND t2.datetime <= t.datetime THEN t2.vibration END) AS vibration_mean_12h,
        STDDEV_POP(CASE WHEN t2.datetime > t.datetime - INTERVAL '12 hours' AND t2.datetime <= t.datetime THEN t2.vibration END) AS vibration_std_12h,
        
        -- 24h window
        AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '24 hours' AND t2.datetime <= t.datetime THEN t2.volt END) AS voltage_mean_24h,
        STDDEV_POP(CASE WHEN t2.datetime > t.datetime - INTERVAL '24 hours' AND t2.datetime <= t.datetime THEN t2.volt END) AS voltage_std_24h,
        AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '24 hours' AND t2.datetime <= t.datetime THEN t2.rotate END) AS rotation_mean_24h,
        STDDEV_POP(CASE WHEN t2.datetime > t.datetime - INTERVAL '24 hours' AND t2.datetime <= t.datetime THEN t2.rotate END) AS rotation_std_24h,
        AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '24 hours' AND t2.datetime <= t.datetime THEN t2.pressure END) AS pressure_mean_24h,
        STDDEV_POP(CASE WHEN t2.datetime > t.datetime - INTERVAL '24 hours' AND t2.datetime <= t.datetime THEN t2.pressure END) AS pressure_std_24h,
        AVG(CASE WHEN t2.datetime > t.datetime - INTERVAL '24 hours' AND t2.datetime <= t.datetime THEN t2.vibration END) AS vibration_mean_24h,
        STDDEV_POP(CASE WHEN t2.datetime > t.datetime - INTERVAL '24 hours' AND t2.datetime <= t.datetime THEN t2.vibration END) AS vibration_std_24h
        
    FROM telemetry t
    LEFT JOIN telemetry t2 ON t.machineid = t2.machineid AND t2.datetime <= t.datetime AND t2.datetime > t.datetime - INTERVAL '24 hours'
    GROUP BY t.machineid, t.datetime
),

-- ---------------------------------------------------------------
-- CTE 3: Error features (count, recency, diversity)
-- ---------------------------------------------------------------
error_features AS (
    SELECT
        t.machineid,
        t.datetime AS observation_time,
        COUNT(CASE WHEN e.datetime > t.datetime - INTERVAL '24 hours' AND e.datetime <= t.datetime THEN 1 END) AS error_count_24h,
        EXTRACT(EPOCH FROM (t.datetime - MAX(CASE WHEN e.datetime <= t.datetime THEN e.datetime END))) / 3600.0 AS hours_since_last_error,
        COUNT(DISTINCT CASE WHEN e.datetime <= t.datetime THEN e.errorID END) AS distinct_error_types
    FROM telemetry t
    LEFT JOIN errors e ON t.machineid = e.machineid AND e.datetime <= t.datetime
    GROUP BY t.machineid, t.datetime
),

-- ---------------------------------------------------------------
-- CTE 4: Maintenance features (recency, diversity)
-- ---------------------------------------------------------------
maintenance_features AS (
    SELECT
        t.machineid,
        t.datetime AS observation_time,
        EXTRACT(EPOCH FROM (t.datetime - MAX(CASE WHEN m.datetime <= t.datetime THEN m.datetime END))) / 86400.0 AS days_since_last_maintenance,
        COUNT(DISTINCT CASE WHEN m.datetime <= t.datetime THEN m.comp END) AS component_diversity
    FROM telemetry t
    LEFT JOIN maintenance m ON t.machineid = m.machineid AND m.datetime <= t.datetime
    GROUP BY t.machineid, t.datetime
),

-- ---------------------------------------------------------------
-- CTE 5: Failure history features (count, recency, diversity)
-- ---------------------------------------------------------------
failure_history AS (
    SELECT
        t.machineid,
        t.datetime AS observation_time,
        COUNT(CASE WHEN f.datetime <= t.datetime THEN 1 END) AS total_prior_failures,
        EXTRACT(EPOCH FROM (t.datetime - MAX(CASE WHEN f.datetime <= t.datetime THEN f.datetime END))) / 86400.0 AS days_since_last_failure,
        COUNT(DISTINCT CASE WHEN f.datetime <= t.datetime THEN f.failure END) AS distinct_failure_types
    FROM telemetry t
    LEFT JOIN failures f ON t.machineid = f.machineid
    GROUP BY t.machineid, t.datetime
)

-- ---------------------------------------------------------------
-- FINAL: Assemble all features + label (failure in next 24h?)
-- ---------------------------------------------------------------
INSERT INTO model_input_features (
    machineid,
    observation_time,
    -- static
    model,
    age,
    age_category,
    -- telemetry: 3h, 12h, 24h
    voltage_mean_3h, voltage_std_3h, voltage_min_3h, voltage_max_3h,
    rotation_mean_3h, rotation_std_3h, rotation_min_3h, rotation_max_3h,
    pressure_mean_3h, pressure_std_3h, pressure_min_3h, pressure_max_3h,
    vibration_mean_3h, vibration_std_3h, vibration_min_3h, vibration_max_3h,
    voltage_mean_12h, voltage_std_12h,
    rotation_mean_12h, rotation_std_12h,
    pressure_mean_12h, pressure_std_12h,
    vibration_mean_12h, vibration_std_12h,
    voltage_mean_24h, voltage_std_24h,
    rotation_mean_24h, rotation_std_24h,
    pressure_mean_24h, pressure_std_24h,
    vibration_mean_24h, vibration_std_24h,
    -- errors
    error_count_24h,
    hours_since_last_error,
    distinct_error_types,
    -- maintenance
    days_since_last_maintenance,
    component_diversity,
    -- failure history
    total_prior_failures,
    days_since_last_failure,
    distinct_failure_types,
    -- label
    label
)
SELECT
    t.machineid,
    t.observation_time,
    -- static
    s.model,
    s.age,
    s.age_category,
    -- telemetry
    t.voltage_mean_3h, t.voltage_std_3h, t.voltage_min_3h, t.voltage_max_3h,
    t.rotation_mean_3h, t.rotation_std_3h, t.rotation_min_3h, t.rotation_max_3h,
    t.pressure_mean_3h, t.pressure_std_3h, t.pressure_min_3h, t.pressure_max_3h,
    t.vibration_mean_3h, t.vibration_std_3h, t.vibration_min_3h, t.vibration_max_3h,
    t.voltage_mean_12h, t.voltage_std_12h,
    t.rotation_mean_12h, t.rotation_std_12h,
    t.pressure_mean_12h, t.pressure_std_12h,
    t.vibration_mean_12h, t.vibration_std_12h,
    t.voltage_mean_24h, t.voltage_std_24h,
    t.rotation_mean_24h, t.rotation_std_24h,
    t.pressure_mean_24h, t.pressure_std_24h,
    t.vibration_mean_24h, t.vibration_std_24h,
    -- errors
    e.error_count_24h,
    e.hours_since_last_error,
    e.distinct_error_types,
    -- maintenance
    m.days_since_last_maintenance,
    m.component_diversity,
    -- failure history
    f.total_prior_failures,
    f.days_since_last_failure,
    f.distinct_failure_types,
    -- LABEL: Does a failure occur in the 24 hours AFTER observation_time?
    -- Strictly uses data available at observation_time (no leakage)
    CASE WHEN EXISTS (
        SELECT 1 FROM failures fut
        WHERE fut.machineid = t.machineid
          AND fut.datetime > t.observation_time
          AND fut.datetime <= t.observation_time + INTERVAL '24 hours'
    ) THEN 1 ELSE 0 END AS label

FROM telemetry_features t
LEFT JOIN static_features s ON t.machineid = s.machineid
LEFT JOIN error_features e ON t.machineid = e.machineid AND t.observation_time = e.observation_time
LEFT JOIN maintenance_features m ON t.machineid = m.machineid AND t.observation_time = m.observation_time
LEFT JOIN failure_history f ON t.machineid = f.machineid AND t.observation_time = f.observation_time
ORDER BY t.machineid, t.observation_time;
