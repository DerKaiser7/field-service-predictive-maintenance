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
-- Output:     A prototype feature row per (machineid, observation_time)
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

-- CTE: machine_static
WITH machine_static AS (
    SELECT
        machineid,
        model,
        age,
        CASE
            WHEN age < 5  THEN 'young'
            WHEN age < 12 THEN 'mid'
            ELSE               'old'
        END AS age_category
    FROM machines
)
SELECT * FROM machine_static
ORDER BY machineid;


-- ---------------------------------------------------------------
-- BLOCK 2: Telemetry rolling features (3h window)
-- For each telemetry reading, compute rolling mean and std
-- for all four sensors using ROWS BETWEEN 2 PRECEDING AND CURRENT ROW.
-- This CTE becomes the spine of the feature table.
-- ---------------------------------------------------------------

-- CTE: telemetry_rolling
WITH telemetry_rolling AS (
    SELECT
        machineid,
        datetime                                                            AS observation_time,
        AVG(volt)      OVER w                                              AS voltage_mean_3h,
        STDDEV(volt)   OVER w                                              AS voltage_std_3h,
        AVG(rotate)    OVER w                                              AS rotation_mean_3h,
        STDDEV(rotate) OVER w                                              AS rotation_std_3h,
        AVG(pressure)  OVER w                                              AS pressure_mean_3h,
        STDDEV(pressure) OVER w                                            AS pressure_std_3h,
        AVG(vibration) OVER w                                              AS vibration_mean_3h,
        STDDEV(vibration) OVER w                                           AS vibration_std_3h
    FROM telemetry
    WINDOW w AS (
        PARTITION BY machineid
        ORDER BY datetime
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )
)
SELECT * FROM telemetry_rolling
ORDER BY machineid, observation_time
LIMIT 100;


-- ---------------------------------------------------------------
-- BLOCK 3: Error features — count in last 24h, time since last error
-- For each observation_time, count errors in the preceding 24h
-- and compute hours since the most recent error.
-- Demonstrates: correlated-style logic via CTE + time filter join
-- ---------------------------------------------------------------

-- CTE: error_features
WITH telemetry_spine AS (
    SELECT DISTINCT machineid, datetime AS observation_time
    FROM telemetry
),
error_features AS (
    SELECT
        ts.machineid,
        ts.observation_time,
        COUNT(e.id)                                                         AS error_count_24h,
        ROUND(
            EXTRACT(EPOCH FROM (
                ts.observation_time - MAX(e.datetime)
            )) / 3600.0,
            2
        )                                                                   AS hours_since_last_error
    FROM telemetry_spine ts
    LEFT JOIN errors e
        ON  e.machineid = ts.machineid
        AND e.datetime  > ts.observation_time - INTERVAL '24 hours'
        AND e.datetime <= ts.observation_time
    GROUP BY ts.machineid, ts.observation_time
)
SELECT * FROM error_features
ORDER BY machineid, observation_time
LIMIT 100;


-- ---------------------------------------------------------------
-- BLOCK 4: Maintenance features — days since last maintenance
-- For each observation, how many days since the machine last had
-- a maintenance event? Uses MAX with a time filter.
-- ---------------------------------------------------------------

-- CTE: maintenance_features
WITH telemetry_spine AS (
    SELECT DISTINCT machineid, datetime AS observation_time
    FROM telemetry
),
maintenance_features AS (
    SELECT
        ts.machineid,
        ts.observation_time,
        ROUND(
            EXTRACT(EPOCH FROM (
                ts.observation_time - MAX(m.datetime)
            )) / 86400.0,
            2
        )                                                                   AS days_since_last_maintenance
    FROM telemetry_spine ts
    LEFT JOIN maintenance m
        ON  m.machineid = ts.machineid
        AND m.datetime <= ts.observation_time
    GROUP BY ts.machineid, ts.observation_time
)
SELECT * FROM maintenance_features
ORDER BY machineid, observation_time
LIMIT 100;


-- ---------------------------------------------------------------
-- BLOCK 5: Failure history features
-- Total prior failures and days since last failure, both computed
-- strictly before observation_time to prevent label leakage.
-- ---------------------------------------------------------------

-- CTE: failure_history
WITH telemetry_spine AS (
    SELECT DISTINCT machineid, datetime AS observation_time
    FROM telemetry
),
failure_history AS (
    SELECT
        ts.machineid,
        ts.observation_time,
        COUNT(f.id)                                                         AS total_prior_failures,
        ROUND(
            EXTRACT(EPOCH FROM (
                ts.observation_time - MAX(f.datetime)
            )) / 86400.0,
            2
        )                                                                   AS days_since_last_failure
    FROM telemetry_spine ts
    LEFT JOIN failures f
        ON  f.machineid = ts.machineid
        AND f.datetime  < ts.observation_time
    GROUP BY ts.machineid, ts.observation_time
)
SELECT * FROM failure_history
ORDER BY machineid, observation_time
LIMIT 100;


-- ---------------------------------------------------------------
-- BLOCK 6: Assemble full feature row + label
-- Join all CTEs on (machineid, observation_time).
-- Label = 1 if a failure occurs in (observation_time, observation_time + 24h].
-- This is the model_input_features table prototype.
-- ---------------------------------------------------------------

WITH machine_static AS (
    SELECT
        machineid,
        model,
        age,
        CASE
            WHEN age < 5  THEN 'young'
            WHEN age < 12 THEN 'mid'
            ELSE               'old'
        END AS age_category
    FROM machines
),
telemetry_rolling AS (
    SELECT
        machineid,
        datetime                                                            AS observation_time,
        AVG(volt)         OVER w                                           AS voltage_mean_3h,
        STDDEV(volt)      OVER w                                           AS voltage_std_3h,
        AVG(rotate)       OVER w                                           AS rotation_mean_3h,
        STDDEV(rotate)    OVER w                                           AS rotation_std_3h,
        AVG(pressure)     OVER w                                           AS pressure_mean_3h,
        STDDEV(pressure)  OVER w                                           AS pressure_std_3h,
        AVG(vibration)    OVER w                                           AS vibration_mean_3h,
        STDDEV(vibration) OVER w                                           AS vibration_std_3h
    FROM telemetry
    WINDOW w AS (
        PARTITION BY machineid
        ORDER BY datetime
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )
),
error_features AS (
    SELECT
        t.machineid,
        t.observation_time,
        COUNT(e.id)                                                         AS error_count_24h,
        ROUND(
            EXTRACT(EPOCH FROM (t.observation_time - MAX(e.datetime))) / 3600.0, 2
        )                                                                   AS hours_since_last_error
    FROM telemetry_rolling t
    LEFT JOIN errors e
        ON  e.machineid = t.machineid
        AND e.datetime  > t.observation_time - INTERVAL '24 hours'
        AND e.datetime <= t.observation_time
    GROUP BY t.machineid, t.observation_time
),
maintenance_features AS (
    SELECT
        t.machineid,
        t.observation_time,
        ROUND(
            EXTRACT(EPOCH FROM (t.observation_time - MAX(m.datetime))) / 86400.0, 2
        )                                                                   AS days_since_last_maintenance
    FROM telemetry_rolling t
    LEFT JOIN maintenance m
        ON  m.machineid = t.machineid
        AND m.datetime <= t.observation_time
    GROUP BY t.machineid, t.observation_time
),
failure_history AS (
    SELECT
        t.machineid,
        t.observation_time,
        COUNT(f.id)                                                         AS total_prior_failures,
        ROUND(
            EXTRACT(EPOCH FROM (t.observation_time - MAX(f.datetime))) / 86400.0, 2
        )                                                                   AS days_since_last_failure
    FROM telemetry_rolling t
    LEFT JOIN failures f
        ON  f.machineid = t.machineid
        AND f.datetime  < t.observation_time
    GROUP BY t.machineid, t.observation_time
)
SELECT
    t.machineid,
    t.observation_time,
    ms.model,
    ms.age,
    ms.age_category,
    t.voltage_mean_3h,
    t.voltage_std_3h,
    t.rotation_mean_3h,
    t.rotation_std_3h,
    t.pressure_mean_3h,
    t.pressure_std_3h,
    t.vibration_mean_3h,
    t.vibration_std_3h,
    ef.error_count_24h,
    ef.hours_since_last_error,
    mf.days_since_last_maintenance,
    fh.total_prior_failures,
    fh.days_since_last_failure,
    CASE WHEN EXISTS (
        SELECT 1 FROM failures f
        WHERE f.machineid = t.machineid
          AND f.datetime  >  t.observation_time
          AND f.datetime  <= t.observation_time + INTERVAL '24 hours'
    ) THEN 1 ELSE 0 END                                                     AS label
FROM telemetry_rolling t
JOIN machine_static       ms ON ms.machineid = t.machineid
JOIN error_features       ef ON ef.machineid = t.machineid AND ef.observation_time = t.observation_time
JOIN maintenance_features mf ON mf.machineid = t.machineid AND mf.observation_time = t.observation_time
JOIN failure_history      fh ON fh.machineid = t.machineid AND fh.observation_time = t.observation_time
ORDER BY t.machineid, t.observation_time
LIMIT 200;
