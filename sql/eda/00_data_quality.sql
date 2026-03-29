-- ===========================================================================
-- 00_data_quality.sql
-- Purpose:     Pre-flight validation of data quality before EDA or modelling.
-- Checks:      Row counts, nulls, duplicates, FK integrity,
--              temporal coverage, class distribution, outliers, data types.
-- Run first:   Always execute before other SQL files
-- Dataset:     MSFT Azure Predictive Maintenance (PdM)
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- 1. ROW COUNTS
-- Quick sanity check that all tables loaded as expected.
-- Expected: machines=100, telemetry~876k, errors~3.9k,
--           maintenance~3.3k, failures~761
-- ---------------------------------------------------------------------------
SELECT 'machines' AS table_name, COUNT(*) AS row_count FROM machines
UNION ALL
SELECT 'telemetry', COUNT(*) FROM telemetry
UNION ALL
SELECT 'errors', COUNT(*) FROM errors
UNION ALL
SELECT 'maintenance', COUNT(*) FROM maintenance
UNION ALL
SELECT 'failures', COUNT(*) FROM failures
ORDER BY table_name;


-- -------------------------------------------------------------
-- 2. TEMPORAL COVERAGE
-- Confirms the dataset spans the expected date range and that
-- no tables have suspiciously narrow or gapped time windows.
-- A mismatch between tables would indicate a loading issue.
-- -------------------------------------------------------------
SELECT 
    'telemetry' AS source,
    MIN(datetime) AS earliest,
    MAX(datetime) AS latest,
    COUNT(DISTINCT DATE_TRUNC('month', datetime)) AS months_covered
FROM telemetry

UNION ALL

SELECT 
    'failures',
    MIN(datetime),
    MAX(datetime),
    COUNT(DISTINCT DATE_TRUNC('month', datetime))
FROM failures 

UNION ALL 

SELECT 
    'maintenance',
    MIN(datetime),
    MAX(datetime),
    COUNT(DISTINCT DATE_TRUNC('month', datetime))
FROM maintenance

ORDER BY source;


-- -------------------------------------------------------------
-- 3. NULL CHECKS — TELEMETRY
-- Nulls in datetime or machine_id would break any time-series
-- feature engineering downstream. Sensor nulls are acceptable
-- but need to be quantified for imputation decisions.
-- -------------------------------------------------------------
SELECT
    COUNT(*)                                   AS total_rows,
    COUNT(*) FILTER (WHERE datetime  IS NULL)  AS null_datetime,
    COUNT(*) FILTER (WHERE machineID IS NULL)  AS null_machine_id,
    COUNT(*) FILTER (WHERE volt      IS NULL)  AS null_volt,
    COUNT(*) FILTER (WHERE rotate    IS NULL)  AS null_rotate,
    COUNT(*) FILTER (WHERE pressure  IS NULL)  AS null_pressure,
    COUNT(*) FILTER (WHERE vibration IS NULL)  AS null_vibration
FROM telemetry;


-- -------------------------------------------------------------
-- 4. NULL CHECKS — EVENT TABLES
-- Nulls in any column of errors, maintenance, or failures would
-- indicate a promotion issue from staging. Expect zero nulls.
-- -------------------------------------------------------------
SELECT
    'errors' AS table_name,
    COUNT(*) FILTER (WHERE datetime  IS NULL),
    COUNT(*) FILTER (WHERE errorID   IS NULL),
    COUNT(*) FILTER (WHERE machineID IS NULL)
FROM errors

UNION ALL

SELECT 
    'maintenance',
    COUNT(*) FILTER (WHERE datetime  IS NULL),
    COUNT(*) FILTER (WHERE machineID IS NULL),
    COUNT(*) FILTER (WHERE comp IS NULL)
FROM maintenance

UNION ALL

SELECT 
    'failures',
    COUNT(*) FILTER (WHERE datetime  IS NULL),
    COUNT(*) FILTER (WHERE machineID IS NULL),
    COUNT(*) FILTER (WHERE failure   IS NULL)
FROM failures;


-- -------------------------------------------------------------
-- 5. DUPLICATE CHECK — TELEMETRY
-- Each machine should have exactly one reading per timestamp.
-- Duplicates would inflate rolling averages and window features.
-- LIMIT 10 surfaces examples without scanning the full result.
-- -------------------------------------------------------------
SELECT machineID, datetime, COUNT(*) AS duplicate_count
FROM telemetry
GROUP BY machineID, datetime
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC
LIMIT 10;


-- -------------------------------------------------------------
-- 6. FOREIGN KEY INTEGRITY — ALL EVENT TABLES
-- Orphaned records (machineID not in machines) would cause
-- silent data loss when joining for feature engineering.
-- Anti-join pattern: LEFT JOIN + WHERE right side IS NULL.
-- Expect zero orphans in all tables.
-- -------------------------------------------------------------
SELECT 'telemetry' AS table_name, COUNT(*) AS orphaned_rows
FROM telemetry t
LEFT JOIN machines m ON t.machineID = m.machineID
WHERE m.machineID IS NULL

UNION ALL

SELECT 'errors', COUNT(*)
FROM errors e
LEFT JOIN machines m ON e.machineID = m.machineID
WHERE m.machineID IS NULL

UNION ALL

SELECT 'failures', COUNT(*)
FROM failures f
LEFT JOIN machines m ON f.machineID = m.machineID
WHERE m.machineID IS NULL

UNION ALL

SELECT 'maintenance', COUNT(*)
FROM maintenance mt
LEFT JOIN machines m ON mt.machineID = m.machineID
WHERE m.machineID IS NULL;


-- -------------------------------------------------------------
-- 7. MACHINE COVERAGE
-- Confirms all 100 machines have telemetry readings. Machines
-- with no telemetry cannot be scored by the model at inference.
-- -------------------------------------------------------------
SELECT
    COUNT(DISTINCT m.machineID)     AS total_machines,
    COUNT(DISTINCT t.machineID)     AS machines_with_telemetry_reading,
    COUNT(DISTINCT m.machineID) - 
        COUNT(DISTINCT t.machineID) AS machines_missing_telemetry_reading
FROM machines m
LEFT JOIN telemetry t ON m.machineID = t.machineID;


-- ------------------------------------------------------------------
-- 8. FAILURE CLASS DISTRIBUTION
-- Quantifies class imbalance across failure types. Critical
-- input to modelling decisions: SMOTE, class weights, threshold
-- tuning, and PR curve preference over ROC.
-- Window function: SUM() OVER () for percentage without a subquery.
-- ------------------------------------------------------------------
SELECT 
    failure, COUNT(*) AS occurrences,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER()
    , 2)              AS pct_of_total
FROM failures
GROUP BY failure
ORDER BY occurrences DESC;


-- -------------------------------------------------------------
-- 9. FAILURE RATE PER MACHINE
-- Identifies machines with disproportionately high failure
-- counts — potential outliers or high-risk assets.
-- Uses LEFT JOIN to include machines with zero failures.
-- -------------------------------------------------------------
SELECT 
    m.machineID, 
    m.model, 
    m.age, 
    COUNT(f.id) AS total_failures,
    ROUND(
        COUNT(f.id) * 100.0 / NULLIF(
            SUM(COUNT(f.id)) OVER (), 0
        )
    , 2)       AS pct_of_total_failures
FROM machines m
LEFT JOIN failures f ON m.machineID = f.machineID
GROUP BY m.machineID, m.model, m.age
ORDER BY total_failures DESC
LIMIT 20; 


-- -------------------------------------------------------------
-- 10. ERROR FREQUENCY BY TYPE
-- Surfaces which error codes are most prevalent. High error
-- frequency in the hours before failure is a key feature
-- candidate for the ML model.
-- -------------------------------------------------------------
SELECT 
    errorID, COUNT(*)         AS occurrences,
    COUNT(DISTINCT machineID) AS machines_affected,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()
    , 2)                      AS pct_of_total
FROM errors
GROUP BY errorID
ORDER BY occurrences DESC;