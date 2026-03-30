-- Promote data from staging tables into typed base tables

-- clear base tables for repeatable local reloads
TRUNCATE TABLE prediction_logs RESTART IDENTITY;
TRUNCATE TABLE failures RESTART IDENTITY CASCADE;
TRUNCATE TABLE maintenance RESTART IDENTITY CASCADE;
TRUNCATE TABLE errors RESTART IDENTITY CASCADE;
TRUNCATE TABLE telemetry RESTART IDENTITY CASCADE;
TRUNCATE TABLE machines RESTART IDENTITY CASCADE;

-- =========================
-- machines
-- =========================
INSERT INTO machines (machineid, model, age)
SELECT DISTINCT
    TRIM(machineid),
    TRIM(model),
    TRIM(age)::INTEGER
FROM staging_machines
WHERE machineid IS NOT NULL
  AND model IS NOT NULL
  AND age IS NOT NULL;

-- =========================
-- telemetry
-- =========================
INSERT INTO telemetry (datetime, machineid, volt, rotate, pressure, vibration)
SELECT
    TRIM(datetime)::TIMESTAMP,
    TRIM(machineid),
    NULLIF(TRIM(volt), '')::NUMERIC(10,4),
    NULLIF(TRIM(rotate), '')::NUMERIC(10,4),
    NULLIF(TRIM(pressure), '')::NUMERIC(10,4),
    NULLIF(TRIM(vibration), '')::NUMERIC(10,4)
FROM staging_telemetry
WHERE datetime IS NOT NULL
  AND machineid IS NOT NULL;

-- =========================
-- errors
-- =========================
INSERT INTO errors (datetime, machineid, errorID)
SELECT
    TRIM(datetime)::TIMESTAMP,
    TRIM(machineid),
    TRIM(errorID)
FROM staging_errors
WHERE datetime IS NOT NULL
  AND machineid IS NOT NULL
  AND errorID IS NOT NULL;

-- =========================
-- maintenance
-- =========================
INSERT INTO maintenance (datetime, machineid, comp)
SELECT
    TRIM(datetime)::TIMESTAMP,
    TRIM(machineid),
    TRIM(comp)
FROM staging_maintenance
WHERE datetime IS NOT NULL
  AND machineid IS NOT NULL
  AND comp IS NOT NULL;

-- =========================
-- failures
-- =========================
INSERT INTO failures (datetime, machineid, failure)
SELECT
    TRIM(datetime)::TIMESTAMP,
    TRIM(machineid),
    TRIM(failure)
FROM staging_failures
WHERE datetime IS NOT NULL
  AND machineid IS NOT NULL
  AND failure IS NOT NULL;