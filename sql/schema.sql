-- Field Service Predictive Maintenance Schema
-- MSFT Azure Predictive Maintenance dataset structure
-- Bootstrapped by Docker on Postgres init

-- =========================================================
-- Optional cleanup for fresh local rebuilds
-- =========================================================
DROP TABLE IF EXISTS prediction_logs CASCADE;
DROP TABLE IF EXISTS failures CASCADE;
DROP TABLE IF EXISTS maintenance CASCADE;
DROP TABLE IF EXISTS errors CASCADE;
DROP TABLE IF EXISTS telemetry CASCADE;
DROP TABLE IF EXISTS machines CASCADE;

DROP TABLE IF EXISTS staging_failures CASCADE;
DROP TABLE IF EXISTS staging_maintenance CASCADE;
DROP TABLE IF EXISTS staging_errors CASCADE;
DROP TABLE IF EXISTS staging_telemetry CASCADE;
DROP TABLE IF EXISTS staging_machines CASCADE;

-- ===========================================================
-- STAGING TABLES
-- Raw landing zone loaded from CSVs with minimal constraints
-- ===========================================================

CREATE TABLE staging_machines(
    machineid TEXT,
    model TEXT,
    age TEXT
);

CREATE TABLE staging_telemetry (
    datetime        TEXT,
    machineid      TEXT,
    volt            TEXT,
    rotate          TEXT,
    pressure        TEXT,
    vibration       TEXT
);

CREATE TABLE staging_errors (
    datetime        TEXT,
    machineid      TEXT,
    errorID        TEXT
);

CREATE TABLE staging_maintenance (
    datetime        TEXT,
    machineid      TEXT,
    comp            TEXT
);

CREATE TABLE staging_failures (
    datetime        TEXT,
    machineid      TEXT,
    failure         TEXT
);

-- =====================================
-- BASE TABLES
-- Cleaned data promoted from staging
-- =====================================

CREATE TABLE machines (
    machineid TEXT PRIMARY KEY,
    model VARCHAR(20) NOT NULL,
    age INTEGER NOT NULL
);

CREATE TABLE telemetry (
    id BIGSERIAL PRIMARY KEY,
    datetime TIMESTAMP NOT NULL,
    machineid TEXT REFERENCES machines(machineid),
    volt NUMERIC(8,4),
    rotate NUMERIC(8,4),
    pressure NUMERIC(8,4),
    vibration NUMERIC(8,4)
);

CREATE TABLE errors (
    id BIGSERIAL PRIMARY KEY,
    datetime TIMESTAMP NOT NULL,
    machineid TEXT REFERENCES machines(machineid),
    errorID VARCHAR(20) NOT NULL
);

CREATE TABLE maintenance (
    id BIGSERIAL PRIMARY KEY,
    datetime TIMESTAMP NOT NULL,
    machineid TEXT REFERENCES machines(machineid),
    comp VARCHAR(20) NOT NULL
);

CREATE TABLE failures (
    id BIGSERIAL PRIMARY KEY,
    datetime TIMESTAMP NOT NULL,
    machineid TEXT REFERENCES machines(machineid),
    failure VARCHAR(20) NOT NULL
);

CREATE TABLE prediction_logs (
    prediction_id BIGSERIAL PRIMARY KEY,
    predicted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    machineid TEXT NOT NULL REFERENCES machines(machineid),
    observation_time TIMESTAMP NOT NULL,
    failure_probability NUMERIC(8,6) NOT NULL CHECK (failure_probability >= 0 AND failure_probability <= 1),
    predicted_label INTEGER NOT NULL CHECK (predicted_label IN (0,1)),
    model_version   VARCHAR
);

-- ===================================
-- Indexes for query performance
-- ===================================

CREATE INDEX IF NOT EXISTS idx_telemetry_machine_dt ON telemetry(machine_id, datetime);
CREATE INDEX IF NOT EXISTS idx_errors_machine_dt ON errors(machine_id, datetime);
CREATE INDEX IF NOT EXISTS idx_failures_machine_dt ON failures(machine_id, datetime);
CREATE INDEX IF NOT EXISTS idx_maint_machine_dt ON maintenance(machine_id, datetime);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_machine_obs_dt ON prediction_logs(machine_id, observation_time);