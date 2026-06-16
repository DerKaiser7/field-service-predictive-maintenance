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
-- FEATURE TABLE
-- Engineered features for model training
-- ===================================

CREATE TABLE model_input_features (
    feature_id BIGSERIAL PRIMARY KEY,
    machineid TEXT NOT NULL REFERENCES machines(machineid),
    observation_time TIMESTAMP NOT NULL,
    
    -- Static features
    model VARCHAR(20),
    age INTEGER,
    age_category VARCHAR(20),
    
    -- Telemetry: 3h windows
    voltage_mean_3h NUMERIC(8,4),
    voltage_std_3h NUMERIC(8,4),
    voltage_min_3h NUMERIC(8,4),
    voltage_max_3h NUMERIC(8,4),
    rotation_mean_3h NUMERIC(8,4),
    rotation_std_3h NUMERIC(8,4),
    rotation_min_3h NUMERIC(8,4),
    rotation_max_3h NUMERIC(8,4),
    pressure_mean_3h NUMERIC(8,4),
    pressure_std_3h NUMERIC(8,4),
    pressure_min_3h NUMERIC(8,4),
    pressure_max_3h NUMERIC(8,4),
    vibration_mean_3h NUMERIC(8,4),
    vibration_std_3h NUMERIC(8,4),
    vibration_min_3h NUMERIC(8,4),
    vibration_max_3h NUMERIC(8,4),
    
    -- Telemetry: 12h windows
    voltage_mean_12h NUMERIC(8,4),
    voltage_std_12h NUMERIC(8,4),
    rotation_mean_12h NUMERIC(8,4),
    rotation_std_12h NUMERIC(8,4),
    pressure_mean_12h NUMERIC(8,4),
    pressure_std_12h NUMERIC(8,4),
    vibration_mean_12h NUMERIC(8,4),
    vibration_std_12h NUMERIC(8,4),
    
    -- Telemetry: 24h windows
    voltage_mean_24h NUMERIC(8,4),
    voltage_std_24h NUMERIC(8,4),
    rotation_mean_24h NUMERIC(8,4),
    rotation_std_24h NUMERIC(8,4),
    pressure_mean_24h NUMERIC(8,4),
    pressure_std_24h NUMERIC(8,4),
    vibration_mean_24h NUMERIC(8,4),
    vibration_std_24h NUMERIC(8,4),
    
    -- Error features
    error_count_24h INTEGER,
    hours_since_last_error NUMERIC(10,2),
    distinct_error_types INTEGER,
    
    -- Maintenance features
    days_since_last_maintenance NUMERIC(10,2),
    component_diversity INTEGER,
    
    -- Failure history features
    total_prior_failures INTEGER,
    days_since_last_failure NUMERIC(10,2),
    distinct_failure_types INTEGER,
    
    -- Label (target)
    label INTEGER CHECK (label IN (0, 1)),
    
    UNIQUE(machineid, observation_time)
);

-- ===================================
-- MONITORING TABLE
-- MLOps: Track data/prediction drift alerts
-- ===================================

CREATE TABLE model_monitoring (
    monitor_id BIGSERIAL PRIMARY KEY,
    checked_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    feature_name VARCHAR,
    drift_metric NUMERIC(6,4),
    alert_triggered BOOLEAN DEFAULT FALSE,
    alert_message TEXT
);

-- ===================================
-- Indexes for query performance
-- ===================================

CREATE INDEX IF NOT EXISTS idx_telemetry_machine_dt ON telemetry(machineid, datetime);
CREATE INDEX IF NOT EXISTS idx_errors_machine_dt ON errors(machineid, datetime);
CREATE INDEX IF NOT EXISTS idx_failures_machine_dt ON failures(machineid, datetime);
CREATE INDEX IF NOT EXISTS idx_maint_machine_dt ON maintenance(machineid, datetime);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_machine_obs_dt ON prediction_logs(machineid, observation_time);
CREATE INDEX IF NOT EXISTS idx_model_input_features_machine_time ON model_input_features(machineid, observation_time);