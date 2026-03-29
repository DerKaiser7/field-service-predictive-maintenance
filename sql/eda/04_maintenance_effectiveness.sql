-- =============================================================
-- 04_maintenance_effectiveness.sql
-- Purpose:    Assess whether maintenance actually prevents failures
-- Business Q: Which maintenance activities reduce failure rates?
--             Do machines that get component X replaced still fail
--             at the same rate as those that don't?
--             How long does maintenance protection last?
-- Concepts:   Multi-table joins across 4 tables,
--             LEFT JOIN / IS NULL as anti-join pattern
--             ("find machines that had maintenance but no subsequent
--             failure" — the "did it work?" query),
--             time window joins (failure within N days of maintenance)
-- Note:       Anti-joins are a specific senior-level pattern that
--             interviewers use to separate mid from senior candidates.
-- Output:     Identifies which components are worth prioritising in
--             a proactive maintenance schedule. Justifies including
--             days_since_last_maintenance and component features.
-- Tables:     machines, maintenance, failures
-- =============================================================


-- ---------------------------------------------------------------
-- BLOCK 1: Maintenance frequency by component
-- Which components get replaced most often across the fleet?
-- Baseline for understanding maintenance activity volume.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: component, maintenance_count, distinct_machines_serviced


-- ---------------------------------------------------------------
-- BLOCK 2: Failures after maintenance — did it work?
-- For each maintenance event, check if a failure occurred within
-- 30 days afterward. Join on machine_id + time window.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: machine_id, maintenance_datetime, component,
--          failure_within_30d (boolean or failure_datetime)


-- ---------------------------------------------------------------
-- BLOCK 3: Post-maintenance failure rate by component
-- Which components, when replaced, have the lowest subsequent
-- failure rate? Builds the maintenance effectiveness scorecard.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: component, total_replacements, failures_within_30d,
--          post_maintenance_failure_rate


-- ---------------------------------------------------------------
-- BLOCK 4: Anti-join — machines that had maintenance but NO failure
-- LEFT JOIN failures ON machine_id + failure AFTER maintenance
-- WHERE failure.machine_id IS NULL
-- This finds machines where maintenance genuinely appeared to work.
-- Classic anti-join pattern — interviewers test this explicitly.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: machine_id, maintenance_datetime, component
-- Filter: no failure recorded after the maintenance event


-- ---------------------------------------------------------------
-- BLOCK 5: Time from maintenance to next failure
-- For machines that did eventually fail after maintenance,
-- what is the typical protection window?
-- Uses LAG/LEAD or a self-join on ordered events.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: machine_id, component, maintenance_datetime,
--          next_failure_datetime, days_until_failure
-- Aggregate: avg_days_until_failure per component
