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

SELECT
    comp                                    AS component,
    COUNT(*)                                AS maintenance_count,
    COUNT(DISTINCT machineid)               AS distinct_machines_serviced
FROM maintenance
GROUP BY comp
ORDER BY maintenance_count DESC;


-- ---------------------------------------------------------------
-- BLOCK 2: Failures after maintenance — did it work?
-- For each maintenance event, check if a failure occurred within
-- 30 days afterward. Join on machineid + time window.
-- ---------------------------------------------------------------

SELECT
    m.machineid,
    m.datetime                              AS maintenance_datetime,
    m.comp                                  AS component,
    f.datetime                              AS failure_datetime,
    CASE WHEN f.machineid IS NOT NULL
         THEN TRUE ELSE FALSE
    END                                     AS failure_within_30d
FROM maintenance m
LEFT JOIN failures f
    ON  f.machineid = m.machineid
    AND f.datetime  > m.datetime
    AND f.datetime <= m.datetime + INTERVAL '30 days'
ORDER BY m.machineid, m.datetime;


-- ---------------------------------------------------------------
-- BLOCK 3: Post-maintenance failure rate by component
-- Which components, when replaced, have the lowest subsequent
-- failure rate? Builds the maintenance effectiveness scorecard.
-- ---------------------------------------------------------------

WITH maintenance_outcomes AS (
    SELECT
        m.machineid,
        m.datetime  AS maintenance_datetime,
        m.comp      AS component,
        f.datetime  AS failure_datetime
    FROM maintenance m
    LEFT JOIN failures f
        ON  f.machineid = m.machineid
        AND f.datetime  > m.datetime
        AND f.datetime <= m.datetime + INTERVAL '30 days'
)
SELECT
    component,
    COUNT(*)                                            AS total_replacements,
    COUNT(failure_datetime)                             AS failures_within_30d,
    ROUND(
        COUNT(failure_datetime)::NUMERIC / COUNT(*), 4
    )                                                   AS post_maintenance_failure_rate
FROM maintenance_outcomes
GROUP BY component
ORDER BY post_maintenance_failure_rate ASC;


-- ---------------------------------------------------------------
-- BLOCK 4: Anti-join — machines that had maintenance but NO failure
-- LEFT JOIN failures ON machineid + failure AFTER maintenance
-- WHERE failure.machineid IS NULL
-- This finds machines where maintenance genuinely appeared to work.
-- Classic anti-join pattern — interviewers test this explicitly.
-- ---------------------------------------------------------------

SELECT
    m.machineid,
    m.datetime      AS maintenance_datetime,
    m.comp          AS component
FROM maintenance m
LEFT JOIN failures f
    ON  f.machineid = m.machineid
    AND f.datetime  > m.datetime
WHERE f.machineid IS NULL
ORDER BY m.machineid, m.datetime;


-- ---------------------------------------------------------------
-- BLOCK 5: Time from maintenance to next failure
-- For machines that did eventually fail after maintenance,
-- what is the typical protection window?
-- Uses LEAD window function on ordered events per machine.
-- ---------------------------------------------------------------

WITH ordered_events AS (
    SELECT
        m.machineid,
        m.comp                                                      AS component,
        m.datetime                                                   AS maintenance_datetime,
        MIN(f.datetime)                                              AS next_failure_datetime
    FROM maintenance m
    JOIN failures f
        ON  f.machineid = m.machineid
        AND f.datetime  > m.datetime
    GROUP BY m.machineid, m.comp, m.datetime
)
SELECT
    machineid,
    component,
    maintenance_datetime,
    next_failure_datetime,
    ROUND(
        EXTRACT(EPOCH FROM (next_failure_datetime - maintenance_datetime)) / 86400.0,
        1
    )                                                               AS days_until_failure,
    ROUND(
        AVG(
            EXTRACT(EPOCH FROM (next_failure_datetime - maintenance_datetime)) / 86400.0
        ) OVER (PARTITION BY component),
        1
    )                                                               AS avg_days_until_failure
FROM ordered_events
ORDER BY component, days_until_failure;
