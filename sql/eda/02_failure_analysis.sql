-- =============================================================
-- 02_failure_analysis.sql
-- Purpose:    Characterise failure behaviour over time
-- Business Q: What is the failure rate per machine over time?
--             Which machines are repeat offenders?
--             What is the typical time between failures (MTBF)?
-- Concepts:   ROW_NUMBER(), LAG(), LEAD(), PARTITION BY,
--             running totals, time difference calculations,
--             filtering with window results via CTEs
-- Output:     Identifies high-risk repeat-failure machines and
--             quantifies mean time between failures — both direct
--             inputs to the feature engineering plan (days since
--             last failure, total prior failures).
-- Tables:     machines, failures
-- =============================================================


-- ---------------------------------------------------------------
-- BLOCK 1: Failure sequence per machine
-- Number each failure event per machine in chronological order.
-- ROW_NUMBER() OVER (PARTITION BY machine_id ORDER BY datetime)
-- ---------------------------------------------------------------
SELECT 
    machineid,
    datetime, 
    failure, 
    ROW_NUMBER() OVER (PARTITION BY machineid ORDER BY datetime) AS failure_number
FROM failures 

-- ---------------------------------------------------------------
-- BLOCK 2: Time between failures (MTBF proxy)
-- Use LAG() to get the previous failure datetime for each machine,
-- then calculate the gap in days.
-- Columns: machine_id, failure_type, datetime, prev_failure_datetime,
--          days_since_last_failure
-- ---------------------------------------------------------------
WITH failure_sequence AS (
    SELECT 
        machineid,
        failure AS failure_type,
        datetime,
        LAG(datetime) OVER (PARTITION BY machineid ORDER BY datetime) AS prev_failure_datetime
    FROM failures
)
SELECT 
    machineid,
    failure_type,
    datetime,
    prev_failure_datetime,
    ROUND(
        EXTRACT (EPOCH FROM (datetime - prev_failure_datetime)) / 86400.0
    , 1) AS days_since_prev_failure
FROM failure_sequence
ORDER BY machineid::INT, datetime;


-- ---------------------------------------------------------------
-- BLOCK 3: Average MTBF per machine
-- Aggregate the per-event gaps into a per-machine average.
-- Machines with short MTBF are higher priority for proactive dispatch
-- Columns: machine_id, total_failures, avg_days_between_failures
-- Filter: only machines with 2+ failures (need at least one gap)
-- ---------------------------------------------------------------
WITH failure_gaps AS (
    SELECT
        machineid,
        EXTRACT(EPOCH FROM (
            datetime - LAG(datetime) OVER (PARTITION BY machineid ORDER BY datetime))
        ) / 86400.0 AS days_since_prev_failure
    FROM failures
)
SELECT 
    machineid,
    COUNT(*) + 1 AS total_failures,
    ROUND(AVG(days_since_prev_failure)::NUMERIC, 1) AS avg_days_between_failures
FROM failure_gaps
WHERE days_since_prev_failure IS NOT NULL
GROUP BY machineid
HAVING COUNT(*) >= 1
ORDER BY avg_days_between_failures ASC; 


-- ---------------------------------------------------------------
-- BLOCK 4: Repeat offenders — top N machines by failure count
-- Running total of failures per machine to identify chronic assets.
-- Columns: machine_id, model, age, datetime, failure_type,
--          cumulative_failures (running total per machine)
-- ---------------------------------------------------------------
SELECT
    f.machineid,
    m.model,
    m.age,
    f.datetime,
    f.failure AS failure_type,
    ROW_NUMBER() OVER (
        PARTITION BY f.machineid
        ORDER BY f.datetime, f.failure
    ) AS cumulative_failures
FROM failures f
JOIN machines m ON f.machineid = m.machineid
ORDER BY f.machineid::INT, f.datetime;


-- ------------------------------------------------------------------
-- BLOCK 5: Failure volume over time (monthly trend)
-- Are failures increasing? Seasonal patterns? Gives ops context.
-- Demonstrates: DATE_TRUNC for time bucketing + GROUP BY
-- Columns: month, failure_count, rolling_3m_avg (optional stretch)
-- ------------------------------------------------------------------
WITH monthly_failures AS (
    SELECT 
        DATE_TRUNC('month', datetime) AS month,
        COUNT(*) AS failure_count
    FROM failures
    GROUP BY DATE_TRUNC('month', datetime)
)
SELECT 
    month,
    failure_count,
    ROUND(AVG(failure_count) OVER (
        ORDER BY month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW 
    ), 1) AS rolling_3m_avg
FROM monthly_failures
ORDER BY month;