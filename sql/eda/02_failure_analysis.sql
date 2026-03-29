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

-- TODO: Write query here
-- Columns: machine_id, datetime, failure_type, failure_number


-- ---------------------------------------------------------------
-- BLOCK 2: Time between failures (MTBF proxy)
-- Use LAG() to get the previous failure datetime for each machine,
-- then calculate the gap in days.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: machine_id, failure_type, datetime, prev_failure_datetime,
--          days_since_last_failure


-- ---------------------------------------------------------------
-- BLOCK 3: Average MTBF per machine
-- Aggregate the per-event gaps into a per-machine average.
-- Machines with short MTBF are higher priority for proactive dispatch.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: machine_id, total_failures, avg_days_between_failures
-- Filter: only machines with 2+ failures (need at least one gap)


-- ---------------------------------------------------------------
-- BLOCK 4: Repeat offenders — top N machines by failure count
-- Running total of failures per machine to identify chronic assets.
-- Demonstrates: SUM() OVER (PARTITION BY ... ORDER BY ...) as
-- a running total — different from the aggregate window in 01.
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: machine_id, model, age, datetime, failure_type,
--          cumulative_failures (running total per machine)


-- ---------------------------------------------------------------
-- BLOCK 5: Failure volume over time (monthly trend)
-- Are failures increasing? Seasonal patterns? Gives ops context.
-- Demonstrates: DATE_TRUNC for time bucketing + GROUP BY
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: month, failure_count, rolling_3m_avg (optional stretch)
