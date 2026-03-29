-- =============================================================
-- 06_query_optimisation.sql
-- Purpose:    Demonstrate query planning and index strategy
-- Business Q: Can we make the rolling telemetry feature query
--             fast enough to run in a production scoring pipeline?
-- Concepts:   EXPLAIN ANALYZE, reading execution plans,
--             index creation and its effect on query cost,
--             before/after performance narrative.
--             This file separates senior candidates from mid-level —
--             most DS candidates cannot read a query plan or
--             articulate an indexing strategy.
-- Output:     A documented before/after showing the impact of
--             indexing on a heavy time-series join. One clear
--             narrative is enough — do not over-engineer.
-- Tables:     telemetry, failures
-- Instructions: Run each block separately. Paste EXPLAIN ANALYZE
--               output as comments below each block so the GitHub
--               reader can see the plan without running the query.
-- =============================================================


-- ---------------------------------------------------------------
-- STEP 1: Baseline query — rolling telemetry + failure join
-- Use the heaviest query from 03_telemetry_patterns.sql or
-- 05_feature_candidates.sql as the target for optimisation.
-- Run with EXPLAIN ANALYZE first to see the baseline plan.
-- ---------------------------------------------------------------

EXPLAIN ANALYZE
-- TODO: Paste the target query here (e.g. rolling avg + failure join)
;

-- TODO: Paste the EXPLAIN ANALYZE output as a comment block here
-- e.g.:
-- QUERY PLAN
-- Seq Scan on telemetry  (cost=... rows=... width=...)
-- ...
-- Planning time: X ms
-- Execution time: X ms


-- ---------------------------------------------------------------
-- STEP 2: Identify the bottleneck
-- What does the plan show? Sequential scan? Nested loop?
-- Note the most expensive node and why it's slow.
-- ---------------------------------------------------------------

-- TODO: Write a comment explaining what the plan revealed:
-- e.g. "Sequential scan on telemetry (2M rows) was the bottleneck.
--       No index on (machine_id, datetime) so every row was read."


-- ---------------------------------------------------------------
-- STEP 3: Create targeted indexes
-- Index on the columns used in WHERE, JOIN ON, ORDER BY, PARTITION BY.
-- For time-series telemetry queries, the composite index on
-- (machine_id, datetime) is the high-value target.
-- ---------------------------------------------------------------

-- Index for telemetry time-series lookups
CREATE INDEX IF NOT EXISTS idx_telemetry_machine_datetime
    ON telemetry (machine_id, datetime);

-- Index for failure lookups by machine + time (used in label join)
CREATE INDEX IF NOT EXISTS idx_failures_machine_datetime
    ON failures (machine_id, datetime);

-- Index for error lookups (used in error feature aggregation)
CREATE INDEX IF NOT EXISTS idx_errors_machine_datetime
    ON errors (machine_id, datetime);


-- ---------------------------------------------------------------
-- STEP 4: Re-run with EXPLAIN ANALYZE after indexing
-- Same query as STEP 1. Compare the plan — look for Index Scan
-- replacing Seq Scan and the change in execution time.
-- ---------------------------------------------------------------

EXPLAIN ANALYZE
-- TODO: Same query as STEP 1
;

-- TODO: Paste the new EXPLAIN ANALYZE output as a comment block here
-- e.g.:
-- QUERY PLAN
-- Index Scan using idx_telemetry_machine_datetime on telemetry
-- ...
-- Planning time: X ms
-- Execution time: X ms  ← compare to baseline


-- ---------------------------------------------------------------
-- STEP 5: Before / after narrative (README-ready summary)
-- Write this in plain English — this is what a hiring manager reads.
-- ---------------------------------------------------------------

-- TODO: Write 3–5 sentences summarising:
-- 1. What the query does (business context)
-- 2. What the baseline plan showed and why it was slow
-- 3. What indexes were added and why
-- 4. The measured improvement (execution time before vs after)
-- 5. What this means for production scoring latency
--
-- Example structure:
-- "The rolling telemetry feature query performs a window aggregation
--  over 2M rows joined to the failures table. Without indexing,
--  PostgreSQL chose a sequential scan on telemetry (cost: X).
--  Adding a composite index on (machine_id, datetime) reduced
--  execution time from Xms to Yms — a Z% improvement — enabling
--  the query to run within the latency budget of the scoring pipeline."
