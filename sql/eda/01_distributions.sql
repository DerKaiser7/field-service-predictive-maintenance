-- =============================================================
-- 01_distributions.sql
-- Purpose:    Understand the shape and composition of the fleet
-- Business Q: What does our fleet look like? How are machines
--             distributed by model and age? How is failure burden
--             distributed across models and age groups?
-- Concepts:   GROUP BY, HAVING, COUNT/AVG/ROUND, multi-level
--             aggregation, CASE WHEN for inline categorisation,
--             window function inside aggregate: SUM(COUNT(*)) OVER()
-- Output:     Fleet composition baseline. Identifies which models
--             and age brackets carry the most failure risk —
--             the first slide in any ops review before modelling.
-- Tables:     machines, failures
-- =============================================================

-- ---------------------------------------------------------------
-- BLOCK 1: Fleet composition by model
-- How many machines per model? Average age? Share of total fleet?
-- ---------------------------------------------------------------
SELECT 
    model, 
    COUNT(*) AS machine_count,
    ROUND(AVG(age)::NUMERIC, 1) AS avg_age,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM machines
GROUP BY model
ORDER BY machine_count DESC;


-- ---------------------------------------------------------------
-- BLOCK 2: Age category distribution
-- Using data-driven quartile thresholds from 00_data_quality.sql:
--   young   → age < 7       (below p25)
--   midlife → age 7–11      (p25 to p50, exclusive)
--   mature  → age 12–16     (p50 to p75)
--   aging   → age > 16      (above p75)
-- How many machines fall into each life-stage bucket?
-- ---------------------------------------------------------------
WITH age_categories AS (
    SELECT 
        machineID, 
        age,
        CASE 
            WHEN age < 7 THEN 'young'
            WHEN age >= 7  AND age < 12 THEN 'midlife'
            WHEN age BETWEEN 12 AND 16 THEN 'mature'
            ELSE 'aging'
        END AS age_category
    FROM machines
)
SELECT 
    age_category, 
    COUNT(*) as category_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total 
FROM age_categories
GROUP BY age_category
ORDER BY 
    CASE age_category
        WHEN 'young' THEN 1
        WHEN 'midlife' THEN 2
        WHEN 'mature' THEN 3
        WHEN 'mature' THEN 4
    END;


-- ---------------------------------------------------------------
-- BLOCK 3: Failure burden by model
-- Which model has the most failures? Is it proportional to fleet
-- size or are some models disproportionately failure-prone?
-- Demonstrates: multi-table join + aggregation + ratio calculation
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: model, machine_count, total_failures, failures_per_machine


-- ---------------------------------------------------------------
-- BLOCK 4: Failure burden by age category
-- Do older machines fail more? This is the core hypothesis for
-- why machine age should be a feature in the model.
-- Demonstrates: CASE WHEN + GROUP BY + aggregate ratio
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: age_category, machine_count, total_failures, avg_failures_per_machine


-- ---------------------------------------------------------------
-- BLOCK 5: Failure type distribution by model (HAVING filter)
-- Are certain failure types concentrated in specific models?
-- Only show model/failure_type combos with more than N occurrences.
-- Demonstrates: multi-column GROUP BY, HAVING for post-agg filter
-- ---------------------------------------------------------------

-- TODO: Write query here
-- Columns: model, failure_type, count — filter with HAVING > threshold