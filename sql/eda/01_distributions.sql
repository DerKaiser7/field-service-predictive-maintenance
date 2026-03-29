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
        WHEN 'aging' THEN 4
    END;


-- ---------------------------------------------------------------------
-- BLOCK 3: Failure burden by model
-- Which model has the most failures? Is it proportional to fleet
-- size or are some models disproportionately failure-prone?
-- Demonstrates: multi-table join + aggregation + ratio calculation
-- Columns: model, machine_count, total_failures, failures_per_machine
-- ---------------------------------------------------------------------
SELECT 
    m.model,
    COUNT(DISTINCT m.machineID) AS machine_count,
    COUNT(f.machineID) AS total_failures,
    ROUND(
        COUNT(f.machineID)::NUMERIC / NULLIF(COUNT(DISTINCT m.machineID), 0)
    , 2) AS failures_per_machine,
    ROUND(
        100.0 * COUNT(f.machineid) / NULLIF(SUM(COUNT(f.machineid)) OVER (), 0)
    , 2) AS pct_of_total_failures
FROM machines m
LEFT JOIN failures f on m.machineid = f.machineid
GROUP BY m.model
ORDER BY total_failures DESC;


-- ---------------------------------------------------------------
-- BLOCK 4: Failure burden by age category
-- Do older machines fail more? This is the core hypothesis for
-- why machine age should be a feature in the model.
-- Demonstrates: CASE WHEN + GROUP BY + aggregate ratio
-- ---------------------------------------------------------------
WITH age_categories AS (
    SELECT  
        machineid,
        age,
        CASE
            WHEN age < 7 THEN 'young' 
            WHEN age >=7 AND age < 12 THEN 'midlife'
            WHEN age BETWEEN 12 AND 16 THEN 'mature'
            ELSE 'aging'
        END AS age_category
    FROM machines
)
SELECT 
    age_category,
    COUNT(DISTINCT ac.machineID) AS machine_count,
    COUNT(f.machineID) AS total_failures,
    ROUND(
        COUNT(f.machineid)::NUMERIC / NULLIF(COUNT(DISTINCT ac.machineid), 0)
    , 2) AS failures_per_machine,
    ROUND(
        100.0 * COUNT(f.machineid) / NULLIF(SUM(COUNT(f.machineid)) OVER (), 0)
    , 2) AS pct_of_total_failures
FROM age_categories ac
LEFT JOIN failures f on ac.machineID = f.machineid
GROUP BY age_category
ORDER BY 
    CASE age_category
        WHEN 'young' THEN 1
        WHEN 'midlife' THEN 2
        WHEN 'mature' THEN 3
        WHEN 'aging' THEN 4
    END;


-- ---------------------------------------------------------------
-- BLOCK 5: Failure type distribution by model (HAVING filter)
-- Are certain failure types concentrated in specific models?
-- Only show model/failure_type combos with more than N occurrences.
-- Demonstrates: multi-column GROUP BY, HAVING for post-agg filter
-- ---------------------------------------------------------------
SELECT 
    m.model,
    f.failure AS failure_type,
    COUNT(*) AS failure_count,
    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY m.model)
    , 2) AS pct_within_model
FROM failures f
JOIN machines m ON f.machineid = m.machineid
GROUP BY m.model, f.failure
HAVING COUNT(*) > 10
ORDER BY m.model, failure_count DESC 


-- ---------------------------------------------------------------
-- BLOCK 6: Pareto analysis — which machines drive most failures?
-- Business Q: What percentage of machines account for 80% of
--             failures? Identifies the chronic assets that a
--             proactive dispatch strategy should prioritise.
-- Concepts:   SUM() OVER (ORDER BY ... DESC) as a running total,
--             cumulative percentage, ROUND + filter to find the
--             80% threshold. Different window pattern to blocks
--             1–5 — ordered running total vs partition aggregate.
-- Output:     A ranked machine list with cumulative failure share.
--             The point where cumulative_pct crosses 80% defines
--             your high-priority asset cohort.
-- ---------------------------------------------------------------
WITH machine_failures AS (
    SELECT 
        m.machineID,
        m.model,
        m.age,
        COUNT(f.machineID) AS total_failures
    FROM machines m
    LEFT JOIN failures f ON m.machineid = f.machineid
    GROUP BY m.machineid, m.model, m.age   
), 
machine_pct AS (
    SELECT 
        machineid,
        model, 
        age,
        total_failures,
        100.0 * total_failures / NULLIF(SUM(total_failures) OVER (), 0) AS pct_of_total
    FROM machine_failures
)
SELECT 
    machineid,
    model, 
    age,
    total_failures,
    ROUND(pct_of_total, 2) AS pct_of_total,
    ROUND(
        SUM(pct_of_total) OVER (ORDER BY total_failures DESC, machineID)
    , 2) AS cumulative_pct
FROM machine_pct
ORDER BY total_failures DESC;


-- ---------------------------------------------------------------
-- BLOCK 7: Machines with zero failures
-- Business Q: What proportion of the fleet has never failed?
--             This directly quantifies label imbalance before
--             modelling — the lower this number, the more severe
--             the class imbalance problem.
-- Concepts:   Anti-join (LEFT JOIN / IS NULL) to find machines
--             with no matching failure records. Simple but
--             analytically important — pairs with the class
--             imbalance check in 00_data_quality.sql.
-- Output:     Count and percentage of never-failed machines.
--             Feeds directly into the decision to use SMOTE,
--             PR curves over ROC, and threshold tuning.
-- ---------------------------------------------------------------
SELECT 
    COUNT(DISTINCT m.machineID) AS total_machines,
    COUNT(CASE WHEN f.machineid IS NULL THEN 1 END) AS never_failed_count,
    ROUND(
        100.0 * COUNT(CASE WHEN f.machineID IS NULL THEN 1 END) /
        NULLIF(COUNT(DISTINCT m.machineID), 0)
    , 2) AS never_failed_pct
FROM machines m
LEFT JOIN failures f ON m.machineID = f.machineID;
 