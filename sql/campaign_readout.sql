-- Campaign readout: one row per test arm, with absolute and relative lift vs the control group.
-- Run with DuckDB from the repo root, e.g.  duckdb -c ".read sql/campaign_readout.sql"

CREATE OR REPLACE VIEW customers AS
SELECT
    row_number() OVER () AS customer_id,
    *,
    CASE
        WHEN mens = 1 AND womens = 1 THEN 'both'
        WHEN mens = 1 THEN 'mens only'
        ELSE 'womens only'
    END AS buyer_type
FROM read_csv_auto('data/raw/hillstrom.csv.gz');

-- 1. Headline readout per arm
WITH arms AS (
    SELECT
        segment AS arm,
        count(*)                AS customers,
        avg(visit)              AS visit_rate,
        avg(conversion)         AS conversion_rate,
        avg(spend)              AS spend_per_customer,
        var_samp(spend)         AS spend_var
    FROM customers
    GROUP BY 1
),
control AS (SELECT * FROM arms WHERE arm = 'No E-Mail')
SELECT
    a.arm,
    a.customers,
    round(a.visit_rate * 100, 2)                                  AS visit_pct,
    round((a.visit_rate - c.visit_rate) * 100, 2)                 AS visit_lift_pts,
    round(a.conversion_rate * 100, 3)                             AS conversion_pct,
    round((a.conversion_rate / c.conversion_rate - 1) * 100, 0)   AS conversion_lift_pct,
    round(a.spend_per_customer, 3)                                AS spend_per_customer,
    round(a.spend_per_customer - c.spend_per_customer, 3)         AS incremental_spend,
    -- 95% interval for the spend difference (Welch, normal approximation)
    CASE WHEN a.arm <> 'No E-Mail' THEN round(a.spend_per_customer - c.spend_per_customer
          - 1.96 * sqrt(a.spend_var / a.customers + c.spend_var / c.customers), 3) END AS incremental_spend_low,
    CASE WHEN a.arm <> 'No E-Mail' THEN round(a.spend_per_customer - c.spend_per_customer
          + 1.96 * sqrt(a.spend_var / a.customers + c.spend_var / c.customers), 3) END AS incremental_spend_high
FROM arms a CROSS JOIN control c
ORDER BY a.arm;

-- 2. Visit lift by buyer type and arm (who responds to which email?)
WITH cell AS (
    SELECT buyer_type, segment AS arm, avg(visit) AS visit_rate, count(*) AS n
    FROM customers
    GROUP BY 1, 2
)
SELECT
    t.buyer_type,
    t.arm,
    t.n,
    round(t.visit_rate * 100, 1)                     AS visit_pct,
    round((t.visit_rate - c.visit_rate) * 100, 1)    AS visit_lift_pts
FROM cell t
JOIN cell c ON c.buyer_type = t.buyer_type AND c.arm = 'No E-Mail'
WHERE t.arm <> 'No E-Mail'
ORDER BY t.buyer_type, t.arm;

-- 3. Randomisation check: covariate means should be nearly identical across arms
SELECT
    segment AS arm,
    round(avg(recency), 2)  AS avg_recency_months,
    round(avg(history), 1)  AS avg_history_usd,
    round(avg(newbie), 3)   AS share_new_customers,
    round(avg(mens), 3)     AS share_bought_mens,
    round(avg(womens), 3)   AS share_bought_womens
FROM customers
GROUP BY 1
ORDER BY 1;
