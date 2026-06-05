-- Validate raw tables, analytical views, and key project metrics

-- 1. Raw table row counts

SELECT *
FROM (
    SELECT
        'customers' AS table_name,
        COUNT(*) AS row_count
    FROM customers

    UNION ALL

    SELECT
        'geolocation' AS table_name,
        COUNT(*) AS row_count
    FROM geolocation

    UNION ALL

    SELECT
        'order_items' AS table_name,
        COUNT(*) AS row_count
    FROM order_items

    UNION ALL

    SELECT
        'order_payments' AS table_name,
        COUNT(*) AS row_count
    FROM order_payments

    UNION ALL

    SELECT
        'order_reviews' AS table_name,
        COUNT(*) AS row_count
    FROM order_reviews

    UNION ALL

    SELECT
        'orders' AS table_name,
        COUNT(*) AS row_count
    FROM orders

    UNION ALL

    SELECT
        'products' AS table_name,
        COUNT(*) AS row_count
    FROM products

    UNION ALL

    SELECT
        'sellers' AS table_name,
        COUNT(*) AS row_count
    FROM sellers

    UNION ALL

    SELECT
        'product_category_translation' AS table_name,
        COUNT(*) AS row_count
    FROM product_category_translation
) table_counts
ORDER BY table_name;


-- 2. Analytical views created

SELECT
    table_name AS view_name
FROM information_schema.views
WHERE table_schema = 'public'
  AND table_name LIKE 'vw_%'
ORDER BY table_name;


-- 3. Purchase coverage

SELECT
    MIN(order_purchase_timestamp)::date AS purchase_start,
    MAX(order_purchase_timestamp)::date AS purchase_end
FROM orders;


-- 4. Completed orders summary

SELECT
    COUNT(DISTINCT order_id) AS completed_orders,
    SUM(order_gmv) AS total_gmv,
    AVG(order_gmv) AS avg_order_gmv
FROM vw_completed_orders;


-- 5. Repeat customer rate

SELECT
    AVG(repeat_customer::numeric) AS repeat_customer_rate
FROM vw_customer_level;


-- 6. Top categories by GMV

SELECT
    product_category_name_english,
    gmv,
    orders,
    avg_review,
    avg_freight_share
FROM vw_category_summary
ORDER BY gmv DESC
LIMIT 10;


-- 7. Cohort retention check

SELECT
    cohort_month,
    cohort_index,
    customers,
    cohort_size,
    retention
FROM vw_cohort_retention
WHERE cohort_size >= 100
  AND cohort_index <= 12
ORDER BY cohort_month, cohort_index;