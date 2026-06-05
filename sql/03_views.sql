-- analytical views created:
-- vw_geo_zip
-- vw_products_enriched
-- vw_customers_enriched
-- vw_sellers_enriched
-- vw_payments_agg
-- vw_reviews_agg
-- vw_orders_enriched
-- vw_completed_orders
-- vw_delivered_orders
-- vw_completed_items
-- vw_category_summary
-- vw_seller_summary
-- vw_state_summary
-- vw_route_summary
-- vw_distance_summary
-- vw_customer_level
-- vw_monthly_customer_mix
-- vw_cohort_retention

-- 1. Product, geolocation, customer, seller, payment, review, and order enrichment views

CREATE OR REPLACE VIEW vw_products_enriched AS
SELECT
    p.product_id,
    COALESCE(p.product_category_name, 'unknown') AS product_category_name,
    COALESCE(t.product_category_name_english, 'unknown') AS product_category_name_english,
    p.product_name_lenght,
    p.product_description_lenght,
    p.product_photos_qty,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm
FROM products p
LEFT JOIN product_category_translation t
    ON p.product_category_name = t.product_category_name;


CREATE OR REPLACE VIEW vw_geo_zip AS
SELECT
    geolocation_zip_code_prefix,
    AVG(geolocation_lat) AS geolocation_lat,
    AVG(geolocation_lng) AS geolocation_lng,
    MODE() WITHIN GROUP (ORDER BY geolocation_city) AS geolocation_city,
    MODE() WITHIN GROUP (ORDER BY geolocation_state) AS geolocation_state
FROM geolocation
GROUP BY geolocation_zip_code_prefix;


CREATE OR REPLACE VIEW vw_customers_enriched AS
SELECT
    c.customer_id,
    c.customer_unique_id,
    c.customer_zip_code_prefix,
    c.customer_city,
    c.customer_state,
    g.geolocation_lat AS customer_lat,
    g.geolocation_lng AS customer_lng,
    g.geolocation_city AS customer_geo_city,
    g.geolocation_state AS customer_geo_state
FROM customers c
LEFT JOIN vw_geo_zip g
    ON c.customer_zip_code_prefix = g.geolocation_zip_code_prefix;


CREATE OR REPLACE VIEW vw_sellers_enriched AS
SELECT
    s.seller_id,
    s.seller_zip_code_prefix,
    s.seller_city,
    s.seller_state,
    g.geolocation_lat AS seller_lat,
    g.geolocation_lng AS seller_lng,
    g.geolocation_city AS seller_geo_city,
    g.geolocation_state AS seller_geo_state
FROM sellers s
LEFT JOIN vw_geo_zip g
    ON s.seller_zip_code_prefix = g.geolocation_zip_code_prefix;


CREATE OR REPLACE VIEW vw_payments_agg AS
SELECT
    order_id,
    SUM(payment_value) AS payment_value,
    MAX(payment_installments) AS payment_installments,
    MODE() WITHIN GROUP (ORDER BY payment_type) AS payment_type,
    COUNT(DISTINCT payment_type) AS payment_type_nunique
FROM order_payments
GROUP BY order_id;


CREATE OR REPLACE VIEW vw_reviews_agg AS
SELECT
    order_id,
    AVG(review_score) AS review_score,
    AVG(
        CASE
            WHEN review_comment_message IS NOT NULL THEN 1.0
            ELSE 0.0
        END
    ) AS review_comment_rate,
    COUNT(DISTINCT review_id) AS review_count
FROM order_reviews
GROUP BY order_id;


CREATE OR REPLACE VIEW vw_orders_enriched AS
SELECT
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,

    c.customer_unique_id,
    c.customer_zip_code_prefix,
    c.customer_city,
    c.customer_state,
    c.customer_lat,
    c.customer_lng,
    c.customer_geo_city,
    c.customer_geo_state,

    p.payment_value,
    p.payment_installments,
    p.payment_type,
    p.payment_type_nunique,

    r.review_score,
    r.review_comment_rate,
    r.review_count
FROM orders o
LEFT JOIN vw_customers_enriched c
    ON o.customer_id = c.customer_id
LEFT JOIN vw_payments_agg p
    ON o.order_id = p.order_id
LEFT JOIN vw_reviews_agg r
    ON o.order_id = r.order_id;


-- 2. Item-level base view

CREATE OR REPLACE VIEW vw_order_item_base AS
SELECT
    oi.order_id,
    oi.order_item_id,
    oi.product_id,
    oi.seller_id,
    oi.shipping_limit_date,
    oi.price,
    oi.freight_value,

    p.product_category_name,
    p.product_category_name_english,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm,

    s.seller_zip_code_prefix,
    s.seller_city,
    s.seller_state,
    s.seller_lat,
    s.seller_lng,

    o.customer_id,
    o.customer_unique_id,
    o.customer_zip_code_prefix,
    o.customer_city,
    o.customer_state,
    o.customer_lat,
    o.customer_lng,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    o.payment_value,
    o.payment_installments,
    o.payment_type,
    o.payment_type_nunique,
    o.review_score,
    o.review_comment_rate,
    o.review_count,

    oi.price + oi.freight_value AS item_gmv,

    oi.freight_value / NULLIF((oi.price + oi.freight_value), 0) AS freight_share,

    COALESCE(p.product_length_cm, 0)
        * COALESCE(p.product_height_cm, 0)
        * COALESCE(p.product_width_cm, 0) AS volume_cm3,

    CASE
        WHEN o.customer_state = s.seller_state THEN 1
        ELSE 0
    END AS same_state_route

FROM order_items oi
LEFT JOIN vw_products_enriched p
    ON oi.product_id = p.product_id
LEFT JOIN vw_sellers_enriched s
    ON oi.seller_id = s.seller_id
LEFT JOIN vw_orders_enriched o
    ON oi.order_id = o.order_id;


-- 3. Geographic distance view

CREATE OR REPLACE VIEW vw_order_item_geo AS
SELECT
    *,

    CASE
        WHEN customer_lat IS NOT NULL
         AND customer_lng IS NOT NULL
         AND seller_lat IS NOT NULL
         AND seller_lng IS NOT NULL
        THEN
            6371 * 2 * ASIN(
                SQRT(
                    POWER(SIN(RADIANS(seller_lat - customer_lat) / 2), 2)
                    +
                    COS(RADIANS(customer_lat))
                    * COS(RADIANS(seller_lat))
                    * POWER(SIN(RADIANS(seller_lng - customer_lng) / 2), 2)
                )
            )
        ELSE NULL
    END AS distance_km

FROM vw_order_item_base;


-- 4. Order-level analytical view

CREATE OR REPLACE VIEW vw_order_level AS
WITH order_agg AS (
    SELECT
        order_id,

        MAX(customer_unique_id) AS customer_unique_id,
        MAX(customer_state) AS customer_state,
        MAX(order_status) AS order_status,
        MAX(order_purchase_timestamp) AS order_purchase_timestamp,
        MAX(order_approved_at) AS order_approved_at,
        MAX(order_delivered_customer_date) AS order_delivered_customer_date,
        MAX(order_estimated_delivery_date) AS order_estimated_delivery_date,

        MAX(review_score) AS review_score,
        MAX(payment_value) AS payment_value,
        MAX(payment_installments) AS payment_installments,
        MODE() WITHIN GROUP (ORDER BY payment_type) AS payment_type,

        SUM(item_gmv) AS order_gmv,
        SUM(freight_value) AS freight_gmv,
        COUNT(order_item_id) AS item_count,
        COUNT(DISTINCT seller_id) AS seller_count,
        COUNT(DISTINCT product_category_name_english) AS category_count,
        MODE() WITHIN GROUP (ORDER BY product_category_name_english) AS dominant_category,
        AVG(distance_km) AS avg_distance_km

    FROM vw_order_item_geo
    GROUP BY order_id
)

SELECT
    *,

    EXTRACT(
        EPOCH FROM (
            order_delivered_customer_date - order_purchase_timestamp
        )
    ) / 86400.0 AS delivery_days,

    EXTRACT(
        EPOCH FROM (
            order_delivered_customer_date - order_estimated_delivery_date
        )
    ) / 86400.0 AS estimated_vs_actual_days,

    EXTRACT(
        EPOCH FROM (
            order_approved_at - order_purchase_timestamp
        )
    ) / 3600.0 AS approval_lag_hours,

    CASE
        WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1
        ELSE 0
    END AS is_late_delivery,

    DATE_TRUNC('month', order_purchase_timestamp)::date AS purchase_month,

    payment_value - order_gmv AS gmv_payment_gap

FROM order_agg;


-- 5. Completed and delivered order/item filter views

CREATE OR REPLACE VIEW vw_completed_orders AS
SELECT *
FROM vw_order_level
WHERE order_status IN (
    'delivered',
    'shipped',
    'invoiced',
    'processing',
    'approved'
);

CREATE OR REPLACE VIEW vw_delivered_orders AS
SELECT *
FROM vw_order_level
WHERE order_status = 'delivered';

CREATE OR REPLACE VIEW vw_completed_items AS
SELECT *
FROM vw_order_item_geo
WHERE order_status IN (
    'delivered',
    'shipped',
    'invoiced',
    'processing',
    'approved'
);

CREATE OR REPLACE VIEW vw_delivered_items AS
SELECT *
FROM vw_order_item_geo
WHERE order_status = 'delivered';


-- 6. Customer-level analytical view

CREATE OR REPLACE VIEW vw_customer_level AS
WITH analysis_window AS (
    SELECT
        MAX(order_purchase_timestamp) AS analysis_end
    FROM vw_completed_orders
),

customer_agg AS (
    SELECT
        customer_unique_id,

        SUM(order_gmv) AS total_gmv,
        COUNT(DISTINCT order_id) AS total_orders,

        MIN(order_purchase_timestamp) AS first_purchase,
        MAX(order_purchase_timestamp) AS last_purchase,

        AVG(order_gmv) AS avg_order_value,
        AVG(review_score) AS avg_review_score,
        AVG(delivery_days) AS avg_delivery_days,
        AVG(is_late_delivery::numeric) AS late_delivery_rate

    FROM vw_completed_orders
    GROUP BY customer_unique_id
),

customer_features AS (
    SELECT
        c.*,

        EXTRACT(
            EPOCH FROM (
                aw.analysis_end - c.last_purchase
            )
        ) / 86400.0 AS recency_days,

        GREATEST(
            EXTRACT(
                EPOCH FROM (
                    c.last_purchase - c.first_purchase
                )
            ) / 86400.0,
            0
        ) AS tenure_days,

        CASE
            WHEN EXTRACT(
                EPOCH FROM (
                    c.last_purchase - c.first_purchase
                )
            ) / 86400.0 > 0
            THEN
                c.total_orders / (
                    (
                        EXTRACT(
                            EPOCH FROM (
                                c.last_purchase - c.first_purchase
                            )
                        ) / 86400.0
                    ) / 30.4
                )
            ELSE NULL
        END AS purchase_frequency_per_month,

        c.total_gmv AS realized_ltv,

        CASE
            WHEN c.total_orders > 1 THEN 1
            ELSE 0
        END AS repeat_customer

    FROM customer_agg c
    CROSS JOIN analysis_window aw
),

customer_segments AS (
    SELECT
        *,

        NTILE(4) OVER (
            ORDER BY realized_ltv
        ) AS value_quartile

    FROM customer_features
)

SELECT
    *,

    CASE
        WHEN value_quartile = 1 THEN 'Low'
        WHEN value_quartile = 2 THEN 'Mid-Low'
        WHEN value_quartile = 3 THEN 'Mid-High'
        WHEN value_quartile = 4 THEN 'High'
    END AS value_segment

FROM customer_segments;


-- 7. Monthly marketplace KPI view

CREATE OR REPLACE VIEW vw_monthly_metrics AS
SELECT
    purchase_month,
    SUM(order_gmv) AS gmv,
    COUNT(DISTINCT order_id) AS orders,
    COUNT(DISTINCT customer_unique_id) AS active_customers,
    SUM(order_gmv) / NULLIF(COUNT(DISTINCT order_id), 0) AS aov
FROM vw_completed_orders
GROUP BY purchase_month;


-- 8. Monthly customer acquisition and repeat behavior view

CREATE OR REPLACE VIEW vw_monthly_customer_mix AS
WITH customer_order_sequence AS (
    SELECT
        order_id,
        customer_unique_id,
        purchase_month,
        order_purchase_timestamp,

        ROW_NUMBER() OVER (
            PARTITION BY customer_unique_id
            ORDER BY order_purchase_timestamp, order_id
        ) AS order_rank

    FROM vw_completed_orders
),

order_flags AS (
    SELECT
        *,
        CASE
            WHEN order_rank = 1 THEN 1
            ELSE 0
        END AS is_new_order,

        CASE
            WHEN order_rank > 1 THEN 1
            ELSE 0
        END AS is_repeat_order

    FROM customer_order_sequence
)

SELECT
    purchase_month,
    COUNT(DISTINCT customer_unique_id) AS active_customers,
    SUM(is_new_order) AS new_customers,
    SUM(is_repeat_order) AS repeat_orders,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_unique_id) - SUM(is_new_order) AS returning_customer_proxy,
    SUM(is_repeat_order)::numeric / NULLIF(COUNT(DISTINCT order_id), 0) AS repeat_order_share
FROM order_flags
GROUP BY purchase_month;


-- 9. Category and seller KPI views

CREATE OR REPLACE VIEW vw_category_summary AS
SELECT
    product_category_name_english,
    SUM(item_gmv) AS gmv,
    COUNT(DISTINCT order_id) AS orders,
    AVG(review_score) AS avg_review,
    SUM(freight_value) / NULLIF(SUM(item_gmv), 0) AS avg_freight_share
FROM vw_completed_items
GROUP BY product_category_name_english;


CREATE OR REPLACE VIEW vw_seller_summary AS
WITH seller_base AS (
    SELECT
        i.seller_id,
        SUM(i.item_gmv) AS gmv,
        COUNT(DISTINCT i.order_id) AS orders,
        COUNT(DISTINCT i.customer_unique_id) AS customers,
        AVG(i.review_score) AS avg_review,
        AVG(o.is_late_delivery::numeric) AS late_delivery_rate
    FROM vw_completed_items i
    LEFT JOIN vw_order_level o
        ON i.order_id = o.order_id
    GROUP BY i.seller_id
)

SELECT
    *,
    SUM(gmv) OVER (
        ORDER BY gmv DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) / SUM(gmv) OVER () AS cumulative_gmv_share
FROM seller_base;


-- 10. Geographic, route, and distance KPI views

CREATE OR REPLACE VIEW vw_state_summary AS
SELECT
    customer_state,
    SUM(order_gmv) AS gmv,
    COUNT(DISTINCT order_id) AS orders,
    COUNT(DISTINCT customer_unique_id) AS customers,
    AVG(review_score) AS avg_review,
    SUM(order_gmv) / NULLIF(COUNT(DISTINCT order_id), 0) AS avg_order_value
FROM vw_completed_orders
GROUP BY customer_state;


CREATE OR REPLACE VIEW vw_route_summary AS
SELECT
    CASE
        WHEN same_state_route = 1 THEN 'same_state'
        ELSE 'cross_state'
    END AS route_type,

    COUNT(*) AS items,
    SUM(item_gmv) AS gmv,
    AVG(freight_value) AS avg_freight,
    AVG(review_score) AS avg_review,
    SUM(item_gmv) / NULLIF(
        SUM(SUM(item_gmv)) OVER (),
        0
    ) AS gmv_share
FROM vw_completed_items
GROUP BY
    CASE
        WHEN same_state_route = 1 THEN 'same_state'
        ELSE 'cross_state'
    END;


CREATE OR REPLACE VIEW vw_distance_summary AS
WITH distance_base AS (
    SELECT
        distance_km,
        freight_value,
        review_score,
        NTILE(5) OVER (
            ORDER BY distance_km
        ) AS distance_bucket
    FROM vw_completed_items
    WHERE distance_km IS NOT NULL
)

SELECT
    distance_bucket,
    AVG(distance_km) AS avg_distance_km,
    AVG(freight_value) AS avg_freight,
    AVG(review_score) AS avg_review
FROM distance_base
GROUP BY distance_bucket;


-- 11. Cohort retention view

CREATE OR REPLACE VIEW vw_cohort_retention AS
WITH cohort_base AS (
    SELECT
        customer_unique_id,
        order_id,
        DATE_TRUNC('month', order_purchase_timestamp)::date AS order_month,
        MIN(DATE_TRUNC('month', order_purchase_timestamp)::date) OVER (
            PARTITION BY customer_unique_id
        ) AS cohort_month
    FROM vw_completed_orders
),

cohort_activity AS (
    SELECT
        customer_unique_id,
        cohort_month,
        order_month,
        (
            (EXTRACT(YEAR FROM order_month) - EXTRACT(YEAR FROM cohort_month)) * 12
            + (EXTRACT(MONTH FROM order_month) - EXTRACT(MONTH FROM cohort_month))
        )::int AS cohort_index
    FROM cohort_base
),

cohort_sizes AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT customer_unique_id) AS cohort_size
    FROM cohort_activity
    WHERE cohort_index = 0
    GROUP BY cohort_month
),

cohort_counts AS (
    SELECT
        cohort_month,
        cohort_index,
        COUNT(DISTINCT customer_unique_id) AS customers
    FROM cohort_activity
    GROUP BY
        cohort_month,
        cohort_index
),

max_month AS (
    SELECT
        MAX(order_month) AS max_order_month
    FROM cohort_activity
),

cohort_grid AS (
    SELECT
        cs.cohort_month,
        generate_series(
            0,
            (
                (EXTRACT(YEAR FROM mm.max_order_month) - EXTRACT(YEAR FROM cs.cohort_month)) * 12
                + (EXTRACT(MONTH FROM mm.max_order_month) - EXTRACT(MONTH FROM cs.cohort_month))
            )::int
        ) AS cohort_index,
        cs.cohort_size
    FROM cohort_sizes cs
    CROSS JOIN max_month mm
)

SELECT
    g.cohort_month,
    g.cohort_index,
    COALESCE(c.customers, 0) AS customers,
    g.cohort_size,
    COALESCE(c.customers, 0)::numeric / NULLIF(g.cohort_size, 0) AS retention
FROM cohort_grid g
LEFT JOIN cohort_counts c
    ON g.cohort_month = c.cohort_month
   AND g.cohort_index = c.cohort_index;