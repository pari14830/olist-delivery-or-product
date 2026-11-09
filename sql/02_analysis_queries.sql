-- =====================================================================
-- Olist project - Part 2: clean table and analysis queries (MySQL)
-- Needs analysis_table from sql/01_build_analysis_table.sql
-- =====================================================================
use olist;


-- ---------------------------------------------------------------------
-- 1. cleaned_table: one row per order and seller, with
--    review_status (negative = score 1-2), is_late and is_different_state
-- Note: this table still has the orders with more than one seller
-- (one row per seller). In Python these orders were removed, so the
-- numbers here are a little different from the notebook.
-- ---------------------------------------------------------------------
drop table if exists cleaned_table;
create table cleaned_table as(
select distinct order_id ,  seller_id , customer_id , review_score,case
when review_score <= 2 then 'negative'
when review_score = 3 then 'neutral'
when review_score = 4 then 'positive'
else 'very positive'
end as review_status  , late_day ,case
when late_day >0 then  1
else 0
end as is_late
 , customer_state , seller_state,case
when seller_state != customer_state then 1
else 0
end as is_different_state  from analysis_table) ;

-- check: rows and sellers
select count(*),count(distinct seller_id) FROM cleaned_table;


-- ---------------------------------------------------------------------
-- 2. KPI: number of orders, negatives and negative rate,
--    on-time (is_late = 0) vs late (is_late = 1)
-- ---------------------------------------------------------------------
 select  is_late ,count(is_late) as number_of_orders ,
 sum(case when review_status = 'negative' then 1 else 0 end ) number_of_negatives,
 ROUND(SUM(CASE WHEN review_status = 'negative' THEN 1 ELSE 0 END) / COUNT(is_late) * 100, 1) AS negative_percent
 from cleaned_table  group by is_late ;


-- ---------------------------------------------------------------------
-- 3. Worst sellers on on-time orders (50+ orders)
-- Small differences from the pandas result are expected:
-- cleaned_table still contains multi-seller orders and duplicate reviews.
-- ---------------------------------------------------------------------
 select seller_id  ,count(*) as number_of_orders, SUM(CASE WHEN review_status = 'negative' THEN 1 ELSE 0 END) AS number_of_negatives,
 ROUND(SUM(CASE WHEN review_status = 'negative' THEN 1 ELSE 0 END) / COUNT(*) * 100, 1) AS negative_percent
 FROM cleaned_table
WHERE is_late = 0
GROUP BY seller_id
HAVING number_of_orders >= 50
ORDER BY negative_percent DESC
LIMIT 10;


-- ---------------------------------------------------------------------
-- 4. Negative rate by delay group
-- Same pattern as in pandas: a big jump after 3 days late, stable after 8 days.
-- The on-time rate is a little higher than in pandas (9.9% vs 8.7%),
-- because cleaned_table still contains multi-seller orders.
-- ---------------------------------------------------------------------
select  case
when late_day <=0 then'on time'
when late_day <= 3 then'1-3 days'
when late_day <=7 then '4-7 days'
when late_day <= 13 then  '8-13 days'
else '14+ days' end as delay_group ,count(*) as number_of_orders ,
SUM(CASE WHEN review_status = 'negative' THEN 1 ELSE 0 END) AS number_of_negatives,
ROUND(SUM(CASE WHEN review_status = 'negative' THEN 1 ELSE 0 END) / COUNT(*) * 100, 1) AS negative_percent
from cleaned_table group by delay_group order by MIN(late_day);


-- ---------------------------------------------------------------------
-- 5. Reviews written BEFORE the order was delivered
-- Result: 5,127 of 96,359 reviews (5.3%).
-- ---------------------------------------------------------------------
select count(*) as total_reviews, sum(case when r.review_creation_date < date(o.order_delivered_customer_date)
then 1 else 0 end) as review_before_delivery,
round(sum(case when r.review_creation_date < date(o.order_delivered_customer_date)
then 1 else 0 end)/count(*)*100,1) as percent_before_delivery FROM reviews as r
JOIN orders as  o ON r.order_id = o.order_id
WHERE o.order_delivered_customer_date IS NOT NULL;

-- Of these reviews, how many are negative, and how many orders are late?
-- Result: 71.1% negative, 93.5% late.
-- Meaning: many customers write a bad review while they are still
-- waiting for a late order.
SELECT
    COUNT(*) AS reviews_before_delivery,
    ROUND(AVG(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) * 100, 1) AS percent_negative,
    ROUND(AVG(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
                   THEN 1 ELSE 0 END) * 100, 1) AS percent_late
FROM reviews AS r
JOIN orders AS o ON r.order_id = o.order_id
WHERE o.order_delivered_customer_date IS NOT NULL
  AND r.review_creation_date < DATE(o.order_delivered_customer_date);
