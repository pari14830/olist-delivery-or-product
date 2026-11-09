-- =====================================================================
-- Olist project - Part 1: build the analysis table (MySQL)
-- The raw tables were loaded into MySQL from Python
-- (notebooks/01_data_understanding.ipynb).
-- =====================================================================


-- ---------------------------------------------------------------------
-- 1. Setup
-- ---------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS olist;
SHOW DATABASES;
use olist;
show tables;


-- ---------------------------------------------------------------------
-- 2. Checks on the orders table
-- ---------------------------------------------------------------------

-- total number of orders
select count(*) from orders;

-- orders by status
select count(*) as num_of_orders , order_status from orders group by order_status order by num_of_orders desc;

-- delivered orders with a delivery date (only these are used)
SELECT COUNT(*)
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL;

-- late_day = days between the real delivery and the estimated date
-- (positive = late, negative = early). The 10 latest orders:
SELECT
    order_id,
    DATEDIFF(order_delivered_customer_date,
            order_estimated_delivery_date) AS late_day
FROM
    orders
WHERE
    order_status = 'delivered'
        AND order_delivered_customer_date IS NOT NULL
        order by late_day desc
LIMIT 10;


-- ---------------------------------------------------------------------
-- 3. delivery_reviews: late_day and review score for each delivered order
-- ---------------------------------------------------------------------
 DROP TABLE IF EXISTS delivery_reviews;
 create  table delivery_reviews as (with late_delivery_days as(SELECT
    order_id,
    DATEDIFF(order_delivered_customer_date,
            order_estimated_delivery_date) AS late_day
FROM
    orders
WHERE
    order_status = 'delivered'
        AND order_delivered_customer_date IS NOT NULL
 )select L.order_id ,l.late_day ,r.review_score from late_delivery_days as l join reviews as r on l.order_id=r.order_id);

-- check: some orders have more than one review
 SELECT COUNT(*), COUNT(DISTINCT order_id) FROM delivery_reviews;
 SELECT order_id, COUNT(*) FROM delivery_reviews GROUP BY order_id HAVING COUNT(*) > 1 LIMIT 10;
 SELECT * FROM delivery_reviews WHERE order_id = 'a7dbcf5043158d6fa72859eead2f3d10';


-- ---------------------------------------------------------------------
-- 4. analysis_table: delivery_reviews + orders, customers, items,
--    products, English category names, sellers and payments
-- ---------------------------------------------------------------------
 DROP TABLE IF EXISTS analysis_table;
 create table analysis_table as (select d.* , o.customer_id , o.order_purchase_timestamp ,o.order_estimated_delivery_date, o.order_delivered_customer_date,
 c.customer_state , c.customer_zip_code_prefix,c.customer_city, s.seller_id ,s.seller_city, s.seller_state,s.seller_zip_code_prefix,
 i.product_id ,i.price,i.freight_value , p.product_category_name,pct.product_category_name_english ,pa.payment_value ,pa.payment_type
 from delivery_reviews as d join orders as o
 on d.order_id = o.order_id join customers as c on o.customer_id = c.customer_id left join items as i on i.order_id = o.order_id left join products as p on i.product_id= p.product_id
 left join product_category_name_translation as pct on pct.product_category_name = p.product_category_name left join sellers as s on s.seller_id = i.seller_id
 left join payments as pa on pa.order_id = d.order_id);

-- check: rows vs orders
-- Result: 114,854 rows but 95,824 orders. One order can have several rows
-- (several items or payments). One row per order is made in Python
-- (notebooks/02_cleaning_eda.ipynb).
 select count(*), count(distinct order_id) from analysis_table;
