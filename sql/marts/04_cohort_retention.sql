create table mart.cohort_retention as
with c as (
    select customer_id, date_trunc('month', min(order_date))::date as cohort
    from mart.purchases
    group by 1
), size as (
    select cohort, count(*) as cohort_size from c group by 1
), m as (
    select c.cohort, date_trunc('month', p.order_date)::date as month,
           count(distinct p.customer_id) as buyers, sum(p.margin) as margin
    from mart.purchases p
    join c using (customer_id)
    group by 1, 2
)
-- 2026-06 — последний месяц, целиком попавший в данные
select s.cohort,
       ((extract(year from g) - extract(year from s.cohort)) * 12
        + extract(month from g) - extract(month from s.cohort))::int as k,
       g::date as month,
       s.cohort_size,
       round(coalesce(m.buyers, 0)::numeric / s.cohort_size, 4) as retention,
       round(sum(coalesce(m.margin, 0)) over (partition by s.cohort order by g) / s.cohort_size, 2) as cum_margin
from size s
cross join generate_series(s.cohort, date '2026-06-01', interval '1 month') g
left join m on m.cohort = s.cohort and m.month = g
order by 1, 2;
