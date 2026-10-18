-- 2026-06-30 — последний день данных: берём паузы, у которых окно N + 365 дней целиком наблюдаемо
create table mart.false_churn as
with p as (
    select t.n, p.segment,
           case when p.purchase_n = 1 then 'first' else 'repeat' end as pause,
           p.gap <= t.n + 365 as returned
    from mart.pauses p
    cross join (values (60), (90), (120), (180)) t(n)
    where p.order_date + t.n + 365 <= date '2026-06-30'
      and (p.gap > t.n or p.gap is null)
)
select n,
       coalesce(segment, 'all') as segment,
       coalesce(pause, 'all') as pause,
       count(*) as pauses_over_n,
       count(*) filter (where returned) as returned,
       round(count(*) filter (where returned)::numeric / count(*), 4) as share
from p
group by n, grouping sets ((segment, pause), (segment), (pause), ())
order by 1, 2, 3;
