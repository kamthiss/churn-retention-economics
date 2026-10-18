-- лидер на момент покупки — покупка с наибольшим (номер внутри категории, purchase_n) среди уже сделанных:
-- её категория самая частая, а при равенстве куплена позже остальных равных
create table mart.pauses as
with p as (
    select customer_id, order_date, purchase_n, category,
           lead(order_date) over (partition by customer_id order by purchase_n) as next_date,
           row_number() over (partition by customer_id, category order by purchase_n) as cat_n
    from mart.purchases
), lead_n as (
    select *,
           (max(array[cat_n, purchase_n]) over (partition by customer_id order by purchase_n))[2] as leader_n
    from p
)
select l.customer_id, l.order_date, l.purchase_n, l.next_date,
       l.next_date - l.order_date as gap,
       s.category as segment
from lead_n l
join mart.purchases s on s.customer_id = l.customer_id and s.purchase_n = l.leader_n;

alter table mart.pauses add primary key (customer_id, purchase_n);
