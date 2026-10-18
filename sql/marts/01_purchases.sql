create table mart.purchases as
select o.*,
       o.net - o.cost as margin,
       row_number() over w as purchase_n,
       lag(o.order_date) over w as prev_date,
       o.order_date - lag(o.order_date) over w as gap_from_prev
from raw.orders o
where o.status = 'delivered'
window w as (partition by o.customer_id order by o.order_date, o.order_id);

alter table mart.purchases add primary key (order_id);
create unique index on mart.purchases (customer_id, purchase_n);
