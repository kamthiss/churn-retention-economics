drop schema if exists raw cascade;
create schema raw;

create table raw.customers (
    customer_id      integer primary key,
    signup_date      date not null,
    channel          text not null,
    region           text not null,
    last_order_date  date not null,
    lifetime_orders  integer not null,
    lifetime_revenue numeric(12,2) not null,
    loyalty_tier     text not null,
    status           text not null,
    unsubscribed_at  date
);

create table raw.orders (
    order_id    integer primary key,
    customer_id integer not null references raw.customers,
    order_date  date not null,
    status      text not null check (status in ('delivered', 'cancelled')),
    category    text not null,
    gross       numeric(12,2) not null,
    discount    numeric(12,2) not null,
    net         numeric(12,2) not null,
    cost        numeric(12,2) not null,
    promo_code  text
);

create index on raw.orders (customer_id, order_date);

create table raw.campaigns (
    campaign_id integer primary key,
    name        text not null,
    sent_at     date not null,
    promo_code  text not null,
    discount    numeric(12,2) not null,
    min_order   numeric(12,2) not null,
    valid_days  integer not null
);

create table raw.campaign_contacts (
    campaign_id integer not null references raw.campaigns,
    customer_id integer not null references raw.customers,
    sent_at     date not null,
    group_name  text not null check (group_name in ('treatment', 'control')),
    primary key (campaign_id, customer_id)
);
