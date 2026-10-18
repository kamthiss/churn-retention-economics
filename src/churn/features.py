from datetime import timedelta

import numpy as np
import pandas as pd

from churn.db import connect
from churn.params import CAMPAIGN_DATE, CAMPAIGN_PROMO, CATEGORIES, CHURN_N, TEST_POINT, TRAIN_POINTS

FEATURES = [
    "days_since_last", "days_since_first",
    "purchases", "purchases_30d", "purchases_90d", "purchases_120d",
    "net_total", "margin_total", "avg_net",
    "median_gap", "last_gap", "recency_to_median_gap",
    "main_category", *(f"share_{c}" for c in CATEGORIES),
    "share_site10", "purchases_back300", "cancelled",
    "channel", "region", "got_campaign",
]

DDL = """
drop table if exists mart.model_dataset;
create table mart.model_dataset (
    point                 date not null,
    customer_id           integer not null,
    days_since_last       integer not null,
    days_since_first      integer not null,
    purchases             integer not null,
    purchases_30d         integer not null,
    purchases_90d         integer not null,
    purchases_120d        integer not null,
    net_total             float8 not null,
    margin_total          float8 not null,
    avg_net               float8 not null,
    median_gap            float8,
    last_gap              integer,
    recency_to_median_gap float8,
    main_category         text not null,
    share_cat_food        float8 not null,
    share_dog_food        float8 not null,
    share_accessories     float8 not null,
    share_site10          float8 not null,
    purchases_back300     integer not null,
    cancelled             integer not null,
    channel               text not null,
    region                text not null,
    got_campaign          boolean not null,
    churn                 smallint not null,
    is_test               boolean not null,
    excluded_campaign     boolean not null,
    primary key (point, customer_id)
);
"""


def query(conn, sql, params=None):
    """Результат запроса как DataFrame."""
    cur = conn.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def history(conn, t, schema="raw"):
    """Заказы, получатели кампании и клиенты: только то, что известно до t."""
    orders = query(conn, f"""
        select customer_id, order_id, order_date, status, category,
               net::float8 as net, (net - cost)::float8 as margin, promo_code
        from {schema}.orders where order_date < %s""", (t,))
    treated = query(conn, f"""
        select customer_id from {schema}.campaign_contacts
        where sent_at < %s and group_name = 'treatment'""", (t,))
    customers = query(conn, f"select customer_id, channel, region from {schema}.customers")
    return orders, treated, customers


def features(orders, treated, customers, t):
    """Признаки клиентов, активных на t."""
    t = pd.Timestamp(t)
    o = orders.assign(order_date=pd.to_datetime(orders["order_date"]))
    # порядок строк фиксирует порядок сложения float: иначе суммы зависят от того, как база отдала строки
    o = o.sort_values(["customer_id", "order_date", "order_id"], ignore_index=True)
    p = o[o["status"] == "delivered"].copy()
    p["days"] = (t - p["order_date"]).dt.days
    active = p.loc[p["days"] <= CHURN_N, "customer_id"].unique()
    p = p[p["customer_id"].isin(active)].reset_index(drop=True)
    cid = p["customer_id"]

    df = p.groupby("customer_id").agg(
        days_since_last=("days", "min"), days_since_first=("days", "max"), purchases=("days", "size"),
        net_total=("net", "sum"), margin_total=("margin", "sum"),
    )
    for d in (30, 90, 120):
        df[f"purchases_{d}d"] = (p["days"] <= d).groupby(cid).sum()
    df["avg_net"] = df["net_total"] / df["purchases"]

    gap = p.groupby("customer_id")["order_date"].diff().dt.days
    df["median_gap"] = gap.groupby(cid).median()
    df["last_gap"] = gap.groupby(cid).last().astype("Int64")
    df["recency_to_median_gap"] = df["days_since_last"] / df["median_gap"]

    # как в mart.pauses: самая частая категория, при равенстве — купленная позже
    c = p.assign(i=np.arange(len(p))).groupby(["customer_id", "category"])["i"].agg(["size", "max"])
    df["main_category"] = c.sort_values(["size", "max"]).reset_index().groupby("customer_id")["category"].last()
    share = pd.crosstab(cid, p["category"], normalize="index").reindex(columns=list(CATEGORIES), fill_value=0.0)
    for cat in CATEGORIES:
        df[f"share_{cat}"] = share[cat]
    df["share_site10"] = (p["promo_code"] == "SITE10").groupby(cid).mean()
    df["purchases_back300"] = (p["promo_code"] == CAMPAIGN_PROMO).groupby(cid).sum()

    cancelled = o[o["status"] == "cancelled"].groupby("customer_id").size()
    df["cancelled"] = cancelled.reindex(df.index, fill_value=0)
    df = df.join(customers.set_index("customer_id")[["channel", "region"]])
    df["got_campaign"] = df.index.isin(treated["customer_id"])
    return df.reset_index()[["customer_id", *FEATURES]]


def buyers(conn, t, schema="raw"):
    """Клиенты с покупкой в окне [t, t + N)."""
    return query(conn, f"""
        select distinct customer_id from {schema}.orders
        where status = 'delivered' and order_date >= %s and order_date < %s""",
                 (t, t + timedelta(days=CHURN_N)))["customer_id"]


def build():
    """Пересоздаёт mart.model_dataset по всем точкам прогноза."""
    with connect() as conn:
        participants = query(conn, "select customer_id from raw.campaign_contacts")["customer_id"]
        parts = []
        for t in (*TRAIN_POINTS, TEST_POINT):
            df = features(*history(conn, t), t)
            df["churn"] = (~df["customer_id"].isin(buyers(conn, t))).astype(int)
            df["is_test"] = t == TEST_POINT
            overlap = t <= CAMPAIGN_DATE < t + timedelta(days=CHURN_N)
            df["excluded_campaign"] = overlap & df["customer_id"].isin(participants)
            df.insert(0, "point", t)
            parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        conn.execute(DDL)
        with conn.cursor().copy(f"copy mart.model_dataset ({', '.join(df.columns)}) from stdin (format csv)") as cp:
            cp.write(df.to_csv(index=False, header=False))


if __name__ == "__main__":
    build()
    with connect() as conn:
        n = conn.execute("select count(*) from mart.model_dataset").fetchone()[0]
        print(f"mart.model_dataset: {n}")
