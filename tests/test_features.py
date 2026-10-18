from datetime import date

import pandas as pd
import pytest

from churn.db import connect
from churn.features import FEATURES, build, features, history, query
from churn.params import CHURN_N

pytestmark = pytest.mark.db

ALLOWED = {
    "days_since_last", "days_since_first", "purchases", "purchases_30d", "purchases_90d", "purchases_120d",
    "net_total", "margin_total", "avg_net", "median_gap", "last_gap", "recency_to_median_gap",
    "main_category", "share_cat_food", "share_dog_food", "share_accessories", "share_site10",
    "purchases_back300", "cancelled", "channel", "region", "got_campaign",
}
FORBIDDEN = {"last_order_date", "lifetime_orders", "lifetime_revenue", "loyalty_tier", "status", "unsubscribed_at"}
EXCLUDED_POINTS = {date(2025, 7, 1), date(2025, 8, 1), date(2025, 9, 1)}


@pytest.fixture(scope="module")
def conn():
    build()
    with connect() as conn:
        conn.autocommit = True
        yield conn


@pytest.fixture(scope="module")
def dataset(conn):
    return query(conn, "select * from mart.model_dataset")


@pytest.fixture(scope="module")
def orders(conn):
    o = query(conn, "select customer_id, order_date from raw.orders where status = 'delivered'")
    o["order_date"] = pd.to_datetime(o["order_date"])
    return o


@pytest.mark.parametrize("t", [date(2025, 1, 1), date(2025, 10, 1), date(2026, 1, 1)])
def test_no_leak(conn, t):
    conn.execute("drop schema if exists cut cascade")
    conn.execute("create schema cut")
    conn.execute("create table cut.customers as select * from raw.customers")
    conn.execute("create table cut.orders as select * from raw.orders where order_date < %s", (t,))
    conn.execute("create table cut.campaign_contacts as select * from raw.campaign_contacts where sent_at < %s", (t,))
    try:
        full = features(*history(conn, t), t)
        cut = features(*history(conn, t, "cut"), t)
    finally:
        conn.execute("drop schema cut cascade")
    assert len(full) > 0
    pd.testing.assert_frame_equal(full, cut)


def test_columns(dataset):
    assert set(FEATURES) == ALLOWED
    assert list(dataset.columns) == ["point", "customer_id", *FEATURES, "churn", "is_test", "excluded_campaign"]
    assert not FORBIDDEN & set(dataset.columns)


@pytest.mark.parametrize("t", [date(2025, 3, 1), date(2026, 1, 1)])
def test_population_and_churn(dataset, orders, t):
    ts, n = pd.Timestamp(t), pd.Timedelta(days=CHURN_N)
    before = orders[(orders["order_date"] >= ts - n) & (orders["order_date"] < ts)]
    after = orders[(orders["order_date"] >= ts) & (orders["order_date"] < ts + n)]
    active = pd.Index(before["customer_id"].unique()).sort_values()
    exp = (~active.isin(after["customer_id"])).astype(int)
    d = dataset[dataset["point"] == t].sort_values("customer_id")
    assert (d["customer_id"].to_numpy() == active.to_numpy()).all()
    assert (d["churn"].to_numpy() == exp).all()


def test_points(dataset):
    points = sorted(dataset["point"].unique())
    assert len(points) == 16
    assert set(dataset.loc[dataset["is_test"], "point"]) == {date(2026, 1, 1)}


def test_excluded_campaign(conn, dataset):
    participants = set(query(conn, "select customer_id from raw.campaign_contacts")["customer_id"])
    x = dataset[dataset["excluded_campaign"]]
    assert set(x["point"]) == EXCLUDED_POINTS
    assert set(x["customer_id"]) <= participants
    on = dataset[dataset["point"].isin(EXCLUDED_POINTS)]
    assert (on["excluded_campaign"] == on["customer_id"].isin(participants)).all()
