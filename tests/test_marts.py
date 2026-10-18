from collections import Counter

import numpy as np
import pandas as pd
import pytest

from churn.db import connect
from churn.marts import build

pytestmark = pytest.mark.db

END = pd.Timestamp("2026-06-30")
HORIZON = 365


@pytest.fixture(scope="module")
def conn():
    build()
    with connect() as conn:
        conn.autocommit = True
        yield conn


@pytest.fixture(scope="module")
def orders(conn):
    o = query(conn, "select * from raw.orders where status = 'delivered'")
    o["order_date"] = pd.to_datetime(o["order_date"])
    return o.sort_values(["customer_id", "order_date", "order_id"], ignore_index=True)


def query(conn, sql, params=None):
    cur = conn.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def segments(cats):
    """Основная категория после каждой покупки: самая частая, при равенстве — купленная последней."""
    counts, last, out = Counter(), {}, []
    for i, c in enumerate(cats):
        counts[c] += 1
        last[c] = i
        top = max(counts.values())
        out.append(max((k for k in counts if counts[k] == top), key=last.get))
    return out


def test_purchases(conn):
    n, margin = conn.execute("select count(*), sum(margin) from mart.purchases").fetchone()
    raw_n, raw_margin = conn.execute(
        "select count(*), sum(net - cost) from raw.orders where status = 'delivered'").fetchone()
    assert n == raw_n == 257_060
    assert margin == raw_margin


def test_pauses_one_per_purchase(conn):
    n, keys, joined = conn.execute("""
        select count(*), count(distinct (p.customer_id, p.purchase_n)), count(s.order_id)
        from mart.pauses p
        left join mart.purchases s using (customer_id, purchase_n)
    """).fetchone()
    assert n == keys == joined == conn.execute("select count(*) from mart.purchases").fetchone()[0]


def test_segment_no_future(conn, orders):
    ids = np.random.default_rng(0).choice(orders["customer_id"].unique(), 200, replace=False)
    p = query(conn, "select customer_id, order_date, segment from mart.pauses where customer_id = any(%s)",
              (list(map(int, ids)),))
    p["order_date"] = pd.to_datetime(p["order_date"])
    o = orders[orders["customer_id"].isin(ids)]
    for cid, d, seg in p.itertuples(index=False):
        cut = o[(o["customer_id"] == cid) & (o["order_date"] <= d)]
        assert segments(list(cut["category"]))[-1] == seg, (cid, d)


def test_false_churn(conn, orders):
    o = orders.copy()
    g = o.groupby("customer_id")
    o["gap"] = (g["order_date"].shift(-1) - o["order_date"]).dt.days
    o["pause"] = np.where(g.cumcount() == 0, "first", "repeat")
    o["segment"] = g["category"].transform(lambda s: segments(list(s)))
    rows = []
    for n in (60, 90, 120, 180):
        x = o[(o["order_date"] + pd.Timedelta(days=n + HORIZON) <= END) & ~(o["gap"] <= n)]
        x = x.assign(back=x["gap"] <= n + HORIZON)
        for seg in ("cat_food", "dog_food", "accessories", "all"):
            for pause in ("first", "repeat", "all"):
                y = x[((x["segment"] == seg) | (seg == "all")) & ((x["pause"] == pause) | (pause == "all"))]
                rows.append((n, seg, pause, len(y), int(y["back"].sum())))
    exp = pd.DataFrame(rows, columns=["n", "segment", "pause", "pauses_over_n", "returned"])
    mart = query(conn, "select * from mart.false_churn")
    df = exp.merge(mart, on=["n", "segment", "pause"], suffixes=("", "_mart"), validate="1:1")
    assert len(df) == len(mart) == 48
    assert (df["pauses_over_n"] == df["pauses_over_n_mart"]).all()
    assert (df["returned"] == df["returned_mart"]).all()
    assert df["share"].astype(float).to_numpy() == pytest.approx(df["returned"] / df["pauses_over_n"], abs=5e-5)


def test_cohort_month0(conn):
    shares = conn.execute("select distinct retention from mart.cohort_retention where k = 0").fetchall()
    assert shares == [(1,)]
