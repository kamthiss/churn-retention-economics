import pandas as pd
import pytest

from churn.db import connect
from churn.load import RAW, TABLES, load

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def conn():
    load()
    with connect() as conn:
        # иначе открытая транзакция держит блокировку и повторный load() ждёт на drop schema
        conn.autocommit = True
        yield conn


@pytest.fixture(scope="module")
def raw():
    return {name: pd.read_parquet(RAW / f"{name}.parquet") for name in TABLES}


def counts(conn):
    return {name: conn.execute(f"select count(*) from raw.{name}").fetchone()[0] for name in TABLES}


def test_row_counts(conn, raw):
    assert counts(conn) == {name: len(df) for name, df in raw.items()}


def test_order_totals(conn, raw):
    o = raw["orders"]
    d = o[o["status"] == "delivered"]
    net, cost, cancelled, back = conn.execute("""
        select sum(net) filter (where status = 'delivered'),
               sum(cost) filter (where status = 'delivered'),
               count(*) filter (where status = 'cancelled'),
               count(*) filter (where promo_code = 'BACK300')
        from raw.orders
    """).fetchone()
    assert float(net) == pytest.approx(d["net"].sum(), abs=0.005)
    assert float(cost) == pytest.approx(d["cost"].sum(), abs=0.005)
    assert cancelled == (o["status"] == "cancelled").sum()
    assert back == (o["promo_code"] == "BACK300").sum()


def test_reload(conn):
    before = counts(conn)
    load()
    assert counts(conn) == before
