from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from churn.db import connect
from churn.features import FEATURES
from churn.model import load, split
from churn.uplift import (
    boot_weights, compare, effect, fit_models, group_size, outcomes, participants, profit, risk, top,
)

T = date(2025, 10, 1)


def order(cid, days, margin=100.0, status="delivered", promo=None):
    return {"customer_id": cid, "order_id": cid * 100 + days, "order_date": T + timedelta(days=days),
            "status": status, "margin": margin, "promo_code": promo}


def test_outcomes_windows():
    o = pd.DataFrame([
        order(1, -1, 1000), order(1, -30, 1000, promo="BACK300"),
        order(2, 0, 10, promo="BACK300"), order(2, 89, 20), order(2, 179, 40), order(2, 269, 80), order(2, 270, 160),
        order(3, 90, 10), order(3, 119, 20), order(3, 180, 40),
        order(4, 120, 10), order(4, 5, 999, status="cancelled", promo="BACK300"),
    ])
    out = outcomes(o, T).set_index("customer_id")
    assert 1 not in out.index
    assert out.loc[2, ["bought_90", "bought_120", "redeemed"]].tolist() == [True, True, True]
    assert out.loc[2, ["margin_90", "margin_180", "margin_270"]].tolist() == [30, 70, 150]
    assert out.loc[3, ["bought_90", "bought_120"]].tolist() == [False, True]
    assert out.loc[3, ["margin_90", "margin_180", "margin_270"]].tolist() == [0, 30, 70]
    assert out.loc[4, ["bought_90", "bought_120", "redeemed"]].tolist() == [False, False, False]


@pytest.mark.db
def test_participants_no_leak():
    train, _ = split(load())
    models = fit_models(train[train["point"].isin(sorted(train["point"].unique())[-4:])].sample(3000, random_state=0))
    with connect() as conn:
        conn.autocommit = True
        conn.execute("drop schema if exists cut cascade")
        conn.execute("create schema cut")
        conn.execute("create table cut.customers as select * from raw.customers")
        conn.execute("create table cut.orders as select * from raw.orders where order_date < %s", (T,))
        conn.execute("create table cut.campaign_contacts as select * from raw.campaign_contacts")
        try:
            full, cut = participants(conn), participants(conn, "cut")
        finally:
            conn.execute("drop schema cut cascade")
    cols = ["customer_id", "treat", *FEATURES, "segment"]
    assert len(full) == 6946
    pd.testing.assert_frame_equal(full[cols], cut[cols])
    for name, s in risk(models, full).items():
        np.testing.assert_array_equal(s, risk(models, cut)[name], err_msg=name)
    assert not cut["bought_120"].any() and (cut["margin_180"] == 0).all()


def test_strategy_recovers_effect():
    rng = np.random.default_rng(0)
    n = 40_000
    score = rng.random(n)
    treat = rng.random(n) < 0.7
    margin = rng.normal(500, 100, n) + 200 * treat * (score > 0.5)
    size = n // 2
    sel = top(score, size)
    assert sel.sum() == size and (score[sel] > 0.5).all()
    assert profit(margin[sel], treat[sel]) == pytest.approx(195, abs=5)
    assert profit(margin, treat) == pytest.approx(95, abs=5)
    df = pd.DataFrame({"treat": treat, "margin_180": margin})
    res = compare(df, -score, score, w=boot_weights(treat, n=200))
    assert res["разница"] == pytest.approx(200 * size, rel=0.05)
    assert res["lo"] > 0


def test_bootstrap_seed():
    rng = np.random.default_rng(1)
    treat = rng.random(500) < 0.7
    w = boot_weights(treat, n=100, seed=7)
    np.testing.assert_array_equal(w, boot_weights(treat, n=100, seed=7))
    assert not np.array_equal(w, boot_weights(treat, n=100, seed=8))
    assert (w[:, treat].sum(1) == treat.sum()).all() and (w[:, ~treat].sum(1) == (~treat).sum()).all()
    df = pd.DataFrame({"treat": treat, "margin_180": rng.normal(500, 100, 500),
                       "bought_90": rng.random(500) < 0.3, "bought_120": rng.random(500) < 0.3,
                       "redeemed": treat & (rng.random(500) < 0.2)})
    assert effect(df, seed=3) == effect(df, seed=3)


def test_group_size():
    assert group_size(1000, 100) == 1570
