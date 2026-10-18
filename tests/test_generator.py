import numpy as np
import pandas as pd
import pytest

from churn.generate import generate, write
from churn.params import (
    CAMPAIGN_DATE, CAMPAIGN_MIN_ORDER, CAMPAIGN_PROMO, CAMPAIGN_VALID_DAYS, CAMPAIGN_WINDOW,
    CANCEL_P, CHANNELS, END, HORIZON, LOYALTY, N_CUSTOMERS, REGIONS, SEED, UPLIFT_R, ACTIVE_FROM,
)
from churn.theory import theory

CAMP = pd.Timestamp(CAMPAIGN_DATE)


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    write(generate(), out)
    return {p.stem: pd.read_parquet(p) for p in out.glob("*/*.parquet")}


@pytest.fixture(scope="session")
def delivered(data):
    o = data["orders"]
    return o[o["status"] == "delivered"].sort_values(["customer_id", "order_date", "order_id"])


def test_customers_and_shares(data):
    c = data["customers"]
    assert len(c) == N_CUSTOMERS
    for col, probs in (("channel", CHANNELS), ("region", REGIONS)):
        share = c[col].value_counts(normalize=True)
        for k, p in probs.items():
            assert share[k] == pytest.approx(p, abs=0.01), k


def test_order_dates(data, delivered):
    c = data["customers"].set_index("customer_id")
    first = delivered.groupby("customer_id")["order_date"].min()
    assert (first == c.loc[first.index, "signup_date"]).all()
    o = data["orders"]
    signup = o["customer_id"].map(c["signup_date"])
    assert (o["order_date"] >= signup).all()
    assert (o["order_date"] <= pd.Timestamp(END)).all()


def test_cancel_share(data, delivered):
    # перед первым заказом отмен нет, поэтому не 0.04 / 1.04
    cancels = CANCEL_P * (len(delivered) - len(data["customers"]))
    share = (data["orders"]["status"] == "cancelled").mean()
    assert share == pytest.approx(cancels / (len(delivered) + cancels), abs=0.003)


def test_dead_do_not_buy(data, delivered):
    t = data["customers_truth"].set_index("customer_id")
    death = delivered["customer_id"].map(t["last_purchase_before_death"])
    assert not (delivered["order_date"] > death).any()
    dead = t[t["died"]]
    assert dead["last_purchase_before_death"].notna().all()
    assert t.loc[~t["died"], "last_purchase_before_death"].isna().all()


def test_snapshot_matches_orders(data, delivered):
    c = data["customers"].set_index("customer_id")
    g = delivered.groupby("customer_id")
    last = g["order_date"].max()
    revenue = g["net"].sum()
    assert (c["last_order_date"] == last).all()
    assert (c["lifetime_orders"] == g.size()).all()
    assert np.allclose(c["lifetime_revenue"], revenue)
    tier = pd.Series("bronze", index=revenue.index)
    for name, threshold in reversed(LOYALTY):
        tier[revenue >= threshold] = name
    assert (c["loyalty_tier"] == tier).all()
    status = np.where(last >= pd.Timestamp(ACTIVE_FROM), "active", "dormant")
    assert (c["status"] == status).all()


def test_campaign_groups_and_promo(data, delivered):
    cc = data["campaign_contacts"]
    assert (cc["group_name"] == "treatment").mean() == pytest.approx(0.7, abs=0.015)
    back = data["orders"][data["orders"]["promo_code"] == CAMPAIGN_PROMO]
    assert len(back) > 0
    groups = cc.set_index("customer_id")["group_name"]
    assert (back["customer_id"].map(groups) == "treatment").all()
    assert (back["order_date"] >= CAMP).all()
    assert (back["order_date"] < CAMP + pd.Timedelta(days=CAMPAIGN_VALID_DAYS)).all()
    assert (back["gross"] >= CAMPAIGN_MIN_ORDER).all()


def test_uplift_by_segment(data, delivered):
    t = data["customers_truth"]
    cc = data["campaign_contacts"].merge(t[["customer_id", "uplift_segment"]], on="customer_id")
    window = delivered[(delivered["order_date"] >= CAMP)
                       & (delivered["order_date"] < CAMP + pd.Timedelta(days=CAMPAIGN_WINDOW))]
    cc["bought"] = cc["customer_id"].isin(window["customer_id"])
    for seg, r in UPLIFT_R.items():
        s = cc[cc["uplift_segment"] == seg]
        pt = s.loc[s["group_name"] == "treatment", "bought"]
        pc = s.loc[s["group_name"] == "control", "bought"]
        se = np.sqrt(pt.var(ddof=0) / len(pt) + pc.var(ddof=0) / len(pc))
        observed = pt.mean() - pc.mean()
        expected = r * (1 - pc.mean())
        assert abs(observed - expected) <= 3 * se, seg


def test_interval_medians(data, delivered):
    t = data["customers_truth"].set_index("customer_id")
    d = delivered.copy()
    d["inserted"] = (d["promo_code"] == CAMPAIGN_PROMO) & d["customer_id"].map(t["campaign_converted"])
    g = d.groupby("customer_id")
    d["prev"] = g["order_date"].shift()
    d["gap"] = (d["order_date"] - d["prev"]).dt.days
    # без отсечки длинные паузы в конце данных ещё не закрыты, и медиана
    # accessories занижена (47 против 50)
    closed = d["prev"] <= pd.Timestamp(END) - pd.Timedelta(days=HORIZON)
    keep = closed & ~d["inserted"] & ~g["inserted"].shift(fill_value=False)
    d["main"] = d["customer_id"].map(t["main_category"])
    observed = d[keep].groupby("main")["gap"].median()
    for cat, med in theory()[1].items():
        assert observed[cat] == pytest.approx(med, rel=0.05), cat


def test_deterministic():
    a, b = generate(2000, SEED), generate(2000, SEED)
    for folder in a:
        for name in a[folder]:
            pd.testing.assert_frame_equal(a[folder][name], b[folder][name])
