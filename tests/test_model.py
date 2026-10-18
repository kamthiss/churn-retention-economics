from datetime import date

import numpy as np
import pandas as pd
import pytest

from churn.model import bootstrap, design, fit_logreg, fit_rule, split

FORBIDDEN = {"got_campaign", "purchases_back300", "churn", "customer_id", "point"}
POINTS = [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1), date(2026, 1, 1)]


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 400
    parts = []
    for t in POINTS:
        purchases = rng.integers(1, 6, n)
        gap = np.where(purchases > 1, rng.uniform(20, 60, n), np.nan)
        days = rng.integers(0, 121, n)
        parts.append(pd.DataFrame({
            "point": t,
            "customer_id": np.arange(n),
            "days_since_last": days,
            "days_since_first": days + rng.integers(0, 300, n),
            "purchases": purchases,
            "purchases_30d": rng.integers(0, 2, n),
            "purchases_90d": rng.integers(0, 3, n),
            "purchases_120d": rng.integers(0, 4, n),
            "net_total": rng.uniform(1000, 20000, n),
            "margin_total": rng.uniform(300, 6000, n),
            "avg_net": rng.uniform(1000, 4000, n),
            "median_gap": gap,
            "last_gap": np.where(purchases > 1, rng.integers(10, 80, n), np.nan),
            "recency_to_median_gap": days / gap,
            "main_category": rng.choice(["cat_food", "dog_food", "accessories"], n),
            "share_cat_food": rng.uniform(0, 1, n),
            "share_dog_food": rng.uniform(0, 1, n),
            "share_accessories": rng.uniform(0, 1, n),
            "share_site10": rng.uniform(0, 1, n),
            "purchases_back300": rng.integers(0, 2, n),
            "cancelled": rng.integers(0, 2, n),
            "channel": rng.choice(["organic", "paid_search", "marketplace_promo"], n),
            "region": rng.choice(["Москва", "Санкт-Петербург", "Регионы"], n),
            "got_campaign": rng.random(n) < 0.1,
            "churn": (rng.random(n) < days / 120).astype(int),
            "is_test": t == POINTS[-1],
            "excluded_campaign": (t == POINTS[1]) & (rng.random(n) < 0.3),
        }))
    return pd.concat(parts, ignore_index=True)


def test_split(df):
    train, test = split(df)
    assert not train["is_test"].any()
    assert not train["excluded_campaign"].any()
    assert POINTS[-1] not in set(train["point"])
    assert len(train) == (~df["is_test"] & ~df["excluded_campaign"]).sum()
    assert set(test["point"]) == {POINTS[-1]}


def test_model_features(df):
    train, _ = split(df)
    assert not FORBIDDEN & set(design(train).columns)
    search = fit_logreg(train)
    assert not FORBIDDEN & set(search.best_estimator_[0].feature_names_in_)


def test_rule_deterministic(df):
    train, _ = split(df)
    grid, best = fit_rule(train)
    grid2, best2 = fit_rule(train.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(grid, grid2)
    assert (best["k"], best["x"]) == (best2["k"], best2["x"])


def test_bootstrap_seed(df):
    _, test = split(df)
    rng = np.random.default_rng(1)
    a, b = rng.random(len(test)), rng.random(len(test))
    first = bootstrap(test["churn"], a, b, (50, 100), n=50, seed=7)
    pd.testing.assert_frame_equal(first, bootstrap(test["churn"], a, b, (50, 100), n=50, seed=7))
