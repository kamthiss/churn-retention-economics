import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from churn.params import (
    ACTIVE_FROM, CAMPAIGN_DATE, CAMPAIGN_DISCOUNT, CAMPAIGN_ID, CAMPAIGN_MIN_ORDER,
    CAMPAIGN_NAME, CAMPAIGN_PROMO, CAMPAIGN_VALID_DAYS, CAMPAIGN_WINDOW, CANCEL_DAYS,
    CANCEL_P, CATEGORIES, CHANNELS, ELIGIBLE_GAP, END, GROSS, LOYALTY, MAIN_CATEGORY_P,
    MARGIN, N_CUSTOMERS, NATURAL_REDEEM_P, REGIONS, REPEAT_SEGMENT, SEED,
    SIGNUP_DENSITY_RATIO, SIGNUP_END, SITE10_P, SITE10_RATE, START, TREATMENT_P,
    UNSUB_ALIVE_P, UNSUB_DEAD_DAYS, UNSUB_DEAD_P, UPLIFT_R,
)
from churn.theory import death_probs, draw_intervals, draw_pace

CAMP = (CAMPAIGN_DATE - START).days
END_DAY = (END - START).days
ACTIVE_DAY = (ACTIVE_FROM - START).days


def to_date(days):
    return pd.Timestamp(START) + pd.to_timedelta(days, unit="D")


def draw_signup(rng, n):
    span = (SIGNUP_END - START).days + 1
    # обратная функция CDF линейной плотности 1 + (ratio - 1) * x на [0, 1]
    a = SIGNUP_DENSITY_RATIO - 1
    x = (np.sqrt(1 + a * (2 + a) * rng.random(n)) - 1) / a
    return np.minimum((x * span).astype(int), span - 1)


def pick(rng, probs, n):
    return rng.choice(list(probs), n, p=list(probs.values()))


def segment(cats):
    if len(cats) == 1:
        return "one_time"
    counts = Counter(cats)
    top = max(counts.values())
    last = next(c for c in reversed(cats) if counts[c] == top)
    return REPEAT_SEGMENT[last]


def purchases(rng, signup, main, channel, pace):
    """История покупок одного клиента с учётом кампании."""
    d1, d = death_probs(main, channel)
    others = [c for c in CATEGORIES if c != main]
    days, cats, inserted, natural = [], [], [], []
    nxt, ins, nat = signup, False, False
    group = seg = None
    checked = signup >= CAMP
    while True:
        days.append(nxt)
        cats.append(main if rng.random() < MAIN_CATEGORY_P else others[rng.integers(2)])
        inserted.append(ins)
        natural.append(nat)
        ins = nat = False
        died = rng.random() < (d1 if len(days) == 1 else d)
        nxt = None if died else days[-1] + int(draw_intervals(rng, main, pace))
        if not checked and (nxt is None or nxt >= CAMP):
            checked = True
            if ELIGIBLE_GAP[0] <= CAMP - days[-1] <= ELIGIBLE_GAP[1]:
                group = "treatment" if rng.random() < TREATMENT_P else "control"
                seg = segment(cats)
            if group == "treatment":
                if nxt is not None and nxt < CAMP + CAMPAIGN_VALID_DAYS:
                    nat = True
                elif (nxt is None or nxt >= CAMP + CAMPAIGN_WINDOW) and rng.random() < UPLIFT_R[seg]:
                    nxt, ins = CAMP + int(rng.integers(CAMPAIGN_VALID_DAYS)), True
        if nxt is None or nxt > END_DAY:
            break
    return days, cats, inserted, natural, group, seg, nxt is None


def draw_gross(rng, cats):
    med = np.array([GROSS[c][0] for c in cats])
    sigma = np.array([GROSS[c][1] for c in cats])
    return np.rint(med * np.exp(sigma * rng.standard_normal(len(cats))))


def build_orders(rng, p):
    gross = draw_gross(rng, p["category"])
    redraw = p["inserted"].to_numpy().copy()
    while redraw.any():
        gross[redraw] = draw_gross(rng, p.loc[redraw, "category"])
        redraw &= gross < CAMPAIGN_MIN_ORDER
    p["gross"] = gross
    back = p["inserted"] | (p["natural"] & (p["gross"] >= CAMPAIGN_MIN_ORDER)
                            & (rng.random(len(p)) < NATURAL_REDEEM_P))
    site = ~back & (rng.random(len(p)) < SITE10_P)
    p["promo_code"] = np.where(back, CAMPAIGN_PROMO, np.where(site, "SITE10", None))
    p["discount"] = np.where(back, CAMPAIGN_DISCOUNT, np.where(site, np.rint(p["gross"] * SITE10_RATE), 0.0))
    p["status"] = "delivered"

    cancel = p[(p["seq"] > 0) & (rng.random(len(p)) < CANCEL_P)].copy()
    offset = rng.integers(CANCEL_DAYS[0], CANCEL_DAYS[1] + 1, len(cancel))
    cancel["day"] -= np.minimum(offset, cancel["day"] - cancel["signup"])
    cancel["status"] = "cancelled"
    cancel["promo_code"] = None
    cancel["discount"] = 0.0

    o = pd.concat([p, cancel], ignore_index=True)
    o = o.sort_values(["day", "customer_id"], kind="stable", ignore_index=True)
    o["order_id"] = np.arange(1, len(o) + 1)
    o["order_date"] = to_date(o["day"])
    o["net"] = o["gross"] - o["discount"]
    o["cost"] = (o["gross"] * (1 - o["category"].map(MARGIN))).round(2)
    return o


def build_customers(rng, c, o, died, last_day):
    d = o[o["status"] == "delivered"].groupby("customer_id")
    c["last_order_date"] = to_date(d["day"].max().reindex(c["customer_id"]).to_numpy())
    c["lifetime_orders"] = d.size().reindex(c["customer_id"]).to_numpy()
    c["lifetime_revenue"] = d["net"].sum().reindex(c["customer_id"]).to_numpy()
    c["loyalty_tier"] = "bronze"
    for tier, threshold in reversed(LOYALTY):
        c.loc[c["lifetime_revenue"] >= threshold, "loyalty_tier"] = tier
    c["status"] = np.where(last_day >= ACTIVE_DAY, "active", "dormant")

    n = len(c)
    dead_day = last_day + rng.integers(UNSUB_DEAD_DAYS[0], UNSUB_DEAD_DAYS[1] + 1, n)
    dead_unsub = died & (rng.random(n) < UNSUB_DEAD_P) & (dead_day <= END_DAY)
    alive_day = c["signup"] + (rng.random(n) * (END_DAY - c["signup"] + 1)).astype(int)
    alive_unsub = ~died & (rng.random(n) < UNSUB_ALIVE_P)
    unsub = np.where(dead_unsub, dead_day, np.where(alive_unsub, alive_day, np.nan))
    c["unsubscribed_at"] = to_date(unsub)
    c["signup_date"] = to_date(c["signup"])
    return c[["customer_id", "signup_date", "channel", "region", "last_order_date", "lifetime_orders",
              "lifetime_revenue", "loyalty_tier", "status", "unsubscribed_at"]]


def generate(n=N_CUSTOMERS, seed=SEED):
    """Синтетические таблицы raw и truth."""
    rng = np.random.default_rng(seed)
    c = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "signup": draw_signup(rng, n),
        "channel": pick(rng, CHANNELS, n),
        "region": pick(rng, REGIONS, n),
        "main_category": pick(rng, CATEGORIES, n),
        "pace": draw_pace(rng, n),
    })

    cols = {k: [] for k in ("customer_id", "signup", "seq", "day", "category", "inserted", "natural")}
    truth = []
    for cid, signup, main, channel, pace in c[["customer_id", "signup", "main_category", "channel", "pace"]].itertuples(index=False):
        days, cats, inserted, natural, group, seg, died = purchases(rng, signup, main, channel, pace)
        k = len(days)
        cols["customer_id"] += [cid] * k
        cols["signup"] += [signup] * k
        cols["seq"] += range(k)
        cols["day"] += days
        cols["category"] += cats
        cols["inserted"] += inserted
        cols["natural"] += natural
        truth.append((died, days[-1], group, seg, any(inserted)))
    p = pd.DataFrame(cols)
    died, last_day, group, seg, converted = map(np.array, zip(*truth))

    o = build_orders(rng, p)
    customers = build_customers(rng, c.copy(), o, died, last_day)

    redeemed = o.loc[o["natural"] & (o["promo_code"] == CAMPAIGN_PROMO), "customer_id"]
    customers_truth = pd.DataFrame({
        "customer_id": c["customer_id"],
        "main_category": c["main_category"],
        "pace": c["pace"],
        "died": died,
        "last_purchase_before_death": to_date(np.where(died, last_day, np.nan)),
        "uplift_segment": seg,
        "campaign_group": group,
        "campaign_converted": converted,
        "campaign_redeemed_natural": c["customer_id"].isin(redeemed).to_numpy(),
    })

    contacted = customers_truth[customers_truth["campaign_group"].notna()]
    campaign_contacts = pd.DataFrame({
        "campaign_id": CAMPAIGN_ID,
        "customer_id": contacted["customer_id"].to_numpy(),
        "sent_at": pd.Timestamp(CAMPAIGN_DATE),
        "group_name": contacted["campaign_group"].to_numpy(),
    })
    campaigns = pd.DataFrame([{
        "campaign_id": CAMPAIGN_ID,
        "name": CAMPAIGN_NAME,
        "sent_at": pd.Timestamp(CAMPAIGN_DATE),
        "promo_code": CAMPAIGN_PROMO,
        "discount": CAMPAIGN_DISCOUNT,
        "min_order": CAMPAIGN_MIN_ORDER,
        "valid_days": CAMPAIGN_VALID_DAYS,
    }])
    orders = o[["order_id", "customer_id", "order_date", "status", "category", "gross", "discount", "net",
                "cost", "promo_code"]]
    return {
        "raw": {"customers": customers, "orders": orders, "campaigns": campaigns,
                "campaign_contacts": campaign_contacts},
        "truth": {"customers_truth": customers_truth},
    }


def write(tables, out):
    """Сохраняет таблицы в out/raw и out/truth."""
    for folder, dfs in tables.items():
        path = Path(out) / folder
        path.mkdir(parents=True, exist_ok=True)
        for name, df in dfs.items():
            df.to_parquet(path / f"{name}.parquet", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_CUSTOMERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    tables = generate(args.n, args.seed)
    write(tables, args.out)
    o = tables["raw"]["orders"]
    print(f"customers: {len(tables['raw']['customers'])}")
    print(o["status"].value_counts().to_string())
