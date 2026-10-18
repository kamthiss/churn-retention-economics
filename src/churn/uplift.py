from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from churn.features import features, history, query
from churn.model import design, fit_boosting, fit_logreg, fit_rule, rule_score
from churn.params import CAMPAIGN_DATE, CAMPAIGN_DISCOUNT, CAMPAIGN_PROMO, REPEAT_SEGMENT, SEED, TEST_POINT

SMS = 5
N_BOOT = 2000
HORIZONS = (90, 180, 270)
BOUGHT = (90, 120)
SEGMENTS = ("one_time", "repeat_cat", "repeat_dog", "repeat_acc")
FRACS = tuple(np.round(np.arange(1, 11) / 10, 1))


def outcomes(orders, t=CAMPAIGN_DATE):
    """Исходы по доставленным заказам с датой от t: покупки в окнах, маржа за горизонты, заказы с промокодом."""
    o = orders[orders["status"] == "delivered"]
    o = o.assign(days=(pd.to_datetime(o["order_date"]) - pd.Timestamp(t)).dt.days)
    o = o[o["days"] >= 0].sort_values(["customer_id", "order_date", "order_id"])
    cid = o["customer_id"]
    df = pd.DataFrame(index=pd.Index(cid.unique(), name="customer_id"))
    for d in BOUGHT:
        df[f"bought_{d}"] = (o["days"] < d).groupby(cid).any()
    for h in HORIZONS:
        df[f"margin_{h}"] = o["margin"].where(o["days"] < h, 0.0).groupby(cid).sum()
    df["back300"] = (o["promo_code"] == CAMPAIGN_PROMO).groupby(cid).sum()
    df["redeemed"] = df["back300"] > 0
    return df.reset_index()


def participants(conn, schema="raw", t=CAMPAIGN_DATE):
    """Участники кампании: группа, признаки и сегмент на t, исходы после t."""
    contacts = query(conn, f"""
        select customer_id, group_name = 'treatment' as treat
        from {schema}.campaign_contacts order by customer_id""")
    df = contacts.merge(features(*history(conn, t, schema), t, n=None), on="customer_id", how="left")
    seg = np.where(df["purchases"] == 1, "one_time", df["main_category"].map(REPEAT_SEGMENT))
    df["segment"] = pd.Categorical(seg, categories=SEGMENTS)
    after = query(conn, f"""
        select customer_id, order_id, order_date, status, (net - cost)::float8 as margin, promo_code
        from {schema}.orders where order_date >= %s""", (t,))
    out = outcomes(after, t)
    cols = out.columns.drop("customer_id")
    df = df.merge(out, on="customer_id", how="left")
    df[cols] = df[cols].fillna(0).astype(out[cols].dtypes.to_dict())
    return df


def fit_models(train):
    """Правило, логрегрессия и бустинг, обученные как в model.py."""
    _, best = fit_rule(train)
    return {"k": best["k"], "x": best["x"], "логрегрессия": fit_logreg(train), "бустинг": fit_boosting(train)}


def risk(models, df):
    """Оценки риска правила и моделей."""
    X = design(df)
    return {
        "правило": rule_score(df, models["k"], models["x"]),
        "логрегрессия": models["логрегрессия"].predict_proba(X)[:, 1],
        "бустинг": models["бустинг"].predict_proba(X)[:, 1],
    }


def boot_weights(treat, n=N_BOOT, seed=SEED):
    """Сколько раз каждый клиент попал в каждый повтор бутстрепа; ресемплинг внутри treatment и control."""
    treat = np.asarray(treat, bool)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(treat), np.flatnonzero(~treat)]
    w = np.zeros((n, len(treat)), dtype=np.int32)
    for r in range(n):
        for idx in groups:
            w[r] += np.bincount(idx[rng.integers(0, len(idx), len(idx))], minlength=len(treat))
    return w


def profit(margin, treat, w=None):
    """Прибыль на получателя: средняя маржа treatment − control − SMS; с весами — по каждому повтору."""
    margin, treat = np.asarray(margin, float), np.asarray(treat, bool)
    w = np.ones(len(margin)) if w is None else w
    wt, wc = w * treat, w * ~treat
    return (wt @ margin) / wt.sum(-1) - (wc @ margin) / wc.sum(-1) - SMS


def ci(boot):
    """95% перцентильный интервал."""
    return np.quantile(boot, 0.025), np.quantile(boot, 0.975)


def effect(df, margin="margin_180", seed=SEED):
    """Эффект кампании в группе клиентов: доли купивших, маржа, прибыль на получателя."""
    t, m = df["treat"].to_numpy(bool), df[margin].to_numpy(float)
    nt, nc = t.sum(), (~t).sum()
    row = {"n_t": nt, "n_c": nc}
    for d in BOUGHT:
        y = df[f"bought_{d}"].to_numpy(float)
        a, b = y[t].mean(), y[~t].mean()
        se = np.sqrt(a * (1 - a) / nt + b * (1 - b) / nc)
        row |= {f"b{d}_t": a, f"b{d}_c": b, f"b{d}_diff": a - b,
                f"b{d}_lo": a - b - 1.96 * se, f"b{d}_hi": a - b + 1.96 * se}
    p = profit(m, t)
    lo, hi = ci(profit(m, t, boot_weights(t, seed=seed)))
    row |= {"margin_t": m[t].mean(), "margin_c": m[~t].mean(), "profit": p, "profit_lo": lo, "profit_hi": hi,
            "redeemed_t": df.loc[t, "redeemed"].mean(), "total": p * nt}
    return row


def effects(df, by=None, margin="margin_180"):
    """Эффект по всей кампании и по группам из колонки by."""
    rows = {"все": effect(df, margin)}
    if by is not None:
        rows |= {g: effect(d, margin) for g, d in df.groupby(by, observed=True)}
    return pd.DataFrame(rows).T


def terciles(score):
    """Терцили риска; равные оценки разводятся по порядку строк."""
    return pd.qcut(pd.Series(score).rank(method="first"), 3, labels=["низкий", "средний", "высокий"]).array


def top(score, size):
    """Маска size клиентов с наибольшим риском."""
    sel = np.zeros(len(score), bool)
    sel[np.argsort(-np.asarray(score), kind="stable")[:size]] = True
    return sel


def strategies(df, scores, frac=0.5, w=None):
    """Прибыль стратегий отбора: случайные, топ по риску, сегменты с окупаемой скидкой."""
    t, m = df["treat"].to_numpy(bool), df["margin_180"].to_numpy(float)
    w = boot_weights(t) if w is None else w
    size = int(len(df) * frac)
    seg = effects(df, "segment").drop("все")
    good = seg.index[seg["profit_lo"] > 0]
    sels = {"случайные": np.ones(len(df), bool)} | {f"топ по риску: {k}": top(s, size) for k, s in scores.items()}
    sels[f"сегменты: {', '.join(good) or '—'}"] = df["segment"].isin(good).to_numpy()
    rows = {}
    for name, sel in sels.items():
        n = size if name == "случайные" else int(sel.sum())
        p, (lo, hi) = profit(m[sel], t[sel]), ci(profit(m[sel], t[sel], w[:, sel]))
        rows[name] = {"размер": n, "на получателя": p, "lo": lo, "hi": hi, "прибыль": p * n,
                      "прибыль lo": lo * n, "прибыль hi": hi * n}
    return pd.DataFrame(rows).T


def curve(df, scores, fracs=FRACS):
    """Прибыль стратегий по риску и случайной в зависимости от доли охвата."""
    t, m = df["treat"].to_numpy(bool), df["margin_180"].to_numpy(float)
    rows = []
    for f in fracs:
        size = int(len(df) * f)
        row = {"охват": f, "размер": size, "случайные": profit(m, t) * size}
        for k, s in scores.items():
            sel = top(s, size)
            row[k] = profit(m[sel], t[sel]) * size
        rows.append(row)
    return pd.DataFrame(rows).set_index("охват")


def compare(df, a, b, frac=0.5, w=None):
    """Разница прибыли топ-frac по оценке b и по оценке a: оценка и 95% бутстреп-интервал."""
    t, m = df["treat"].to_numpy(bool), df["margin_180"].to_numpy(float)
    w = boot_weights(t) if w is None else w
    size = int(len(df) * frac)
    sa, sb = top(a, size), top(b, size)
    point = (profit(m[sb], t[sb]) - profit(m[sa], t[sa])) * size
    boot = (profit(m[sb], t[sb], w[:, sb]) - profit(m[sa], t[sa], w[:, sa])) * size
    return pd.Series({"размер": size, "разница": point, "lo": ci(boot)[0], "hi": ci(boot)[1]})


def with_discount(df, discount):
    """Маржа за 180 дней, если бы промокод стоил discount: меняется только списанная скидка."""
    return df["margin_180"] + (CAMPAIGN_DISCOUNT - discount) * df["back300"]


def accessories_check(conn, test, scores, n=180, t=TEST_POINT):
    """PR-AUC на точке проверки у клиентов с main_category = accessories при пороге 120 и n дней."""
    later = query(conn, """
        select distinct customer_id from raw.orders
        where status = 'delivered' and order_date >= %s and order_date < %s""",
                  (t, t + timedelta(days=n)))["customer_id"]
    acc = (test["main_category"] == "accessories").to_numpy()
    y = {"120": test["churn"].to_numpy()[acc],
         str(n): (~test["customer_id"].isin(later)).astype(int).to_numpy()[acc]}
    rows = {k: {"доля churn": v.mean(), **{s: average_precision_score(v, sc[acc]) for s, sc in scores.items()}}
            for k, v in y.items()}
    return pd.DataFrame(rows).T.rename_axis("порог")
