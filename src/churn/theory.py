import numpy as np
import pandas as pd

from churn.params import (
    CATEGORIES, CHANNELS, CYCLE, DEATH, DEATH_MAX, HORIZON, PACE_SIGMA, PROMO_CHANNEL,
    PROMO_D1_MULT, PROMO_D_MULT, SEED, THEORY_N, THRESHOLDS, VACATION_DAYS, VACATION_P,
)


def death_probs(cat, channel):
    """Вероятности смерти (d1, d) для категории и канала."""
    d1, d = DEATH[cat]
    if channel == PROMO_CHANNEL:
        d1, d = min(d1 * PROMO_D1_MULT, DEATH_MAX), min(d * PROMO_D_MULT, DEATH_MAX)
    return d1, d


def draw_pace(rng, size=None):
    """Личный темп клиента, lognormal со средним 1."""
    return np.exp(rng.normal(-PACE_SIGMA**2 / 2, PACE_SIGMA, size))


def draw_intervals(rng, cat, pace):
    """Интервалы до следующей покупки живого клиента в целых днях."""
    cycle, shape = CYCLE[cat]
    pace = np.asarray(pace, dtype=float)
    x = rng.gamma(shape, cycle * pace / shape)
    vacation = rng.random(pace.shape) < VACATION_P
    x = x + np.where(vacation, rng.uniform(*VACATION_DAYS, pace.shape), 0.0)
    return np.maximum(np.rint(x), 1).astype(int)


def simulate(rng, cat, n=THEORY_N):
    """Интервалы живых клиентов категории и число умерших."""
    channel = rng.choice(list(CHANNELS), n, p=list(CHANNELS.values()))
    probs = np.array([death_probs(cat, c) for c in channel])
    survived_first = rng.random(n) >= probs[:, 0]
    # после первой покупки число интервалов до смерти — геометрическое
    k = survived_first * rng.geometric(probs[:, 1])
    pace = np.repeat(draw_pace(rng, n), k)
    return draw_intervals(rng, cat, pace), n


def churn_counts(intervals, deaths, n):
    """Числитель и знаменатель доли ложного оттока при пороге n."""
    long = intervals > n
    back = long & (intervals <= n + HORIZON)
    return back.sum(), long.sum() + deaths


def theory(seed=SEED):
    """Доли ложного оттока по категориям и порогам и медианы интервалов."""
    rng = np.random.default_rng(seed)
    sims = {cat: simulate(rng, cat) for cat in CATEGORIES}
    rows = []
    for n in THRESHOLDS:
        row = {"N": n}
        num = den = 0.0
        for cat, (iv, deaths) in sims.items():
            back, total = churn_counts(iv, deaths, n)
            row[cat] = back / total
            # общая доля — по смеси клиентов, а не среднее долей: у категории
            # с длинными паузами больше пауз > N, и вес у неё больше
            num += CATEGORIES[cat] * back
            den += CATEGORIES[cat] * total
        row["total"] = num / den
        rows.append(row)
    medians = {cat: float(np.median(iv)) for cat, (iv, _) in sims.items()}
    return pd.DataFrame(rows), medians


if __name__ == "__main__":
    df, medians = theory()
    print(df.round(3).to_string(index=False))
    print("медианы:", {k: round(v, 1) for k, v in medians.items()})
