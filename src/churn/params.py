from datetime import date

SEED = 20261018

N_CUSTOMERS = 50_000
START = date(2024, 1, 1)
SIGNUP_END = date(2026, 3, 31)
END = date(2026, 6, 30)
SIGNUP_DENSITY_RATIO = 2.0

CHANNELS = {"organic": 0.40, "paid_search": 0.35, "marketplace_promo": 0.25}
REGIONS = {"Москва": 0.30, "Санкт-Петербург": 0.15, "Регионы": 0.55}
CATEGORIES = {"cat_food": 0.45, "dog_food": 0.35, "accessories": 0.20}

# (d1, d): вероятность смерти после первой и после последующих покупок
DEATH = {"cat_food": (0.35, 0.08), "dog_food": (0.35, 0.08), "accessories": (0.45, 0.12)}
PROMO_CHANNEL = "marketplace_promo"
PROMO_D1_MULT = 1.4
PROMO_D_MULT = 1.2
DEATH_MAX = 0.9

# (cycle, shape) гамма-интервала
CYCLE = {"cat_food": (30, 6), "dog_food": (40, 6), "accessories": (60, 2)}
PACE_SIGMA = 0.15
VACATION_P = 0.01
VACATION_DAYS = (30, 60)

MAIN_CATEGORY_P = 0.8
# (медиана, sigma) lognormal суммы gross
GROSS = {"cat_food": (1800, 0.35), "dog_food": (3200, 0.30), "accessories": (1400, 0.55)}
MARGIN = {"cat_food": 0.30, "dog_food": 0.25, "accessories": 0.45}
SITE10_P = 0.15
SITE10_RATE = 0.10
CANCEL_P = 0.04
CANCEL_DAYS = (1, 5)

CAMPAIGN_ID = 1
CAMPAIGN_NAME = "Возврат 300"
CAMPAIGN_DATE = date(2025, 10, 1)
CAMPAIGN_PROMO = "BACK300"
CAMPAIGN_DISCOUNT = 300
CAMPAIGN_MIN_ORDER = 1500
CAMPAIGN_VALID_DAYS = 30
CAMPAIGN_WINDOW = 90
ELIGIBLE_GAP = (45, 150)
TREATMENT_P = 0.7
NATURAL_REDEEM_P = 0.85
UPLIFT_R = {"one_time": 0.03, "repeat_cat": 0.05, "repeat_dog": 0.20, "repeat_acc": 0.08}
REPEAT_SEGMENT = {"cat_food": "repeat_cat", "dog_food": "repeat_dog", "accessories": "repeat_acc"}

LOYALTY = (("gold", 30000), ("silver", 10000))
ACTIVE_FROM = date(2026, 4, 1)
UNSUB_DEAD_P = 0.4
UNSUB_DEAD_DAYS = (20, 120)
UNSUB_ALIVE_P = 0.03

THEORY_N = 100_000
THRESHOLDS = (60, 90, 120, 180)
HORIZON = 365
