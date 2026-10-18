import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from churn.db import connect
from churn.features import FEATURES, query
from churn.params import SEED

# кампания прошла после всех точек обучения, см. docs/churn_definition.md
EXCLUDED = ("got_campaign", "purchases_back300")
CATEGORICAL = ["main_category", "channel", "region"]
NULLABLE = ["median_gap", "last_gap", "recency_to_median_gap"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL and f not in EXCLUDED] + ["one_purchase"]

K_GRID = (1.5, 2, 2.5, 3, 4)
X_GRID = (30, 45, 60, 90)
C_GRID = np.logspace(-3, 2, 11)
HGB_GRID = {"learning_rate": (0.05, 0.1), "max_leaf_nodes": (15, 31), "max_iter": (200,)}
MIN_GAIN = 0.02
TOP_K = (1000, 2000, 4000)
N_BOOT = 1000


def load():
    """mart.model_dataset целиком."""
    with connect() as conn:
        return query(conn, "select * from mart.model_dataset order by point, customer_id")


def split(df):
    """Обучение: точки обучения без участников кампании; проверка: точка проверки."""
    return df[~df["is_test"] & ~df["excluded_campaign"]], df[df["is_test"]]


def rule_score(df, k, x):
    """Оценка риска правила: риск, если больше 1."""
    days = df["days_since_last"].to_numpy(float)
    gap = df["median_gap"].to_numpy(float)
    one = df["purchases"].to_numpy() == 1
    return np.where(one, days / x, days / (k * gap))


def fit_rule(train):
    """F1 на обучении по сетке k × X и лучшая пара."""
    grid = pd.DataFrame(
        [(k, x, f1_score(train["churn"], rule_score(train, k, x) > 1)) for k in K_GRID for x in X_GRID],
        columns=["k", "x", "f1"],
    )
    return grid, grid.loc[grid["f1"].idxmax()]


def design(df):
    """Признаки модели: пустые интервалы заполнены нулём, флаг одной покупки."""
    X = df[[c for c in NUMERIC if c != "one_purchase"] + CATEGORICAL].copy()
    X[NULLABLE] = X[NULLABLE].astype(float).fillna(0.0)
    X["one_purchase"] = (df["purchases"] == 1).astype(int)
    return X[NUMERIC + CATEGORICAL]


def fit_logreg(train):
    """Логистическая регрессия с C, подобранным по точкам обучения как фолдам."""
    pipe = make_pipeline(
        ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]),
        LogisticRegression(max_iter=2000),
    )
    search = GridSearchCV(pipe, {"logisticregression__C": C_GRID}, scoring="average_precision",
                          cv=LeaveOneGroupOut())
    search.fit(design(train), train["churn"], groups=train["point"])
    return search


def boosting_needed(boot):
    """Логистическая регрессия заметно лучше правила по PR-AUC."""
    row = boot.loc["pr_auc"]
    return bool(row["diff"] >= MIN_GAIN and row["lo"] > 0)


def fit_boosting(train):
    """Градиентный бустинг с параметрами, подобранными по точкам обучения как фолдам."""
    pipe = make_pipeline(
        ColumnTransformer(
            [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
            remainder="passthrough", verbose_feature_names_out=False,
        ).set_output(transform="pandas"),
        HistGradientBoostingClassifier(categorical_features=CATEGORICAL, early_stopping=False, random_state=SEED),
    )
    grid = {f"histgradientboostingclassifier__{k}": v for k, v in HGB_GRID.items()}
    search = GridSearchCV(pipe, grid, scoring="average_precision", cv=LeaveOneGroupOut())
    search.fit(design(train), train["churn"], groups=train["point"])
    return search


def top_k(y, score, k):
    """Точность и полнота среди k клиентов с наибольшим риском."""
    y = np.asarray(y)
    hits = y[np.argsort(-score, kind="stable")[:k]].sum()
    return hits / k, hits / y.sum()


def metrics(y, score, ks):
    """PR-AUC, ROC-AUC, точность и полнота в top-k."""
    m = {"pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score)}
    for k in ks:
        m[f"precision@{k}"], m[f"recall@{k}"] = top_k(y, score, k)
    return m


def diffs(y, a, b, ks):
    """Разница b − a по PR-AUC и точности в top-k."""
    d = {"pr_auc": average_precision_score(y, b) - average_precision_score(y, a)}
    for k in ks:
        d[f"precision@{k}"] = top_k(y, b, k)[0] - top_k(y, a, k)[0]
    return d


def bootstrap(y, a, b, ks, n=N_BOOT, seed=SEED):
    """Бутстреп по клиентам разницы b − a: оценка и 95% интервал."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    boot = pd.DataFrame([diffs(y[i], a[i], b[i], ks) for i in rng.integers(0, len(y), (n, len(y)))])
    return pd.DataFrame({
        "diff": diffs(y, a, b, ks),
        "lo": boot.quantile(0.025),
        "hi": boot.quantile(0.975),
    })


def calibration(y, p, bins=10):
    """Средняя предсказанная вероятность и фактическая доля по корзинам равного размера."""
    df = pd.DataFrame({"y": np.asarray(y), "p": p})
    df["bin"] = pd.qcut(df["p"], bins, labels=False)
    return df.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
