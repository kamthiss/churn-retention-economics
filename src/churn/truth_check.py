import pandas as pd

from churn.db import connect
from churn.features import query
from churn.params import TEST_POINT

TRUTH = "data/truth/customers_truth.parquet"

if __name__ == "__main__":
    with connect() as conn:
        df = query(conn, "select customer_id, churn from mart.model_dataset where point = %s", (TEST_POINT,))
    t = pd.read_parquet(TRUTH, columns=["customer_id", "died", "last_purchase_before_death"])
    df = df.merge(t, on="customer_id", validate="1:1")
    dead = df["died"] & (df["last_purchase_before_death"] < pd.Timestamp(TEST_POINT))
    churned = df["churn"] == 1
    print(f"точка {TEST_POINT}, активных: {len(df)}")
    print(f"churn = 1: {churned.sum()}, из них на самом деле живы: {(~dead[churned]).mean():.3f}")
    print(f"умерли к T: {dead.sum()}, из них получили churn = 1: {churned[dead].mean():.3f}")
