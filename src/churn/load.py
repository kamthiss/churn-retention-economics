from pathlib import Path

import pandas as pd

from churn.db import connect

SCHEMA = Path("sql/raw_schema.sql")
RAW = Path("data/raw")
# порядок важен из-за внешних ключей
TABLES = ("customers", "orders", "campaigns", "campaign_contacts")


def load(raw=RAW):
    """Пересоздаёт схему raw и грузит parquet через COPY."""
    with connect() as conn:
        conn.execute(SCHEMA.read_text())
        for name in TABLES:
            df = pd.read_parquet(Path(raw) / f"{name}.parquet")
            cols = ", ".join(df.columns)
            with conn.cursor().copy(f"copy raw.{name} ({cols}) from stdin (format csv)") as cp:
                cp.write(df.to_csv(index=False, header=False))


if __name__ == "__main__":
    load()
    with connect() as conn:
        for name in TABLES:
            n = conn.execute(f"select count(*) from raw.{name}").fetchone()[0]
            print(f"raw.{name}: {n}")
