from pathlib import Path

from churn.db import connect

MARTS = Path("sql/marts")


def build(marts=MARTS):
    """Пересоздаёт таблицы mart из sql/marts/*.sql по порядку номеров."""
    with connect() as conn:
        # схему целиком не удаляем: в ней же лежит mart.model_dataset из churn.features
        conn.execute("create schema if not exists mart")
        for path in sorted(Path(marts).glob("*.sql")):
            conn.execute(f"drop table if exists mart.{path.stem.split('_', 1)[1]}")
            conn.execute(path.read_text())


if __name__ == "__main__":
    build()
    with connect() as conn:
        for path in sorted(MARTS.glob("*.sql")):
            name = path.stem.split("_", 1)[1]
            n = conn.execute(f"select count(*) from mart.{name}").fetchone()[0]
            print(f"mart.{name}: {n}")
