from pathlib import Path

from churn.db import connect

MARTS = Path("sql/marts")


def build(marts=MARTS):
    """Пересоздаёт схему mart и выполняет sql/marts/*.sql по порядку номеров."""
    with connect() as conn:
        conn.execute("drop schema if exists mart cascade; create schema mart")
        for path in sorted(Path(marts).glob("*.sql")):
            conn.execute(path.read_text())


if __name__ == "__main__":
    build()
    with connect() as conn:
        for path in sorted(MARTS.glob("*.sql")):
            name = path.stem.split("_", 1)[1]
            n = conn.execute(f"select count(*) from mart.{name}").fetchone()[0]
            print(f"mart.{name}: {n}")
