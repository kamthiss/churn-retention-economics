import os

import psycopg

DEFAULT_DSN = "postgresql://churn:churn@localhost:5436/churn"


def connect():
    """Подключение к базе проекта по CHURN_DSN."""
    return psycopg.connect(os.environ.get("CHURN_DSN", DEFAULT_DSN))
