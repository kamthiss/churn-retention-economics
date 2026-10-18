PY = .venv/bin/python

.PHONY: data test db-up db-down load marts test-db test-all reset-db

data:
	$(PY) -m churn.generate

test:
	$(PY) -m pytest -q -m "not db"

db-up:
	docker compose up -d --wait

db-down:
	docker compose down

load:
	$(PY) -m churn.load

marts:
	$(PY) -m churn.marts

test-db:
	$(PY) -m pytest -q -m db

test-all:
	$(MAKE) data db-up load marts
	$(PY) -m pytest -q

reset-db:
	docker compose down -v
	$(MAKE) db-up data load marts
