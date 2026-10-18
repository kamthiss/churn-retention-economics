PY = .venv/bin/python

.PHONY: data test db-up db-down load marts features test-db test-all reset-db notebooks

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

features:
	$(PY) -m churn.features

notebooks:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks/*.ipynb

test-db:
	$(PY) -m pytest -q -m db

test-all:
	$(MAKE) data db-up load marts features
	$(PY) -m pytest -q

reset-db:
	docker compose down -v
	$(MAKE) db-up data load marts features
