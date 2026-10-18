PY = .venv/bin/python

.PHONY: data test

data:
	$(PY) -m churn.generate

test:
	$(PY) -m pytest -q
