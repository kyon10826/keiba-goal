PY := .venv/bin/python

.PHONY: test smoke-ml

test:
	$(PY) -m pytest

smoke-ml:
	$(PY) scripts/smoke_ml.py
