.PHONY: install install-all test test-fast lint lab notebooks

install:        ## create .venv and install the project + dev/notebooks groups (uv.lock, Python 3.13)
	uv sync

install-all:    ## also install the DoubleML cross-check and the Dagster + MLflow pipeline groups
	uv sync --group dml --group pipeline

test:           ## all tests (unit + slow + data-dependent)
	uv run pytest

test-fast:      ## unit tests only (no Favorita files needed, no Monte Carlo)
	uv run pytest -m "not slow and not data"

lint:
	uv run ruff check src tests tools pipeline

lab:
	uv run jupyter lab

notebooks:      ## re-execute every notebook in place (needs data/processed and data/results)
	for nb in notebooks/*.ipynb; do \
		uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$$nb" || exit 1; \
	done
