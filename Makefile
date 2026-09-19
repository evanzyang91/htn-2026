.PHONY: install test lint fmt demo bench

install:  ## create .venv from uv.lock and fetch the Playwright browser
	uv sync
	uv run playwright install chromium

test:  ## run the test suite
	uv run pytest

demo:  ## watch it learn one errand and then repeat it from memory
	uv run python scripts/demo.py --watch

bench:  ## all three suites against all three applications
	uv run python scripts/bench_all.py --model-latency-ms 2500

lint:  ## check style and formatting without changing anything
	uv run ruff check .
	uv run ruff format --check .

fmt:  ## fix what can be fixed automatically
	uv run ruff check --fix .
	uv run ruff format .
