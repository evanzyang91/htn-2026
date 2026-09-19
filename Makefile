.PHONY: install test lint fmt

install:  ## create .venv from uv.lock and fetch the Playwright browser
	uv sync
	uv run playwright install chromium

test:  ## run the test suite
	uv run pytest

lint:  ## check style and formatting without changing anything
	uv run ruff check .
	uv run ruff format --check .

fmt:  ## fix what can be fixed automatically
	uv run ruff check --fix .
	uv run ruff format .
