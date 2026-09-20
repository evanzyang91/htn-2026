.PHONY: install lint fmt embedder

install:  ## create .venv from uv.lock and fetch the Playwright browser
	uv sync
	uv run playwright install chromium

embedder:  ## fetch the retrieval embedding model (~90 MB, optional, off by default)
	uv run python scripts/fetch_embedder.py
	@echo 'now: SKILLWEAVER_EMBEDDER=true uv run python scripts/bench_retrieval.py'

lint:  ## check style and formatting without changing anything
	uv run ruff check .
	uv run ruff format --check .

fmt:  ## fix what can be fixed automatically
	uv run ruff check --fix .
	uv run ruff format .
