# skillweaver

A self-growing skill library for computer-use agents (Hack the North 2026).

Voyager's "write your own skills as code" loop, applied to a general computer-use
agent. The agent solves a task once by trial and error with a computer-use model
(Claude primary, Gemini alongside), saves what worked as a reusable Python skill, and
composes stored skills into faster, model-free runs. It sees through screenshots
(YOLO for elements, OCR for text) and keeps a graph per site of known screens and the
actions that move between them. Browser first; a real desktop controller too.

## Install

Requires [uv](https://docs.astral.sh/uv/) on macOS (Apple silicon is what is tested).
uv fetches Python 3.12 itself.

```sh
make install     # uv sync + playwright install chromium
```

Settings come from the environment or `.env`; `src/skillweaver/config.py` lists every
variable. An Anthropic API key is needed: every run drives a real model against a real
site.

## Lint

```sh
make lint        # ruff check + format check
make fmt         # auto-fix
```

## Layout

- `src/skillweaver/contracts.py` - every shared type and Protocol. Start here.
- `src/skillweaver/<package>/` - implementations, one package per concern.
- `eval/wikipedia.yaml` - the live-site evaluation suite.
