# skillweaver

A self-growing skill library for computer-use agents (Hack the North 2026).

Voyager's "write your own skills as code" loop, applied to a general computer-use
agent. The agent solves a task once by trial and error with a computer-use model
(Claude primary, Gemini alongside), saves what worked as a reusable Python skill, and
composes stored skills into faster, model-free runs. It sees through screenshots
(YOLO for elements, OCR for text) and keeps a graph per site of known screens and the
actions that move between them. Browser first; a real desktop controller too.

## See it work

```sh
make install
make demo          # or: uv run python scripts/demo.py --watch
```

A browser opens and does one errand on a storefront it has never seen - add an item to
the cart, fill in a checkout, place the order - by exploring. Then it does the same
errand again from memory, several times faster and with one model call instead of
thirteen, and prints the Python it wrote for itself in between. About a minute.

Drop `--watch` to run it headless in about thirty seconds.

Or on a real website nobody here built:

```sh
uv run python scripts/demo_real_store.py --watch
```

Sauce Labs' public demo storefront, six screens end to end: sign in, find a named
product among several, add it to the cart, open a cart whose icon has no text at all,
fill in a three-field checkout, confirm. Sixteen actions, about ten seconds, no
selectors and no DOM - every target found in the screenshot. The shop decides whether
it worked, by landing on its own "Thank you for your order!".

A real merchant would be the wrong place to prove this: it would spend someone's money
and create an obligation that cannot be withdrawn. The interesting question - can it
drive six screens of a site nobody built for it, from pixels - is answered without
any of that.

That demo runs a skill a person wrote. With an `ANTHROPIC_API_KEY` the agent writes it
itself, against the same storefront:

```sh
uv run python -m skillweaver.cli --data-dir data/live learn \
  "Sign in as standard_user with the password secret_sauce, add the Sauce Labs \
Backpack to the cart, and complete the checkout as Ada Lovelace with postcode SW1A 1AA." \
  --url https://www.saucedemo.com
```

It explores the errand in 17 actions and 13 model calls, and the admission gate returns
the site to its sign-in page, re-runs the Python it wrote and stores it - first attempt,
no repairs. `run` on the same sentence then replays all six screens in **0 model calls**
and about 11 seconds, and does it for a different product and a different buyer, because
what varied between the two runs became parameters rather than literals.

## What it measures

Three applications, one agent, nothing in it told which it is looking at. A mail
console, a storefront and a kanban tracker share no idiom - rows against cards against
columns, a toolbar against a sidebar against a per-card menu - so a number that holds
across all three is about the agent rather than about a layout.

```sh
uv run python scripts/bench_all.py --model-latency-ms 2500
```

| suite   | tasks | cold ok | warm ok | cold    | warm   | speedup | cold calls | warm calls |
| ------- | ----- | ------- | ------- | ------- | ------ | ------- | ---------- | ---------- |
| console | 14    | 100%    | 100%    | 21.1 s  | 4.0 s  | 5.33x   | 5.57       | 0.86       |
| shop    | 10    | 100%    | 100%    | 18.6 s  | 3.8 s  | 4.86x   | 4.80       | 0.90       |
| board   | 10    | 100%    | 100%    | 18.0 s  | 3.9 s  | 4.59x   | 4.40       | 0.90       |
| **all** | 34    | 100%    | 100%    | 19.4 s  | 3.9 s  | **4.98x** | 5.00     | 0.88       |

Every run is a real Chromium driven through pixels, scored by each application's own
state, which the agent never sees. Hand the task its parameters as well as its sentence
and the warm path consults no model at all: 0.00 calls and 10.6x on the board.

The model is stood in for by a deterministic operator (`scripts/scripted_operator.py`)
that reads the same prompt a model would and nothing else, so these numbers measure
the system rather than the model. `--model-latency-ms 0` removes model time entirely,
which makes only the COLD side cheaper - the speedup is then 2.6x, a floor rather than
a forecast.

## Install

Requires [uv](https://docs.astral.sh/uv/) on macOS (Apple silicon is what is tested).
uv fetches Python 3.12 itself.

```sh
make install     # uv sync + playwright install chromium
```

Settings come from the environment or `.env`; `src/skillweaver/config.py` lists every
variable. No API key is needed to run the tests.

## Test

```sh
make test        # uv run pytest
make lint        # ruff check + format check
make fmt         # auto-fix
```

## Layout

- `src/skillweaver/contracts.py` - every shared type and Protocol. Start here.
- `src/skillweaver/<package>/` - implementations, one package per concern.
- `tests/fakes/` - test doubles for every Protocol, plus a small fake app
  (`scenario.py`), so anything can be tested without a browser, a model or a network.
