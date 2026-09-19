# Northwind Console — sandbox site

A small, deliberately ordinary-looking web app for a computer-use agent to learn
on. It exists to serve three consumers:

1. **The agent** learns multi-step tasks here. No task worth learning can be done
   in a single click.
2. **A YOLO element detector** is trained on screenshots of it, so it uses real
   buttons, bordered inputs, checkboxes, toggles, tabs, menus, table rows and an
   inline-SVG icon set — and three screens with deliberately different visual
   vocabularies.
3. **Evaluation runs** assert outcomes against it, so its state is exactly
   reproducible and its pixels are stable.

It stands alone: no database, no framework, no build step, no npm, and no
dependency on the `skillweaver` package.

## Run it

```sh
python3 apps/sandbox-site/serve.py --port 8765
# http://127.0.0.1:8765
```

Standard library only — no pip install to run the app. Tested on Python 3.13.
`--host` defaults to `127.0.0.1`.

## Endpoints for evaluation code

| Endpoint | What it does |
| --- | --- |
| `GET /__reset` | Restores the exact `seed.json` starting state (data **and** UI) and returns `{"ok": true, "state": {...}}`. Sub-millisecond. |
| `GET /__state` | The complete current state as JSON, so a run can assert an outcome without reading the screen. |
| `GET /__seed`  | The pristine seed, for diffing. |
| `POST /api/act` | `{"action": "...", "payload": {...}}` — the single mutation entry point the UI itself uses. |

The browser does **not** hold state. Every interaction posts to `/api/act` and
re-renders from the response, which is why `/__state` is always an exact
description of what is on screen.

Reset the app between runs:

```sh
curl -s http://127.0.0.1:8765/__reset > /dev/null
```

The page reads state once on load, so reset then reload (or open a fresh tab).

## The three surfaces

**Mail** (blue, sidebar + list + reading pane) — folders with counts, a search
box that filters, per-row checkboxes with a select-all in the header, Archive and
Label actions that stay disabled until something is selected, a split reading
pane, and a compose window with to/subject/body and Send. The multi-step and
composite tasks live here.

**Records** (teal, dense data table) — 40 rows, a text filter, sortable columns,
inline editing of the Name cell, a bulk status action across the selection, and
an export button that records what it exported.

**Settings** (indigo, centered cards) — text input, two selects and four toggle
switches, a Save that opens a confirmation dialog, and a visible success state.
The dialog gives the skill graph a modal node with a real confirm-or-cancel
branch: cancelling keeps the draft and saves nothing.

## Determinism

This is a hard requirement, not a nicety. Reaching the same screen twice produces
a byte-identical screenshot.

- All state derives from `seed.json`. No `Math.random()`, no `Date.now()`, no
  relative times like "2 hours ago". Dates are fixed strings; "today" is
  `meta.today` in the seed.
- No CSS `transition` or `animation` anywhere — both are disabled globally in
  `static/css/app.css` so a screenshot can never catch one mid-flight.
- Text carets do not blink (`caret-color: transparent`), so a focused field
  renders the same every time. Focus rings are kept prominent instead.
- Scrollbar gutters are always reserved, so focus never shifts the layout.
- Actions are sent to the server strictly one at a time. Parallel `fetch` calls
  are free to arrive out of order, and an out-of-order keystroke would corrupt
  the state, so the queue in `static/js/app.js` is what makes a typed string
  reproducible.

`tests/test_determinism.py` asserts all of this, including two byte comparisons
of screenshots.

## Tests

Playwright smoke tests covering every flow above. They need their own venv —
nothing is added to the repository root to make them run.

```sh
cd apps/sandbox-site
python3 -m venv .venv-test
./.venv-test/bin/pip install -r requirements-dev.txt
./.venv-test/bin/playwright install chromium
./.venv-test/bin/python -m pytest tests/ -v
```

The tests start their own server on a free port and reset it before each test, so
they do not care whether one is already running. `.venv-test/` is gitignored.

`pytest.ini` lives here on purpose: without it pytest walks up to the repository
root `pyproject.toml` and inherits its `testpaths`, `addopts` and import mode,
which do not apply to this standalone app. Run the suite from this directory. To
run it from the repository root, point pytest at that config explicitly:

```sh
apps/sandbox-site/.venv-test/bin/python -m pytest \
  apps/sandbox-site/tests -c apps/sandbox-site/pytest.ini
```

A plain `pytest` at the repository root does not collect these tests, and is
unaffected by them.

## Layout

```
serve.py     stdlib HTTP server; owns the state, applies every action
seed.json    the single source of starting state
static/
  index.html app shell and the inline SVG icon sprite
  css/app.css hand-written; per-screen accent via body[data-screen]
  js/app.js   DOM helpers, the serialized action queue, the render loop
  js/mail.js  js/records.js  js/settings.js   one file per surface
  js/boot.js  fetches state once and renders
tests/       Playwright smoke tests (conftest.py owns the server fixture)
pytest.ini   keeps this app's test config out of the repository root's
```

Every interactive element carries a `data-testid`, which doubles as a stable
handle for agents and for the tests.

## Adding to it

Add an action to `apply_action` in `serve.py` and render it in the matching
surface module. Keep the determinism rules above; they are what the evaluation
harness depends on.
