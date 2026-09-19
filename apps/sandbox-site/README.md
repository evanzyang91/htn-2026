# Northwind Console — sandbox site

A small, deliberately ordinary-looking web app for a computer-use agent to learn
on. It exists to serve three consumers:

1. **The agent** learns multi-step tasks here. No task worth learning can be done
   in a single click.
2. **A YOLO element detector** is trained on screenshots of it, so it uses real
   buttons, bordered inputs, checkboxes, toggles, tabs, menus, table rows, chips,
   cards, quantity steppers and an inline-SVG icon set — and four screens with
   deliberately different visual vocabularies.
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

## The four surfaces

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

**Pantry Lane** (amber, card catalogue + menu + cart) — a food-ordering app: seven
restaurants with cuisine, price band and rating, a search box that matches dish
names as well as restaurant names, cuisine filter chips, a menu page you navigate
into, dishes whose option groups are all **required** so a blind click cannot add
anything, a cart with quantity steppers and a running total, and a checkout that
commits the order into an order history. Money is integer cents everywhere — in
`seed.json`, on the server and in the browser — so a total can be asserted exactly
and the screen and the placed order can never disagree by a rounding step.

### Why a food-ordering app, and what it is not

This app is a DoorDash-**shaped** exercise, not an imitation of DoorDash or of any
other real service. It carries none of their name, branding, logo, wording or
styling, and it is not meant to look like them.

It exists because DoorDash itself is not reachable by this project. Pointed at
`https://www.doordash.com/` read-only, the shipped agent never sees any DoorDash
content: Cloudflare serves an interstitial reading "Verify you are human", and
perception finds ten text elements, every one of them belonging to that challenge
page. The identical code finds eighty-one elements on a Wikipedia article, so this
is not a perception-quality problem — the site refuses automated browsers at the
door. Getting past that check is bot-detection evasion, which this project will not
do, and a demo that depended on it would break the day Cloudflare tightened.

What actually matters is the **class** of task, not the brand. An ordering task
differs from a Wikipedia task in three ways, and Pantry Lane has all three:

1. a **catalogue** that has to be searched and filtered before the right row is found;
2. a **detail page** that has to be navigated into, with required choices that make a
   careless run fail rather than accidentally succeed;
3. a multi-step **state change that commits** — a cart that fills up and an order that
   gets placed.

The third is the hard one for this architecture, and it is the reason the app is here
rather than the reason to be embarrassed about it. A candidate skill is only stored
after being **re-run** from its recorded starting screen, so a task that changes state
can only be learned if something can put the world back. A real cart has no reset
endpoint and never will. `GET /__reset` is exactly that capability, which is what makes
"place an order" a learnable task at all. `eval/order.yaml` is the suite built on this
app; eight of its twelve tasks change state.

How exact that is, measured here: drive the whole ordering flow through to a placed
order, `GET /__reset`, reload, and the starting screen's fingerprint is **identical** —
similarity 1.000, same `value` string. Mid-task, for contrast, similarity to that
start is 0.150, which is the world genuinely having moved. The gate is
`Synthesizer.min_similarity` (`MIN_PRECONDITION_SIMILARITY`, currently 0.62), and an
exact match clears it with room to spare.

**Do not read 1.000 as a property of the web.** It is a property of *this site*, bought
by the determinism rules below, and real pages do not behave like it — the repository
`AGENTS.md` records the measurements: the same Wikipedia page on a second load bottoms
out around 0.84, two fresh browsers of opposite headless modes on one page score 0.44,
and a page carrying a sometimes-rendered fundraising banner scores 0.04 against the
same page recorded without one. So this app is evidence that the sandbox reproduces
exactly; it is evidence *against* tuning any similarity gate on the sandbox alone,
because a gate calibrated here looks green until it meets a website.

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
  js/mail.js  js/records.js  js/order.js  js/settings.js   one file per surface
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

Two things Pantry Lane learned the hard way, both worth keeping in mind for any
new surface here:

- **Do not paint a control that is not one.** The cart figure in the Pantry Lane
  header started as a read-only pill. It reads as the cart button every ordering
  site has, and in a real evaluation run the explorer clicked it, saw the screen
  not change, could not tell a dead affordance from a mis-click, and spent its
  whole budget there — one cold run lost, and $1.93 with it. It is a real button
  now. Anything that looks pressable in this app has to do something.
- **A new surface with a modal must CHAIN the dialog builder.** The overlay has
  one `App.screens.dialog` for the whole app; `static/js/order.js` keeps a
  reference to whatever was registered before it and falls through, rather than
  replacing Settings' confirmation dialog by being loaded after it.
