# Lantern Board — sandbox site

A deterministic kanban project tracker for a computer-use agent to learn on.
It is deliberately a **different UI idiom** from the Northwind Console mail /
table / settings surfaces: four columns of cards side by side, an inline move
menu, a slide-in detail panel, and two modal dialogs. Same contract, different
perception and interaction patterns.

It stands alone: no database, no framework, no build step, no npm, and no
dependency on the `skillweaver` package.

## Run it

```sh
python3 apps/board-site/serve.py --port 8767
# http://127.0.0.1:8767
```

Standard library only — no pip install to run the app. `--host` defaults to
`127.0.0.1`, `--port` to `8767`.

## Endpoints for evaluation code

| Endpoint | What it does |
| --- | --- |
| `GET /__reset` | Restores the exact `seed.json` starting state (data **and** UI) and returns `{"ok": true, "state": {...}}`. `POST /__reset` does the same. |
| `GET /__state` | The complete current state as JSON, so a run can assert an outcome without reading the screen. |
| `GET /__seed`  | The pristine seed, for diffing. |
| `POST /api/act` | `{"action": "...", "payload": {...}, "seq": n}` — the single mutation entry point the UI itself uses. Returns `{"seq": n, "state": {...}}`. An unknown action is HTTP 400 `{"error": "unknown action: ..."}`. |

The browser does **not** hold state. Every interaction posts to `/api/act` and
re-renders the whole screen from the response, which is why `/__state` is
always an exact description of what is on screen.

## The board

One screen: a header toolbar (search box, an **Assignee** filter, a
**Priority** filter, **Archive done**, **New ticket**) above four columns —
**Backlog**, **In progress**, **Review**, **Done** — each with a name, a count,
and a vertical stack of ticket cards. A card shows its key (`LAN-14`), title,
assignee, a priority chip, and a **Move** button that opens an inline menu
listing the other three columns, so every move is a plain click (no
drag-and-drop required). Clicking a card **title** opens a detail panel on the
right: key, priority chip, status, an editable title with **Save title**, the
description, an **Assign to** select, the comment list, a comment box with
**Add comment**, and **Close**. **New ticket** opens a modal (title,
description, assignee, priority, **Create** / **Cancel**); new tickets land in
Backlog with the next `LAN-n` key. **Archive done** opens a confirm dialog
("Archive 3 tickets?" / **Archive** / **Keep them**) that moves every Done
ticket to the archive. Every mutating action raises a dismissible banner.

## Action vocabulary

| Action | Payload | Effect |
| --- | --- | --- |
| `board.search` | `{q}` | Set the search text; every column filters on key, title, description and assignee. |
| `board.filterAssignee` | `{assignee}` | Filter cards to one assignee, or `"all"`. |
| `board.filterPriority` | `{priority}` | Filter cards to one priority (`High`/`Medium`/`Low`), or `"all"`. |
| `board.toggleMoveMenu` | `{id}` | Open the move menu on that card, or close it if already open. |
| `board.move` | `{id, column}` | Move the ticket to that column (`backlog`/`inprogress`/`review`/`done`); banner `Moved LAN-n to <Column>`. Same-column moves are no-ops. |
| `board.open` | `{id}` | Open the detail panel; seeds `ui.draft.title` with the ticket's title. |
| `board.close` | — | Close the detail panel and clear the draft. |
| `board.draftTitle` | `{value}` | Update the title draft in the detail panel. |
| `board.saveTitle` | — | Commit the (non-empty) title draft to the open ticket; banner `Renamed LAN-n to <title>`. |
| `board.assign` | `{assignee}` | Reassign the open ticket; banner `Assigned LAN-n to <name>`. |
| `board.draftComment` | `{value}` | Update the comment draft in the detail panel. |
| `board.addComment` | — | Append the (non-empty) comment draft to the open ticket as the signed-in user, dated `meta.today`; banner `Comment added to LAN-n`. |
| `board.composeOpen` | — | Open the New ticket modal with a blank form. |
| `board.composeField` | `{field, value}` | Set a modal field: `title`, `description`, `assignee` or `priority`. |
| `board.composeClose` | — | Close the modal and discard the form. |
| `board.create` | — | Create a ticket from the (non-empty-titled) form in Backlog with the next key; banner `Created LAN-n in Backlog`. |
| `board.archiveDialog` | — | Open the archive confirm dialog (only if Done is non-empty). |
| `board.archiveCancel` | — | Close the dialog, archiving nothing. |
| `board.archiveConfirm` | — | Move every Done ticket to `archived`; banner `Archived n tickets`. |
| `board.dismissBanner` | — | Clear the banner. |

## State shape

```
{
  "meta":  { "appName", "today", "user": {"name", "email"} },
  "board": {
    "columns":    [ {"id", "name"} x4 ],
    "assignees":  [ 5 names ],
    "priorities": ["High", "Medium", "Low"],
    "counters":   { "ticket", "comment" },      // next LAN-n / c-nn come from here
    "tickets":    [ { "id", "key", "title", "description", "assignee",
                      "priority", "column", "comments": [{"id", "author",
                      "text", "date"}] } x14 ]
  },
  "archived": [ tickets removed from the board by Archive done ],
  "ui": {
    "search", "assignee", "priority",           // toolbar filters ("all" = off)
    "openId",                                   // ticket in the detail panel, or null
    "moveMenuFor",                              // card with its move menu open, or null
    "compose": { "open", "title", "description", "assignee", "priority" },
    "dialogOpen",                               // the archive confirm dialog
    "banner",                                   // last mutation notice, or null
    "draft": { "title", "comment" }             // detail-panel edit buffers
  }
}
```

IDs are counters (`t15`/`LAN-15`, `c07`, ...), never random.

## Determinism

The same rules as the Northwind sandbox, asserted by `tests/test_determinism.py`:

- All state derives from `seed.json`. No `Math.random()`, no `Date.now()`, no
  relative times; "today" is `meta.today` in the seed.
- No CSS `transition` or `animation` anywhere — disabled globally in
  `static/css/app.css`.
- Text carets do not blink (`caret-color: transparent`).
- Scrollbar gutters are always reserved, so focus never shifts the layout.
- Actions are sent to the server strictly one at a time through the queue in
  `static/js/app.js`, so a typed string is reproducible.
- The move menu renders **inline inside the card** (no floating overlay), so a
  scrollable column can never clip it and opening it is a pure layout change.

Reachability is part of the contract: column stacks and the detail panel use
`overflow-y: auto` (never `overflow: hidden` around reachable content), every
control has a visible text label, and the whole app is verified usable at a
1280x800 viewport — which is exactly the viewport the tests run at.

## Tests

Playwright smoke tests covering every flow above, plus a browser-free suite
(`tests/test_api.py`) that round-trips the entire action vocabulary through
`POST /api/act`. They need their own venv — nothing is added to the repository
root to make them run.

```sh
cd apps/board-site
python3 -m venv .venv-test
./.venv-test/bin/pip install -r requirements-dev.txt
./.venv-test/bin/playwright install chromium
./.venv-test/bin/python -m pytest tests/ -v
```

The tests start their own server on a free port and reset it before each test,
so they do not care whether one is already running. `.venv-test/` is gitignored.

`pytest.ini` lives here on purpose: without it pytest walks up to the
repository root `pyproject.toml` and inherits its `testpaths`, `addopts` and
import mode, which do not apply to this standalone app. Run the suite from this
directory, or point pytest at the config explicitly:

```sh
apps/board-site/.venv-test/bin/python -m pytest \
  apps/board-site/tests -c apps/board-site/pytest.ini
```

## Layout

```
serve.py     stdlib HTTP server; owns the state, applies every action
seed.json    the single source of starting state (14 tickets, 5 assignees)
static/
  index.html app shell and the inline SVG icon sprite
  css/app.css hand-written; kanban layout, amber accent
  js/app.js   DOM helpers, the serialized action queue, the render loop
  js/board.js the whole surface: toolbar, columns, cards, detail, dialogs
  js/boot.js  fetches state once and renders
tests/       Playwright + API smoke tests (conftest.py owns the server fixture)
pytest.ini   keeps this app's test config out of the repository root's
```

Every interactive element carries a `data-testid`, which doubles as a stable
handle for agents and for the tests.

## Adding to it

Add an action to `apply_action` in `serve.py` and render it in
`static/js/board.js`. Keep the determinism and reachability rules above; they
are what the evaluation harness depends on.
