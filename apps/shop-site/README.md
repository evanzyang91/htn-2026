# Harbour Supply — shop site

A small, deliberately ordinary-looking e-commerce storefront for a computer-use
agent to learn on. It is a sibling of `apps/sandbox-site` with the same server
contract but a **different UI idiom**: a card grid instead of tables and lists,
a filter sidebar with radio buttons, a slide-in cart drawer with quantity
steppers, and a checkout form that ends in a confirm dialog. That difference is
the point — it exercises perception and interaction patterns the console app
does not.

It stands alone: no database, no framework, no build step, no npm, and no
dependency on the `skillweaver` package.

## Run it

```sh
python3 apps/shop-site/serve.py --port 8766
# http://127.0.0.1:8766
```

Standard library only — no pip install to run the app. `--host` defaults to
`127.0.0.1`, `--port` to `8766`.

## Endpoints for evaluation code

| Endpoint | What it does |
| --- | --- |
| `GET /__reset` | Restores the exact `seed.json` starting state (data **and** UI) and returns `{"ok": true, "state": {...}}`. `POST /__reset` does the same. |
| `GET /__state` | The complete current state as JSON, so a run can assert an outcome without reading the screen. |
| `GET /__seed`  | The pristine seed, for diffing. |
| `POST /api/act` | `{"action": "...", "payload": {...}, "seq": n}` — the single mutation entry point the UI itself uses. Returns `{"seq": n, "state": {...}}`. Unknown action → HTTP 400 `{"error": "unknown action: ..."}`. |

The browser does **not** hold state. Every interaction posts to `/api/act` and
re-renders the whole screen from the response, which is why `/__state` is always
an exact description of what is on screen. The page reads state once on load, so
reset then reload (or open a fresh tab). Every response carries
`Cache-Control: no-store`.

## Action vocabulary

All prices are **integer cents**; only display code divides by 100.

| Action | Payload | Effect |
| --- | --- | --- |
| `shop.search` | `{"q": str}` | Sets the header search text. The grid narrows to products whose name or category contains it. |
| `shop.category` | `{"category": "All"\|"Tools"\|"Outdoor"\|"Kitchen"\|"Lighting"}` | Sets the category filter. Other values are ignored. |
| `shop.priceBand` | `{"band": "any"\|"under25"\|"25to75"\|"over75"}` | Sets the price-band radio. Bands: `< $25`, `$25–$75` inclusive, `> $75`. |
| `shop.sort` | `{"sort": "featured"\|"price-asc"\|"price-desc"\|"rating-desc"\|"name-asc"}` | Sets the sort dropdown. Ties always break by product id, so the order is deterministic. |
| `shop.dismissBanner` | `{}` | Clears `ui.banner`. |
| `cart.add` | `{"productId": "p01"}` | Adds one unit; an existing line grows by 1. Unknown ids are ignored. |
| `cart.open` | `{}` | Opens the cart drawer (`cartOpen=true`), showing the lines view. |
| `cart.close` | `{}` | Closes the drawer, the checkout panel and the dialog. |
| `cart.increment` | `{"productId": "p01"}` | Quantity +1 on that line. |
| `cart.decrement` | `{"productId": "p01"}` | Quantity −1, floored at 1 (the − button is disabled at 1). |
| `cart.remove` | `{"productId": "p01"}` | Deletes the line. Emptying the cart also closes checkout. |
| `checkout.open` | `{}` | Shows the checkout form in the drawer. No-op while the cart is empty. |
| `checkout.back` | `{}` | Returns from the form to the cart lines view. |
| `checkout.field` | `{"field": "name"\|"email"\|"address"\|"shipping"\|"save", "value": ...}` | Edits one form field. `shipping` must be a known option id; `save` is coerced to bool. |
| `checkout.submit` | `{}` | Opens the confirm dialog — only if the cart has lines, checkout is open and name/email/address are non-blank ("Place order"). |
| `checkout.cancel` | `{}` | Closes the confirm dialog, keeping the form and cart ("Back"). |
| `checkout.confirm` | `{}` | Places the order: appends to `orders` with counter id `oNN` / number `1000+NN`, empties the cart, closes every panel, sets the confirmation banner. If `form.save` is false the form resets to blank; if true the details are kept for the next order. |

## State shape

```
{
  "meta":    { "appName", "tagline", "today", "currency",
               "shipping": [ { "id", "label", "fee" }, ... ] },
  "catalog": { "products": [ { "id", "name", "category", "price", "rating" }, ... ] },
  "cart":    { "lines": [ { "productId", "qty" }, ... ] },
  "orders":  [ { "id", "number", "items": [ { "productId", "name", "price", "qty" } ],
                 "subtotal", "shippingFee", "total", "shipping",
                 "name", "email", "address", "saveDetails", "placed" }, ... ],
  "ui":      { "screen": "shop", "search", "category", "priceBand", "sort",
               "cartOpen", "checkoutOpen", "dialogOpen", "banner",
               "form": { "name", "email", "address", "shipping", "save" } }
}
```

24 products across 4 categories (Tools, Outdoor, Kitchen, Lighting), 6 each,
with varied prices and ratings. Product ids `p01`–`p24` are counters, as are
order ids.

## The surface

One screen, several distinct visual states:

- **Grid** — a card per product: coloured tile with the product's initials,
  name, category chip, `★ rating`, price, and an "Add to cart" button that
  shows the quantity already in the cart.
- **Sidebar** — Categories as buttons with counts, Price as a radio group.
- **Header** — a labelled search box and a "Cart (n)" button.
- **Cart drawer** — present or absent on the right, never animated. Line items
  with − / + steppers, a "Remove" per line, a subtotal and "Checkout".
- **Checkout panel** — replaces the lines view in the drawer: Full name, Email,
  Address, a Shipping speed `<select>` (fees change the total), a "Save these
  details" checkbox, "Back to cart" and "Place order".
- **Confirm dialog** — "Place this order?" with the item count, total and
  ship-to line; "Back" or "Confirm order".
- **Banner** — "Order #1001 placed - thank you!" with a "Dismiss" button.

## Determinism and reachability

This is a hard requirement, not a nicety. Reaching the same state twice produces
a byte-identical screenshot.

- All state derives from `seed.json`. No `Math.random()`, no `Date.now()`, no
  timers, no relative times. "Today" is `meta.today` in the seed. Ids are
  counters. Money is integer cents.
- No CSS `transition` or `animation` anywhere — both are disabled globally in
  `static/css/app.css`. The drawer and dialog are present or absent, never
  mid-flight.
- Text carets do not blink (`caret-color: transparent`); scrollbar gutters are
  always reserved.
- Actions are sent to the server strictly one at a time; the queue in
  `static/js/app.js` is what makes a typed string reproducible.
- Scrollable panes (`.gridwrap`, `.drawer-body`, `.shop-side`) use
  `overflow-y: auto`, never `overflow: hidden`, so everything below the fold is
  mouse-wheel reachable. The whole app is usable at a 1280×800 viewport.
- Every control shows a readable text label ("Add to cart", "Checkout",
  "Remove", "Close", "Place order", …) and every interactive element carries a
  `data-testid`, which doubles as a stable handle for agents and for the tests.

`tests/test_determinism.py` asserts all of this, including byte comparisons of
screenshots and a reachability check on the last card at 1280×800.

## Tests

Playwright smoke tests covering every flow above. They need their own venv —
nothing is added to the repository root to make them run.

```sh
cd apps/shop-site
python3 -m venv .venv-test
./.venv-test/bin/pip install -r requirements-dev.txt
./.venv-test/bin/playwright install chromium
./.venv-test/bin/python -m pytest tests/ -v
```

The tests start their own server on a free port and reset it before each test,
so they do not care whether one is already running.

`pytest.ini` lives here on purpose: without it pytest walks up to the repository
root `pyproject.toml` and inherits its `testpaths`, `addopts` and import mode,
which do not apply to this standalone app. Run the suite from this directory. To
run it from the repository root, point pytest at that config explicitly:

```sh
python -m pytest apps/shop-site/tests -c apps/shop-site/pytest.ini
```

## Layout

```
serve.py     stdlib HTTP server; owns the state, applies every action
seed.json    the single source of starting state
static/
  index.html app shell and the inline SVG icon sprite
  css/app.css hand-written; card grid, drawer, dialog
  js/app.js   DOM helpers, the serialized action queue, the render loop
  js/shop.js  sidebar, toolbar and product grid
  js/cart.js  the drawer and its line items
  js/checkout.js  the form panel and the confirm dialog
  js/boot.js  fetches state once and renders
tests/       Playwright smoke tests (conftest.py owns the server fixture)
pytest.ini   keeps this app's test config out of the repository root's
```

## Adding to it

Add an action to `apply_action` in `serve.py` and render it in the matching
static module. Keep the determinism and reachability rules above; they are what
the evaluation harness depends on.
