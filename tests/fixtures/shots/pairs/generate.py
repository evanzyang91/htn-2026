"""Render the labelled screenshot pairs in this directory from the HTML in ``pages/``.

Run it from the repository root after editing a page or adding a pair::

    uv run python tests/fixtures/shots/pairs/generate.py

It needs Playwright's chromium (``make install``); the committed PNGs and ``meta.json``
files mean the test suite itself never needs a browser.

Each pair becomes ``<name>/{a.png,b.png,meta.json}``. ``meta.json`` carries the label
(``same`` or ``different`` UI state), the synthetic URL each side was captured under - the
pages load over ``file://``, whose path is machine-specific, so the URL is assigned here -
and the DOM-truth element list in LOGICAL pixels, clipped to the viewport. Elements are
the nodes carrying ``data-sw-kind``, whose value is a ``contracts.ElementKind``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"
WIDTH, HEIGHT = 800, 600

_ELEMENTS_JS = """
(viewport) => Array.from(document.querySelectorAll('[data-sw-kind]')).flatMap((node) => {
  const r = node.getBoundingClientRect();
  const x = Math.max(r.left, 0), y = Math.max(r.top, 0);
  const w = Math.min(r.right, viewport.w) - x, h = Math.min(r.bottom, viewport.h) - y;
  if (w <= 0 || h <= 0) return [];
  return [{
    x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h),
    kind: node.dataset.swKind,
    text: (node.innerText || '').replace(/\\s+/g, ' ').trim(),
  }];
}).sort((a, b) => (a.y - b.y) || (a.x - b.x))
"""


@dataclass(frozen=True)
class Side:
    """One half of a pair: which page to load, under which URL, and how to disturb it."""

    page: str
    url: str | None
    scroll: int = 0
    scale: float = 1.0
    script: str | None = None
    """JavaScript run after load, for the cosmetic differences a re-capture really has."""


@dataclass(frozen=True)
class Pair:
    name: str
    label: str
    why: str
    a: Side
    b: Side
    tags: tuple[str, ...] = field(default=())


_HIDE_CARET = "document.getElementById('search').classList.add('blink')"
_TICK_CLOCK = "document.getElementById('clock').textContent = '09:42'"

PAIRS: tuple[Pair, ...] = (
    Pair(
        "same_screen_twice",
        "same",
        "Two independent captures of one screen: the baseline every other pair is read against.",
        Side("invoices.html", "https://invoices.test/invoices"),
        Side("invoices.html", "https://invoices.test/invoices"),
    ),
    Pair(
        "same_screen_caret_blink",
        "same",
        "One screen recaptured while the text caret was in its off phase - cosmetic only.",
        Side("invoices.html", "https://invoices.test/invoices"),
        Side("invoices.html", "https://invoices.test/invoices", script=_HIDE_CARET),
    ),
    Pair(
        "same_screen_clock_tick",
        "same",
        "One screen a minute later: the clock ticked and the URL grew a cache-busting query.",
        Side("invoices.html", "https://invoices.test/invoices"),
        Side(
            "invoices.html",
            "https://invoices.test/invoices?t=1738&utm_source=x",
            script=_TICK_CLOCK,
        ),
    ),
    Pair(
        "same_screen_scrolled",
        "same",
        "One screen scrolled down six logical pixels, as a hover or a focus ring can cause.",
        Side("invoices.html", "https://invoices.test/invoices"),
        Side("invoices.html", "https://invoices.test/invoices", scroll=6),
    ),
    Pair(
        "same_screen_retina",
        "same",
        "One screen captured on a 1x display and on a 2x Retina display: pure antialiasing.",
        Side("invoices.html", "https://invoices.test/invoices"),
        Side("invoices.html", "https://invoices.test/invoices", scale=2.0),
    ),
    Pair(
        "same_layout_different_content",
        "different",
        "The invoice list for two different accounts: identical layout, every row different.",
        Side("invoices.html", "https://invoices.test/invoices"),
        Side("invoices_alt.html", "https://invoices.test/invoices"),
    ),
    Pair(
        "modal_open_vs_closed",
        "different",
        "The same page with the confirm-payment dialog open: a real state change.",
        Side("invoices.html", "https://invoices.test/invoices"),
        Side("invoices_modal.html", "https://invoices.test/invoices"),
    ),
    Pair(
        "different_screens",
        "different",
        "The invoice list against the billing settings page: different layout, different URL.",
        Side("invoices.html", "https://invoices.test/invoices"),
        Side("settings.html", "https://invoices.test/settings"),
    ),
    Pair(
        "dense_text_scrolled_slightly",
        "same",
        "Wall-to-wall body text scrolled 8px - the hardest case there is for a pixel hash.",
        Side("article.html", "https://invoices.test/handbook/closing-the-books"),
        Side("article.html", "https://invoices.test/handbook/closing-the-books", scroll=8),
    ),
    Pair(
        "dense_text_different_article",
        "different",
        "Two handbook articles: one template, one typography, entirely different prose.",
        Side("article.html", "https://invoices.test/handbook/closing-the-books"),
        Side("article_alt.html", "https://invoices.test/handbook/servicing-the-assembly"),
    ),
    Pair(
        "same_screen_no_url",
        "same",
        "A desktop controller has no URL: the pair must still match on pixels and layout.",
        Side("invoices.html", None),
        Side("invoices.html", None, script=_HIDE_CARET),
    ),
)


def _capture(page: Page, side: Side, png_path: Path) -> dict:
    page.goto((PAGES / side.page).as_uri())
    if side.script:
        page.evaluate(side.script)
    page.evaluate(f"window.scrollTo(0, {side.scroll})")
    page.wait_for_timeout(60)
    png_path.write_bytes(page.screenshot())
    elements = page.evaluate(_ELEMENTS_JS, {"w": WIDTH, "h": HEIGHT})
    return {
        "png": png_path.name,
        "url": side.url,
        "width": WIDTH,
        "height": HEIGHT,
        "scale": side.scale,
        "elements": elements,
    }


def main() -> None:
    with sync_playwright() as driver:
        browser = driver.chromium.launch()
        try:
            for pair in PAIRS:
                out = HERE / pair.name
                out.mkdir(exist_ok=True)
                sides = {}
                for key, side in (("a", pair.a), ("b", pair.b)):
                    context = browser.new_context(
                        viewport={"width": WIDTH, "height": HEIGHT},
                        device_scale_factor=side.scale,
                    )
                    sides[key] = _capture(context.new_page(), side, out / f"{key}.png")
                    context.close()
                meta = {"name": pair.name, "label": pair.label, "why": pair.why, **sides}
                (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
                print(f"wrote {pair.name} ({pair.label})")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
