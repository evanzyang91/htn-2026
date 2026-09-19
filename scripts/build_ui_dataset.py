#!/usr/bin/env python3
"""Harvest a YOLO training set for the UI element detector, with no hand labelling.

The trick that makes this tractable in a hackathon is that the labels are free:
``BrowserGroundTruth`` reads the DOM and hands back exactly the boxes and kinds a
detector is supposed to learn, in the same logical-pixel space the agent clicks in.
So this script drives the sandbox app through a list of states, screenshots each
one, and writes the truth next to the pixels. Nobody draws a rectangle by hand.

``BrowserGroundTruth`` is an OFFLINE TEACHER. It is used here and in evaluation and
nowhere near the agent; that separation is the point of the project.

What varies, and why
--------------------
A detector trained on one screen learns that screen. Four axes of variation keep
the set honest:

* **App state** - :data:`RECIPES` drive the sandbox through menus, dialogs, empty
  filters, selections and banners, so controls move, appear and change enabled
  look. Each recipe is a list of ``/api/act`` calls, applied server-side and then
  loaded, which is far faster and more reproducible than clicking through the UI.
* **Real pages** - :data:`WEB_PAGES` are live public Wikipedia URLs, labelled by
  exactly the same DOM teacher. A model trained on the sandbox alone learns the
  sandbox: measured mean confidence was 0.89 there and 0.39 on an unseen Wikipedia
  page. Real frames are the fix, and the sandbox frames stay in so the cure is not
  a second overfit.
* **Viewport** - :data:`VIEWPORTS` and :data:`WEB_VIEWPORTS` change the layout: the
  mail list reflows, and Wikipedia swaps between its wide, narrow and mobile skins.
* **Scroll** - long surfaces are captured at more than one offset, so an element
  is seen near the top, the middle and the bottom of a frame.

One viewport of each kind is held out for validation, so the reported metric is
"unseen layout width", not "frames the model has already memorized". Stronger than
that: :data:`HELDOUT_PAGES` names real pages - including a site that appears
nowhere in :data:`WEB_PAGES` - that are NEVER harvested into the dataset. They
exist only for ``--bench`` and for the committed test fixtures, so "it generalizes"
is a number somebody measured rather than a hope.

Run it::

    uv run python scripts/build_ui_dataset.py --out data/models/ui-dataset
    uv run python scripts/build_ui_dataset.py --bench --weights a.pt --weights b.pt

The first starts and stops its own copy of the sandbox server unless ``--url``
names one, and reaches the public internet for the real pages (``--no-web`` skips
them). The second scores one or more weight files on held-out frames and prints
the before-and-after table, which is the point of the whole exercise.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:  # allow `python scripts/...` without an install
    sys.path.insert(0, str(REPO_ROOT / "src"))

from skillweaver.config import settings  # noqa: E402
from skillweaver.contracts import (  # noqa: E402
    Box,
    Element,
    ElementKind,
    ElementSource,
    Navigate,
    Point,
    Screenshot,
    Scroll,
    Wait,
)
from skillweaver.controllers.browser import BrowserController, BrowserGroundTruth  # noqa: E402
from skillweaver.perception.detect_yolo import DEFAULT_WEIGHTS_NAME  # noqa: E402
from skillweaver.perception.labeling import (  # noqa: E402
    CLASS_NAMES,
    dataset_yaml,
    recall_at_iou,
    to_label_text,
)

SANDBOX = REPO_ROOT / "apps" / "sandbox-site" / "serve.py"

Act = tuple[str, dict]


@dataclass(frozen=True)
class Recipe:
    """One app state to photograph.

    Attributes:
        name: Goes into the file name, so a bad label can be traced back to a state.
        acts: ``(action, payload)`` pairs posted to ``/api/act`` in order, starting
            from a freshly reset app.
        scrolls: Vertical wheel offsets in logical pixels to capture at, ``0``
            meaning the top of the page. Every recipe is captured at every offset.
    """

    name: str
    acts: tuple[Act, ...] = ()
    scrolls: tuple[int, ...] = (0,)


_MAIL_IDS = ("m01", "m02", "m03", "m04", "m05", "m06")
_ROW_IDS = tuple(f"r{n:02d}" for n in range(1, 41))

RECIPES: tuple[Recipe, ...] = (
    # -- mail: sidebar, list, reading pane, compose, menus --------------------------
    Recipe("mail_inbox", (), (0, 240)),
    Recipe("mail_search_hit", (("mail.search", {"q": "invoice"}),)),
    Recipe("mail_search_empty", (("mail.search", {"q": "zzzz"}),)),
    Recipe(
        "mail_selected",
        (("mail.toggleSelect", {"id": "m01"}), ("mail.toggleSelect", {"id": "m03"})),
        (0, 200),
    ),
    Recipe("mail_select_all", (("mail.selectAll", {"ids": list(_MAIL_IDS), "checked": True}),)),
    Recipe(
        "mail_label_menu",
        (("mail.toggleSelect", {"id": "m02"}), ("mail.toggleLabelMenu", {})),
    ),
    Recipe("mail_reading", (("mail.open", {"id": "m01"}),), (0, 180)),
    Recipe("mail_reading_deep", (("mail.open", {"id": "m05"}),), (0, 320)),
    Recipe("mail_compose_blank", (("mail.composeOpen", {}),)),
    Recipe(
        "mail_compose_filled",
        (
            ("mail.composeOpen", {}),
            ("mail.composeField", {"field": "to", "value": "dana.whitfield@northwind.example"}),
            ("mail.composeField", {"field": "subject", "value": "Re: capacity plan"}),
            ("mail.composeField", {"field": "body", "value": "Approved - go ahead and lock it."}),
        ),
    ),
    Recipe(
        "mail_banner",
        (("mail.toggleSelect", {"id": "m04"}), ("mail.archive", {})),
    ),
    Recipe("mail_archived", (("mail.folder", {"folder": "archived"}),)),
    Recipe(
        "mail_sent",
        (
            ("mail.composeOpen", {}),
            ("mail.composeField", {"field": "to", "value": "priya.raman@northwind.example"}),
            ("mail.composeField", {"field": "subject", "value": "Checklist looks good"}),
            ("mail.send", {}),
            ("mail.folder", {"folder": "sent"}),
        ),
    ),
    # -- records: a dense table, sorting, inline edit, bulk menu --------------------
    Recipe("rec_default", (("nav", {"screen": "records"}),), (0, 400, 900)),
    Recipe(
        "rec_filtered",
        (("nav", {"screen": "records"}), ("records.filter", {"q": "north"})),
    ),
    Recipe(
        "rec_filter_empty",
        (("nav", {"screen": "records"}), ("records.filter", {"q": "qqqq"})),
    ),
    Recipe(
        "rec_sorted",
        (
            ("nav", {"screen": "records"}),
            ("records.sort", {"key": "status"}),
            ("records.sort", {"key": "status"}),
        ),
        (0, 520),
    ),
    Recipe(
        "rec_sorted_priority",
        (("nav", {"screen": "records"}), ("records.sort", {"key": "priority"})),
        (0, 700),
    ),
    Recipe(
        "rec_selected",
        (
            ("nav", {"screen": "records"}),
            ("records.toggleSelect", {"id": "r02"}),
            ("records.toggleSelect", {"id": "r05"}),
            ("records.toggleSelect", {"id": "r09"}),
        ),
        (0, 360),
    ),
    Recipe(
        "rec_bulk_menu",
        (
            ("nav", {"screen": "records"}),
            ("records.toggleSelect", {"id": "r03"}),
            ("records.toggleBulkMenu", {}),
        ),
    ),
    Recipe(
        "rec_select_all",
        (
            ("nav", {"screen": "records"}),
            ("records.selectAll", {"ids": list(_ROW_IDS), "checked": True}),
        ),
        (0, 640),
    ),
    Recipe(
        "rec_editing",
        (("nav", {"screen": "records"}), ("records.editStart", {"id": "r04"})),
    ),
    Recipe(
        "rec_banner",
        (
            ("nav", {"screen": "records"}),
            ("records.export", {"ids": list(_ROW_IDS[:6])}),
        ),
        (0, 300),
    ),
    # -- settings: cards, selects, toggles, a modal dialog --------------------------
    Recipe("set_default", (("nav", {"screen": "settings"}),), (0, 260)),
    Recipe(
        "set_edited",
        (
            ("nav", {"screen": "settings"}),
            ("settings.field", {"field": "displayName", "value": "Avery Q."}),
            ("settings.field", {"field": "density", "value": "compact"}),
            ("settings.field", {"field": "notifyDesktop", "value": True}),
            ("settings.field", {"field": "autoArchive", "value": True}),
        ),
        (0, 260),
    ),
    Recipe(
        "set_dialog",
        (
            ("nav", {"screen": "settings"}),
            ("settings.field", {"field": "timezone", "value": "Europe/Berlin"}),
            ("settings.save", {}),
        ),
    ),
    Recipe(
        "set_saved",
        (
            ("nav", {"screen": "settings"}),
            ("settings.field", {"field": "weeklyDigest", "value": False}),
            ("settings.save", {}),
            ("settings.confirm", {}),
        ),
    ),
)
"""Every app state the dataset covers: 27 states across the three surfaces."""


@dataclass(frozen=True)
class Viewport:
    """One browser geometry, and whether its frames go to train or validation."""

    width: int
    height: int
    scale: float = 1.0
    split: str = "train"

    @property
    def tag(self) -> str:
        return f"{self.width}x{self.height}@{self.scale:g}x"


VIEWPORTS: tuple[Viewport, ...] = (
    Viewport(1280, 800),
    Viewport(1024, 768),
    Viewport(1440, 900),
    Viewport(1120, 700, scale=2.0),
    Viewport(1600, 1000),
    Viewport(1180, 860, split="val"),
)
"""Layouts to photograph. One is held out for validation, and one trains at 2x so
the model sees Retina-sharp pixels as well as ordinary ones - normalized labels are
identical either way, which is what makes mixing them free."""


# --------------------------------------------------------------------------------------
# Real pages
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WebPage:
    """One live public page to photograph, labelled by the same DOM teacher.

    Attributes:
        name: Goes into the file name, so a bad label can be traced back to a URL.
        url: Loaded directly. Nothing is clicked: a URL reaches a Wikipedia search
            result or a page history far more reliably than a click sequence, and
            a harvest that depends on the network should depend on it as little as
            possible.
        scrolls: Vertical wheel offsets in logical pixels, as for :class:`Recipe`.
        settle_ms: Extra quiet time after load and after each scroll. Real pages
            lazy-load images and swap in scripts long after ``load`` fires, and a
            screenshot taken mid-reflow is labelled against a DOM that has already
            moved on.
    """

    name: str
    url: str
    scrolls: tuple[int, ...] = (0,)
    settle_ms: int = 700


_WIKI = "https://en.wikipedia.org"
_WIKI_MOBILE = "https://en.m.wikipedia.org"


def _wiki_search(query: str, *, mobile: bool = False) -> str:
    """The URL Wikipedia's own search box submits to, which is a different layout
    from an article and from the empty search form."""
    host = _WIKI_MOBILE if mobile else _WIKI
    return f"{host}/w/index.php?search={query}&title=Special%3ASearch&fulltext=1&ns0=1"


WIKIPEDIA_PAGES: tuple[WebPage, ...] = (
    # -- the three layouts a search-and-navigate task actually walks through ---------
    WebPage("wiki_main", f"{_WIKI}/wiki/Main_Page", (0, 600, 1400)),
    WebPage("wiki_search_form", f"{_WIKI}/wiki/Special:Search"),
    WebPage("wiki_results_ml", _wiki_search("machine+learning"), (0, 600)),
    WebPage("wiki_results_coffee", _wiki_search("coffee"), (0, 500)),
    # -- articles: the dense case, where the detector was measured worst -------------
    WebPage("wiki_article_city", f"{_WIKI}/wiki/Toronto", (0, 900, 2200)),
    WebPage("wiki_article_tech", f"{_WIKI}/wiki/Python_(programming_language)", (0, 700, 1800)),
    WebPage("wiki_article_town", f"{_WIKI}/wiki/Waterloo,_Ontario", (0, 1100)),
    # -- the chrome around articles: tables, lists, forms, diffs ---------------------
    WebPage("wiki_category", f"{_WIKI}/wiki/Category:Cities_in_Ontario", (0, 700)),
    WebPage("wiki_history", f"{_WIKI}/w/index.php?title=Toronto&action=history", (0, 600)),
    WebPage("wiki_portal", f"{_WIKI}/wiki/Portal:Current_events", (0, 800)),
    # -- the mobile skin, which is a different DOM, not just a narrower one ----------
    WebPage("wiki_mobile_article", f"{_WIKI_MOBILE}/wiki/Toronto", (0, 700)),
    WebPage("wiki_mobile_results", _wiki_search("coffee", mobile=True), (0, 500)),
)
"""Wikipedia, in every layout a search-and-open task crosses. The site the detector
is being optimized for, and the bulk of the real frames."""


OTHER_SITE_PAGES: tuple[WebPage, ...] = (
    WebPage("mdn_reference", "https://developer.mozilla.org/en-US/docs/Web/CSS/Cascade", (0, 900)),
    WebPage("mdn_home", "https://developer.mozilla.org/en-US/", (0, 700)),
    WebPage("arxiv_abstract", "https://arxiv.org/abs/1706.03762", (0, 500)),
    WebPage("debian_home", "https://www.debian.org/", (0, 600)),
    WebPage("iana_home", "https://www.iana.org/"),
    WebPage("w3c_home", "https://www.w3.org/", (0, 600)),
)
"""Seven pages from six sites that are nothing like Wikipedia, and nothing like each
other: a modern documentation site, an academic abstract, three project homepages
from three decades of web design, and a standards body.

One site that was tried and dropped: ``gnu.org`` answered the first request and
then timed out for every one after it, which is a harvest paying 30 seconds a
viewport for nothing. :func:`load_page` skipping it rather than aborting the run is
why the rest of that harvest survived.

They are here because training on Wikipedia alone reproduced the original bug one
level up. Measured on frames from a site in no split at all, the sandbox-only
weights scored 0.007 recall and the sandbox-plus-Wikipedia weights scored 0.057:
the detector had simply learned a second site instead of learning the web. Six
sites will not make it universal either, but "it has seen more than one visual
language" is a different claim from "it has seen one", and
:data:`HELDOUT_PAGES` is what says which one is true today.

``arxiv_abstract`` earns its place twice over: it is the only page anywhere in the
dataset with real ``tab`` elements on it."""


WEB_PAGES: tuple[WebPage, ...] = WIKIPEDIA_PAGES + OTHER_SITE_PAGES
"""Every live page harvested into the dataset."""


HELDOUT_PAGES: tuple[WebPage, ...] = (
    WebPage("wiki_heldout_article", f"{_WIKI}/wiki/Great_Barrier_Reef", (0, 1200)),
    WebPage("wiki_heldout_results", _wiki_search("hack+the+north"), (0,)),
    WebPage("sqlite_home", "https://sqlite.org/index.html", (0,)),
    WebPage("postgres_home", "https://www.postgresql.org/", (0,)),
    WebPage("python_org", "https://www.python.org/", (0, 700)),
    WebPage("hacker_news", "https://news.ycombinator.com/", (0,)),
)
"""Pages that NEVER enter the dataset, in two flavours of unseen.

``wiki_heldout_*`` are Wikipedia pages whose *URLs* are absent from
:data:`WEB_PAGES`, so they answer "does it work on an article it has not
memorized". The other four are whole SITES absent from the dataset, so they answer
the harder question: "did it learn the web, or just learn the sites in
:data:`WEB_PAGES`".

Those four are deliberately not all of a kind, because the first version of this
tuple held only ``python_org`` and ``hacker_news`` and those two turned out to be
the two hardest pages on the list - which made one bleak average out of two
different problems. ``sqlite_home`` and ``postgres_home`` are ordinary light
documentation sites, the normal case. ``python_org`` is light-on-DARK, which the
training set contains almost none of. ``hacker_news`` is a 2007 table layout of
eight-point text, three hundred elements to a screen. Scoring them separately is
how their very different numbers stay visible; ``README.md`` in
``tests/perception/fixtures`` has them frame by frame.

Every one of these questions only means something if nothing here is ever
harvested. That is why they live in their own tuple rather than behind a flag on
:data:`WEB_PAGES`."""


WEB_VIEWPORTS: tuple[Viewport, ...] = (
    Viewport(1280, 800),
    Viewport(1024, 768),
    Viewport(1440, 900, scale=2.0),
    Viewport(820, 1000),
    Viewport(1180, 860, split="val"),
)
"""Geometries for the real pages, held-out validation width included, matching the
sandbox split so a frame never lands in train for one source and val for the other.
The 820-wide one matters most: Wikipedia switches skins below roughly 1000 CSS
pixels, so it is a different page rather than the same page squeezed."""

BENCH_VIEWPORT = Viewport(1200, 780)
"""Where ``--bench`` measures: a width in neither split, so every number it prints
is about a layout the model was not trained on."""

MIN_SIDE = 4
"""Boxes thinner than this in logical pixels are noise: hairline rules, spacers."""

MAX_AREA_FRACTION = 0.75
"""Anything covering more of the frame than this is a layout wrapper, not a control.
Training on it teaches the model to predict the whole screen."""


# --------------------------------------------------------------------------------------
# The sandbox app
# --------------------------------------------------------------------------------------


class Sandbox:
    """The sandbox site: either one this script started, or one already running."""

    def __init__(self, url: str, process: subprocess.Popen | None = None) -> None:
        self.url = url.rstrip("/")
        self._process = process

    @classmethod
    def start(cls, port: int, *, timeout_s: float = 20.0) -> Sandbox:
        """Launch ``serve.py`` on ``port`` and wait until it answers.

        Raises:
            RuntimeError: if the server does not come up in time.
        """
        process = subprocess.Popen(
            [sys.executable, str(SANDBOX), "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"sandbox server exited with code {process.returncode}")
            try:
                urllib.request.urlopen(f"{url}/__state", timeout=1.0).read()
                return cls(url, process)
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(0.1)
        process.terminate()
        raise RuntimeError(f"sandbox server did not start on {url} within {timeout_s}s")

    def reset(self) -> None:
        urllib.request.urlopen(f"{self.url}/__reset", timeout=5.0).read()

    def act(self, action: str, payload: dict) -> None:
        body = json.dumps({"action": action, "payload": payload}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/api/act", data=body, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(request, timeout=5.0).read()

    def close(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - a wedged child
            self._process.kill()
        self._process = None


# --------------------------------------------------------------------------------------
# Filling a gap in the teacher
# --------------------------------------------------------------------------------------

_HIDDEN_CONTROL_JS = """
() => {
  const vw = window.innerWidth, vh = window.innerHeight;
  const out = [];
  for (const el of document.querySelectorAll('input[type="checkbox"], input[type="radio"]')) {
    const style = getComputedStyle(el);
    if (style.display === 'none' || style.visibility !== 'visible') continue;
    // Only the ones BrowserGroundTruth drops: a painted control it can see needs
    // no help from here, and picking it up twice would double-label it.
    if (parseFloat(style.opacity || '1') > 0.01) continue;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) continue;
    if (rect.right <= 0 || rect.bottom <= 0 || rect.left >= vw || rect.top >= vh) continue;
    out.push({
      kind: el.getAttribute('type').toLowerCase() === 'radio' ? 'radio' : 'checkbox',
      text: (el.getAttribute('aria-label') || '').trim(),
      x: rect.left, y: rect.top, w: rect.width, h: rect.height,
    });
  }
  return out;
}
"""


def hidden_controls(controller: BrowserController) -> list[Element]:
    """Recover checkboxes and toggles the DOM teacher cannot see.

    ``BrowserGroundTruth`` skips anything with ``opacity <= 0.01``, which is right
    for a fading dialog and wrong for the most common checkbox pattern on the web:
    a real ``<input type=checkbox>`` at ``opacity: 0`` laid exactly over a styled
    ``<span>`` that does the painting. The sandbox app uses it for every row
    checkbox and every settings toggle (``apps/sandbox-site/static/js/app.js``), so
    without this the teacher reports zero checkboxes on screens that visibly have a
    dozen, and a detector trained on it could never learn one.

    The input's own rectangle IS the visible control's rectangle in that pattern -
    the proxy is positioned on top of it - so the recovered box needs no adjustment.

    This belongs in ``BrowserGroundTruth``. ``browser.py`` is owned by another piece
    of work, so the gap is reported rather than edited and this script carries the
    workaround meanwhile; reaching for the controller's page is the price, and it
    stays confined to this offline dataset builder.
    """
    page = controller._live_page()  # noqa: SLF001 - see the docstring above
    found: list[Element] = []
    for raw in page.evaluate(_HIDDEN_CONTROL_JS):
        found.append(
            Element(
                box=Box(x=round(raw["x"]), y=round(raw["y"]), w=round(raw["w"]), h=round(raw["h"])),
                kind=ElementKind(raw["kind"]),
                text=raw["text"],
                confidence=1.0,
                source=ElementSource.dom,
            )
        )
    return found


# --------------------------------------------------------------------------------------
# Harvesting
# --------------------------------------------------------------------------------------


@dataclass
class Counts:
    """What the run produced, for the summary line and for a sanity check."""

    frames: int = 0
    labels: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    per_split: dict[str, int] = field(default_factory=dict)

    def add(self, split: str, elements: Sequence[Element]) -> None:
        self.frames += 1
        self.labels += len(elements)
        self.per_split[split] = self.per_split.get(split, 0) + 1
        for element in elements:
            self.per_class[element.kind.value] = self.per_class.get(element.kind.value, 0) + 1


def usable_labels(elements: Sequence[Element], viewport: Box) -> list[Element]:
    """Drop the ground-truth boxes a detector should not be asked to learn.

    Three filters, in order of how much damage they would do: sub-pixel slivers
    (a hairline border reported as an element), page-sized wrappers (training on
    "the answer is the whole screen" poisons every other box), and exact duplicates
    where a control and its inner label share a rectangle.
    """
    limit = viewport.area * MAX_AREA_FRACTION
    seen: set[tuple[str, int, int, int, int]] = set()
    kept: list[Element] = []
    for element in elements:
        box = element.box
        if box.w < MIN_SIDE or box.h < MIN_SIDE or box.area > limit:
            continue
        key = (element.kind.value, box.x, box.y, box.w, box.h)
        if key in seen:
            continue
        seen.add(key)
        kept.append(element)
    return kept


Frame = tuple[str, Screenshot, list[Element]]
"""``(name, screenshot, labels)`` - one photographed screen with its ground truth."""


def browser_for(viewport: Viewport, *, headless: bool) -> BrowserController:
    """One browser at ``viewport``'s geometry.

    One browser per viewport, not per frame: Playwright fixes a context's size at
    creation, and re-launching per frame would dominate the runtime.
    """
    return BrowserController(
        headless=headless,
        viewport=(viewport.width, viewport.height),
        device_scale_factor=viewport.scale,
    )


def scrolled_frames(
    controller: BrowserController,
    viewport: Viewport,
    stem: str,
    scrolls: Sequence[int],
    *,
    settle_ms: int = 0,
) -> Iterator[Frame]:
    """Photograph the already-loaded page at each scroll offset, in order.

    The offsets are absolute positions from the top, so they are walked as deltas
    and the caller does not have to think about where the last one left the page.
    Truth is read immediately before the pixels are taken: on a real page those two
    can disagree if anything moves between them, and the closer together they are
    the less often that happens.
    """
    truth = BrowserGroundTruth(controller)
    bounds = Box(0, 0, viewport.width, viewport.height)
    middle = Point(viewport.width // 2, viewport.height // 2)
    offset = 0
    for target in scrolls:
        if target != offset:
            controller.perform(Scroll(middle, dx=0, dy=target - offset))
            offset = target
            if settle_ms:
                controller.perform(Wait(settle_ms))
        labels = truth.elements() + hidden_controls(controller)
        yield (
            f"{stem}__{viewport.tag}__y{target}",
            controller.capture(),
            usable_labels(labels, bounds),
        )


def frames(
    sandbox: Sandbox,
    viewport: Viewport,
    *,
    headless: bool,
    recipes: Sequence[Recipe] = RECIPES,
) -> Iterator[Frame]:
    """Yield a frame for every sandbox recipe and scroll offset."""
    with browser_for(viewport, headless=headless) as controller:
        for recipe in recipes:
            sandbox.reset()
            for action, payload in recipe.acts:
                sandbox.act(action, payload)
            result = controller.perform(Navigate(sandbox.url))
            if not result.ok:
                raise RuntimeError(f"could not load {sandbox.url}: {result.error}")
            yield from scrolled_frames(controller, viewport, recipe.name, recipe.scrolls)


def load_page(controller: BrowserController, page: WebPage) -> bool:
    """Load a real page, once more if the network hiccups. ``False`` if it will not
    come up.

    A harvest across a dozen live URLs and five viewports is sixty chances for a
    transient failure, and losing twenty minutes of frames to one of them would be
    absurd - so a page that refuses twice is skipped loudly and the run goes on.
    """
    for attempt in (1, 2):
        result = controller.perform(Navigate(page.url))
        if result.ok:
            controller.perform(Wait(page.settle_ms))
            return True
        print(f"    {page.name}: load failed ({result.error}); attempt {attempt}", flush=True)
        controller.perform(Wait(1500))
    return False


def web_frames(
    viewport: Viewport,
    *,
    headless: bool,
    pages: Sequence[WebPage] = WEB_PAGES,
) -> Iterator[Frame]:
    """Yield a frame for every real page and scroll offset.

    The teacher is the same ``BrowserGroundTruth`` the sandbox uses - that is the
    whole point, since it means a real page costs nothing more to label than a
    synthetic one.
    """
    with browser_for(viewport, headless=headless) as controller:
        for page in pages:
            if not load_page(controller, page):
                continue
            yield from scrolled_frames(
                controller, viewport, page.name, page.scrolls, settle_ms=page.settle_ms
            )


def write_frame(
    root: Path, split: str, name: str, screenshot: Screenshot, elements: Sequence[Element]
) -> None:
    """Write one image and its label file into the ultralytics directory layout.

    The PNG is saved at its native (physical) resolution, and the labels are
    normalized against the LOGICAL size. Those are the same numbers - normalization
    divides the scale out - so a 2x frame needs no special case anywhere.
    """
    (root / "images" / split / f"{name}.png").write_bytes(screenshot.png)
    (root / "labels" / split / f"{name}.txt").write_text(
        to_label_text(elements, screenshot.width, screenshot.height), encoding="utf-8"
    )


def build(
    out: Path,
    sandbox: Sandbox | None,
    *,
    headless: bool,
    viewports: Sequence[Viewport],
    web_viewports: Sequence[Viewport] = (),
    pages: Sequence[WebPage] = WEB_PAGES,
) -> Counts:
    """Harvest every viewport into a fresh dataset directory under ``out``.

    Sandbox frames first because they need no network, then the real pages, into
    the same two splits - ultralytics sees one dataset and cannot tell which frame
    came from where, which is the point: the detector has to be good at both.
    """
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    counts = Counts()

    def harvest(label: str, viewport: Viewport, stream: Iterator[Frame]) -> None:
        started = time.monotonic()
        before = counts.frames
        for name, screenshot, elements in stream:
            write_frame(out, viewport.split, name, screenshot, elements)
            counts.add(viewport.split, elements)
        print(
            f"  {label:<8} {viewport.tag:>16} -> {viewport.split:<5} "
            f"{counts.frames - before:3d} frames in {time.monotonic() - started:5.1f}s",
            flush=True,
        )

    for viewport in viewports:
        assert sandbox is not None  # --no-sandbox leaves `viewports` empty
        harvest("sandbox", viewport, frames(sandbox, viewport, headless=headless))
    for viewport in web_viewports:
        harvest("web", viewport, web_frames(viewport, headless=headless, pages=pages))

    (out / "data.yaml").write_text(dataset_yaml(str(out.resolve())), encoding="utf-8")
    return counts


# --------------------------------------------------------------------------------------
# Measuring a detector
# --------------------------------------------------------------------------------------


BENCH_RECIPES: tuple[str, ...] = (
    "mail_inbox",
    "mail_compose_filled",
    "rec_default",
    "rec_selected",
    "set_dialog",
)
"""Sandbox states the bench scores, one per surface plus the two densest."""


@dataclass(frozen=True)
class Score:
    """What one set of weights did on one group of frames."""

    frames: int
    truth: int
    detections: int
    mean_confidence: float
    recall: float

    @property
    def per_frame(self) -> float:
        return self.detections / max(self.frames, 1)


def score_frames(detector, group: Sequence[Frame]) -> Score:  # noqa: ANN001 - a contracts.Detector
    """Mean confidence, detection count and pooled recall over ``group``.

    Recall is pooled by element, not averaged by frame, so a page with four
    controls cannot outvote a page with two hundred. Mean confidence is over every
    detection returned above the detector's own threshold - it is the number that
    was measured at 0.89 on the sandbox and 0.39 on Wikipedia, so it is reported
    the same way here even though recall is the metric that decides whether the
    agent can click anything.
    """
    detections = 0
    confidence = 0.0
    truth = 0
    hits = 0.0
    for _, screenshot, expected in group:
        found = detector.detect(screenshot)
        detections += len(found)
        confidence += sum(element.confidence for element in found)
        truth += len(expected)
        hits += recall_at_iou(expected, found, iou=0.5) * len(expected)
    return Score(
        frames=len(group),
        truth=truth,
        detections=detections,
        mean_confidence=confidence / max(detections, 1),
        recall=hits / max(truth, 1),
    )


def bench_groups(sandbox: Sandbox | None, *, headless: bool) -> dict[str, list[Frame]]:
    """Capture the held-out frames every weights file is scored on.

    Captured ONCE and reused for every ``--weights``: a live page changes between
    two harvests, and a before-and-after table where the two rows saw different
    pixels is worse than no table at all.
    """
    by_name = {recipe.name: replace(recipe, scrolls=(0,)) for recipe in RECIPES}
    groups: dict[str, list[Frame]] = {}
    if sandbox is not None:
        groups["sandbox (trained on)"] = list(
            frames(
                sandbox,
                BENCH_VIEWPORT,
                headless=headless,
                recipes=tuple(by_name[name] for name in BENCH_RECIPES),
            )
        )
    wiki = tuple(page for page in HELDOUT_PAGES if page.name.startswith("wiki_"))
    elsewhere = tuple(page for page in HELDOUT_PAGES if not page.name.startswith("wiki_"))
    if wiki:
        groups["wikipedia (unseen pages)"] = list(
            web_frames(BENCH_VIEWPORT, headless=headless, pages=wiki)
        )
    if elsewhere:
        groups["unseen sites"] = list(
            web_frames(BENCH_VIEWPORT, headless=headless, pages=elsewhere)
        )
    return groups


def bench(weights: Sequence[Path], groups: dict[str, list[Frame]]) -> dict[str, dict[str, Score]]:
    """Score every weights file on every group and print the comparison table.

    Every frame gets its own row as well as its group's average, because on real
    pages the spread is the finding: a group average of 0.05 hid one ordinary site
    scoring 0.34 and one dark-themed site scoring 0.10, which are two different
    problems with two different fixes.
    """
    from skillweaver.perception.detect_yolo import YoloDetector

    detectors = {path.name: YoloDetector(path) for path in weights}
    table: dict[str, dict[str, Score]] = {
        name: {label: score_frames(detector, group) for label, group in groups.items()}
        for name, detector in detectors.items()
    }

    rows = max(
        (len(name.rsplit("__", 2)[0]) for group in groups.values() for name, _, _ in group),
        default=0,
    )
    width = max(max((len(label) for label in groups), default=10), rows + 2)
    print(f"\n{'':<{width}}  {'frames':>6} {'truth':>6}", end="")
    for path in weights:
        print(f"   {path.name:>22}", end="")
    print(f"\n{'':<{width}}  {'':>6} {'':>6}", end="")
    for _ in weights:
        print(f"   {'found':>6} {'conf':>6} {'recall':>6}", end="")
    print()
    for label, group in groups.items():
        for name, screenshot, expected in group:
            print(f"  {name.rsplit('__', 2)[0]:<{width - 2}}  {'':>6} {len(expected):>6}", end="")
            for path in weights:
                one = score_frames(detectors[path.name], [(name, screenshot, expected)])
                print(
                    f"   {one.detections:>6} {one.mean_confidence:>6.3f} {one.recall:>6.3f}",
                    end="",
                )
            print()
        first = table[weights[0].name][label]
        print(f"{label:<{width}}  {first.frames:>6} {first.truth:>6}", end="")
        for path in weights:
            got = table[path.name][label]
            print(
                f"   {got.detections:>6} {got.mean_confidence:>6.3f} {got.recall:>6.3f}",
                end="",
            )
        print()
    return table


# --------------------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------------------

FIXTURE_VIEWPORTS: tuple[Viewport, ...] = (Viewport(1200, 780), Viewport(1200, 780, scale=2.0))
"""A geometry that appears in neither the training nor the validation split, at both
device scales. Frames captured here are what the committed detector test scores
against, so its recall floor is a claim about an unseen layout, not about a screen
the model has already been shown."""

FIXTURE_FRAMES: tuple[tuple[str, str, int], ...] = (
    ("mail_inbox", "mail_inbox", 0),
    ("rec_selected", "rec_selected", 0),
    ("set_dialog", "set_dialog", 1),
)
"""``(fixture name, recipe name, index into FIXTURE_VIEWPORTS)`` - one frame per
surface, and one of them at 2x so the test exercises the scale conversion."""

FIXTURE_WEB_FRAMES: tuple[tuple[str, str], ...] = (
    ("wiki_article", "wiki_heldout_article"),
    ("wiki_results", "wiki_heldout_results"),
    ("sqlite_home", "sqlite_home"),
    ("python_org", "python_org"),
)
"""``(fixture name, HELDOUT_PAGES name)`` - the real-page half of the fixtures.

Two Wikipedia layouts whose URLs are not in :data:`WEB_PAGES`, and two pages from
sites that are not in the dataset at all: an ordinary light one and the dark-themed
one that is the worst case. Captured at ``FIXTURE_VIEWPORTS[0]``, and
committed, so the generalization claim keeps being checked by a test that needs no
network. They are photographs of live pages, so regenerating them will not be
byte-identical and the floors they pin have to be remeasured rather than assumed."""

FIXTURE_README = """\
# Detector test fixtures

Committed screenshots and their DOM ground truth, used by
`tests/perception/test_detect_yolo.py` to score the trained detector with no
network and no training step.

They come in two groups, and the test never averages them together.

**`sandbox`** - three frames of the sandbox app at a viewport (1200x780) that is in
neither split of the training dataset, one of them at device scale 2.0, which is
what keeps the test honest about the physical-to-logical conversion.

**`web`** - four photographs of live public pages, none of whose URLs are in the
training set, and two of whose SITES are not in it at all:

| fixture | page | seen in training? |
| --- | --- | --- |
| `wiki_article` | en.wikipedia.org article | the site, not this page |
| `wiki_results` | en.wikipedia.org search results | the site, not this query |
| `sqlite_home` | sqlite.org | no - site never trained on |
| `python_org` | python.org | no - site never trained on |

Labels are ordinary YOLO label files: class id from
`skillweaver.perception.labeling.CLASS_NAMES`, then `cx cy w h` normalized against
the LOGICAL frame size in `frames.json`.

## What the shipped weights measure here

Recall at 0.5 IoU, matching element kind, against the DOM ground truth captured
with each frame. `old` is the first version of these weights, trained on 240 frames
of the sandbox app and nothing else. `new` is what ships now: 425 frames of the
sandbox app, 12 live Wikipedia pages and 6 pages from 5 other sites.

| frame | group | elements | old | new |
| --- | --- | --- | --- | --- |
| `mail_inbox` (1x) | sandbox | 57 | 1.000 | 1.000 |
| `rec_selected` (1x) | sandbox | 185 | 0.941 | 0.957 |
| `set_dialog` (2x) | sandbox | 48 | 0.792 | 0.833 |
| `wiki_article` | web | 102 | 0.039 | 0.618 |
| `wiki_results` | web | 71 | 0.099 | 0.718 |
| `sqlite_home` | web | 61 | 0.016 | 0.377 |
| `python_org` | web | 63 | 0.032 | 0.095 |
| **sandbox, mean** | | | **0.911** | **0.930** |
| **web, mean** | | | **0.046** | **0.452** |
| **sandbox, interactive kinds pooled** | | 95 | **0.958** | **0.948** |
| **web, interactive kinds pooled** | | 180 | **0.028** | **0.583** |

The same frames by the two numbers the gap was originally reported in - mean
confidence over every detection returned, and how many elements are found per
frame:

| | old | new |
| --- | --- | --- |
| sandbox, mean confidence | 0.886 | 0.892 |
| sandbox, elements per frame | 91.7 | 92.7 |
| real pages, mean confidence | 0.499 | 0.648 |
| real pages, elements per frame | 30.5 | 54.0 |

## What is still wrong, stated plainly

**A dark page is close to a blind screen.** `python_org` is light text on a dark
blue field, and it goes from 0.032 to 0.095 - a threefold improvement of a number
that is still almost zero. Nearly every frame in the training set is dark-on-light,
so the detector has barely seen the other polarity. Frames from dark-themed pages,
or a colour-inversion augmentation over the frames already harvested, is the
obvious next thing to try and has not been tried.

**Hacker News is a total blind spot.** It is not a fixture because there is nothing
to pin: on its front page the detector finds 11 boxes where the DOM reports 309,
for a recall of 0.000, and that is true of all three sets of weights. Eight-point
text in a 2007 table layout is further from anything in the training set than any
amount of extra Wikipedia was going to fix. `--bench` keeps measuring it anyway.

**One site is not the web.** Six real sites is enough to move an unseen ordinary
site from 0.016 to 0.377 (`sqlite_home`) and from 0.000 to 0.559 (postgresql.org,
measured by `--bench`), which is real generalization and is also nowhere near
solved. The honest summary is: very good on the app it was trained on, good on
pages of a site it was trained on, partial on an unseen ordinary site, blind on
dark themes and on micro-dense layouts.

## Regenerating

    uv run python scripts/build_ui_dataset.py --fixtures tests/perception/fixtures

The sandbox frames are deterministic and come back byte-identical unless the app
itself changed. The web frames are photographs of live pages and will NOT: they
need the network, and the pinned floors have to be remeasured after a regeneration
rather than adjusted until the test passes.

To remeasure everything, including the pages too unstable or too large to commit:

    uv run python scripts/build_ui_dataset.py --bench \\
        --weights old.pt --weights data/models/ui_detector.pt
"""


def _write_fixture(
    out: Path,
    name: str,
    group: str,
    source: str,
    screenshot: Screenshot,
    elements: Sequence[Element],
) -> dict:
    """Write one fixture PNG and label file; return its manifest entry.

    ``group`` is ``"sandbox"`` or ``"web"``. The test reads it to score the two
    kinds of frame against their own floors, because a detector that finds 95% of
    a synthetic app and 45% of a real page has one number worth knowing and one
    number worth hiding behind an average.
    """
    (out / f"{name}.png").write_bytes(screenshot.png)
    (out / f"{name}.txt").write_text(
        to_label_text(elements, screenshot.width, screenshot.height), encoding="utf-8"
    )
    print(f"  {name:<16} {source:<44} {len(elements):4d} labels", flush=True)
    return {
        "image": f"{name}.png",
        "labels": f"{name}.txt",
        "group": group,
        "source": source,
        "width": screenshot.width,
        "height": screenshot.height,
        "scale": screenshot.scale,
        "elements": len(elements),
    }


def build_fixtures(out: Path, sandbox: Sandbox | None, *, headless: bool, web: bool = True) -> None:
    """Write the committed test frames: PNG, YOLO labels, a manifest and a README."""
    out.mkdir(parents=True, exist_ok=True)
    by_recipe = {recipe.name: recipe for recipe in RECIPES}
    manifest: list[dict] = []
    wanted = FIXTURE_FRAMES if sandbox is not None else ()
    for name, recipe_name, viewport_index in wanted:
        viewport = FIXTURE_VIEWPORTS[viewport_index]
        recipe = replace(by_recipe[recipe_name], scrolls=(0,))
        for _, screenshot, elements in frames(
            sandbox, viewport, headless=headless, recipes=(recipe,)
        ):
            manifest.append(_write_fixture(out, name, "sandbox", recipe_name, screenshot, elements))

    if web:
        by_page = {page.name: replace(page, scrolls=(0,)) for page in HELDOUT_PAGES}
        viewport = FIXTURE_VIEWPORTS[0]
        pages = tuple(by_page[page_name] for _, page_name in FIXTURE_WEB_FRAMES)
        names = {page_name: name for name, page_name in FIXTURE_WEB_FRAMES}
        for stem, screenshot, elements in web_frames(viewport, headless=headless, pages=pages):
            page_name = stem.split("__", 1)[0]
            manifest.append(
                _write_fixture(
                    out, names[page_name], "web", by_page[page_name].url, screenshot, elements
                )
            )

    (out / "frames.json").write_text(
        json.dumps({"frames": manifest}, indent=2) + "\n", encoding="utf-8"
    )
    (out / "README.md").write_text(FIXTURE_README, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Harvest a YOLO training set from the sandbox app and live public pages, "
            "labelled by the DOM; or --bench an existing detector on held-out frames."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/models/ui-dataset"),
        help="dataset directory to create (deleted first if it exists)",
    )
    parser.add_argument("--url", help="a sandbox site already running; default is to start one")
    parser.add_argument("--port", type=int, default=8791, help="port for the sandbox we start")
    parser.add_argument("--headed", action="store_true", help="show the browser while harvesting")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="instead of a dataset, write the committed detector test fixtures here",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="two viewports only - a smoke test of the pipeline, not a training set",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="sandbox frames only; skips every live page and needs no network",
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="live pages only; skips the sandbox app entirely",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="instead of a dataset, score --weights on held-out sandbox and real frames",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        action="append",
        default=None,
        help="a .pt file to score under --bench; repeat it for a before-and-after table",
    )
    args = parser.parse_args(argv)

    want_sandbox = not args.no_sandbox
    sandbox = None
    if want_sandbox:
        sandbox = Sandbox(args.url) if args.url else Sandbox.start(args.port)

    if args.bench:
        weights = args.weights or [settings().models_dir / DEFAULT_WEIGHTS_NAME]
        missing = [path for path in weights if not path.is_file()]
        if missing:
            parser.error(f"no such weights file: {', '.join(str(path) for path in missing)}")
        try:
            print(f"capturing held-out frames at {BENCH_VIEWPORT.tag}")
            groups = bench_groups(sandbox, headless=not args.headed)
            bench(weights, groups)
        finally:
            if sandbox is not None:
                sandbox.close()
        return 0

    if args.fixtures is not None:
        try:
            print(f"writing test fixtures to {args.fixtures}")
            build_fixtures(args.fixtures, sandbox, headless=not args.headed, web=not args.no_web)
        finally:
            if sandbox is not None:
                sandbox.close()
        return 0

    viewports = (
        () if args.no_sandbox else ((VIEWPORTS[0], VIEWPORTS[-1]) if args.quick else VIEWPORTS)
    )
    web_viewports = (
        ()
        if args.no_web
        else ((WEB_VIEWPORTS[0], WEB_VIEWPORTS[-1]) if args.quick else WEB_VIEWPORTS)
    )
    started = time.monotonic()
    try:
        print(
            f"harvesting {len(RECIPES)} app states x {len(viewports)} viewports "
            f"and {len(WEB_PAGES) if web_viewports else 0} real pages "
            f"x {len(web_viewports)} viewports"
        )
        counts = build(
            args.out,
            sandbox,
            headless=not args.headed,
            viewports=viewports,
            web_viewports=web_viewports,
        )
    finally:
        if sandbox is not None:
            sandbox.close()

    print(
        f"\n{counts.frames} frames, {counts.labels} labels "
        f"({counts.labels / max(counts.frames, 1):.1f} per frame) "
        f"in {time.monotonic() - started:.1f}s -> {args.out}"
    )
    print("  split: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.per_split.items())))
    print("  class: " + ", ".join(f"{k}={counts.per_class.get(k, 0)}" for k in CLASS_NAMES))
    missing = [name for name in CLASS_NAMES if not counts.per_class.get(name)]
    if missing:
        print(f"  note: no examples of {', '.join(missing)} - the sandbox app has none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
