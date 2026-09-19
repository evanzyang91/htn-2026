"""A ``contracts.Controller`` that drives a real browser page through Playwright.

The controller is deliberately blind. It moves a pointer and presses keys at
coordinates, exactly as a person would, and never resolves a CSS selector on the
agent's behalf - if the agent could name an element, the project would not be
learning to see. Everything crossing this boundary is in LOGICAL pixels, which for
a browser are CSS pixels relative to the top-left of the viewport.

Two classes live here:

:class:`BrowserController`
    Eyes and hands. Screenshots in, synthetic input out.

:class:`BrowserGroundTruth`
    An OFFLINE TEACHER that reads the DOM. It is constructed explicitly and is
    never handed out by the controller; see its docstring for why.

Typical use::

    with BrowserController(headless=True, viewport=(1280, 800)) as ctl:
        ctl.perform(Navigate("https://example.com"))
        shot = ctl.capture()
        ctl.perform(Click(Point(640, 400)))
"""

from __future__ import annotations

import hashlib
import threading
import time
from types import TracebackType
from typing import Literal

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, Playwright, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from skillweaver.contracts import (
    Action,
    ActionKind,
    ActionResult,
    Box,
    Click,
    Drag,
    Element,
    ElementKind,
    ElementSource,
    Move,
    Navigate,
    Point,
    PressKey,
    Screenshot,
    Scroll,
    TypeText,
    Wait,
    utcnow,
)
from skillweaver.controllers import _coords
from skillweaver.errors import ControllerError

BrowserName = Literal["chromium", "firefox", "webkit"]

_SUPPORTED_ACTIONS: frozenset[str] = frozenset(
    {"click", "move", "drag", "type_text", "press_key", "scroll", "wait", "navigate"}
)
"""Every action kind in the contract. A browser can do all of them, ``navigate``
included - that is the one thing it has that a raw desktop controller does not."""


class _ThreadDriver(threading.local):
    """The Playwright driver for ONE thread, shared by every controller on it.

    Playwright's sync API runs its own event loop per thread and refuses a second
    ``sync_playwright().start()`` while the first is live, so two controllers in
    one thread - an eval harness comparing two browsers, a notebook, a test module
    - have to share one driver. It is started on the first controller and stopped
    when the last one closes. A controller on another thread gets its own, which
    is what Playwright requires anyway.
    """

    playwright: Playwright | None = None
    users: int = 0


_THREAD_DRIVER = _ThreadDriver()


def _acquire_driver() -> Playwright:
    """Start this thread's Playwright driver if needed and claim a reference."""
    if _THREAD_DRIVER.playwright is None:
        _THREAD_DRIVER.playwright = sync_playwright().start()
        _THREAD_DRIVER.users = 0
    _THREAD_DRIVER.users += 1
    return _THREAD_DRIVER.playwright


def _release_driver() -> None:
    """Drop a reference, stopping the driver once nothing is using it."""
    if _THREAD_DRIVER.playwright is None:
        return
    _THREAD_DRIVER.users -= 1
    if _THREAD_DRIVER.users <= 0:
        try:
            _THREAD_DRIVER.playwright.stop()
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass
        _THREAD_DRIVER.playwright = None
        _THREAD_DRIVER.users = 0


_SCROLL_QUIET_MS = 80
"""How long nothing may scroll before a scroll counts as finished."""

_SCROLL_QUIET_DEADLINE_MS = 2_000
"""Hard ceiling on waiting for a scroll, so an endlessly animating page cannot
wedge the controller."""

_SCROLL_QUIET_JS = """
([quietMs, deadlineMs]) => new Promise((resolve) => {
  const start = performance.now();
  let lastScroll = start;
  const bump = () => { lastScroll = performance.now(); };
  document.addEventListener('scroll', bump, { capture: true, passive: true });
  const tick = () => {
    const now = performance.now();
    if (now - lastScroll >= quietMs || now - start >= deadlineMs) {
      document.removeEventListener('scroll', bump, { capture: true });
      resolve(Math.round(now - start));
    } else {
      setTimeout(tick, 16);
    }
  };
  setTimeout(tick, 16);
})
"""


_PAINT_QUIET_MS = 60
"""How long the document may go unchanged before a page counts as drawn.

``document.readyState === "complete"`` means the DOCUMENT has arrived, which on a
client-rendered application says nothing about whether anything has been drawn: the
markup is an empty root element and the content appears later, when a fetch returns.
That is most applications, so an agent that captured on ``complete`` would regularly
read a blank page - and the sandbox in this repository is one of them.
"""

_PAINT_QUIET_DEADLINE_MS = 2_500
"""Hard ceiling on waiting to be drawn. A page that mutates forever - a clock, a
progress bar, an animation - must not wedge the controller, so running out of time
means waiting is over, not that anything failed."""

_PAINT_QUIET_JS = """
([quietMs, deadlineMs]) => new Promise((resolve) => {
  const start = performance.now();
  let lastChange = start;
  const observer = new MutationObserver(() => { lastChange = performance.now(); });
  observer.observe(document.documentElement, {
    childList: true, subtree: true, attributes: true, characterData: true,
  });
  const tick = () => {
    const now = performance.now();
    if (now - lastChange >= quietMs || now - start >= deadlineMs) {
      observer.disconnect();
      resolve(Math.round(now - start));
    } else {
      setTimeout(tick, 16);
    }
  };
  setTimeout(tick, 16);
})
"""


class BrowserController:
    """Playwright-backed eyes and hands on one browser page.

    Headless by default. Use it as a context manager so the browser is always torn
    down, even when the body raises::

        with BrowserController(headless=False) as ctl:
            ...

    Args:
        headless: Run without a visible window. ``False`` opens a real one, which
            is what the demo uses.
        viewport: ``(width, height)`` of the page in LOGICAL pixels.
        device_scale_factor: Physical pixels per logical pixel. ``2.0`` makes
            captures Retina-sharp at the same logical size; ``Screenshot.scale``
            will report it back.
        browser: Which Playwright engine to launch.
        settle_ms: How long to wait after each action for the page to react.
            Set to ``0`` for the fastest possible replay of a known-good script.
        type_delay_ms: Pause between characters in :class:`TypeText`. A few
            milliseconds is enough for input handlers that debounce keystrokes.
        drag_steps: Intermediate pointer positions in a :class:`Drag`, so
            handlers that track movement see a real gesture rather than a jump.
        navigation_timeout_ms: Ceiling on a single :class:`Navigate`.
        settle_timeout_ms: Ceiling on waiting for the page to finish reacting to
            an action, including a navigation the action set off. Settling is
            best-effort, so reaching this is not an error.
        start_url: Loaded once at construction, if given.

    Raises:
        ControllerError: if the browser cannot be launched.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        viewport: tuple[int, int] = (1280, 800),
        device_scale_factor: float = 1.0,
        browser: BrowserName = "chromium",
        settle_ms: float = 120.0,
        type_delay_ms: float = 8.0,
        drag_steps: int = 12,
        navigation_timeout_ms: float = 15_000.0,
        settle_timeout_ms: float = 3_000.0,
        start_url: str | None = None,
    ) -> None:
        width, height = viewport
        if width <= 0 or height <= 0:
            raise ValueError(f"viewport must be positive, got {viewport!r}")
        _coords.check_scale(device_scale_factor)

        self._browser_name: BrowserName = browser
        self._headless = headless
        self._requested_scale = device_scale_factor
        self._viewport = Box(0, 0, int(width), int(height))
        self._settle_ms = max(float(settle_ms), 0.0)
        self._type_delay_ms = max(float(type_delay_ms), 0.0)
        self._drag_steps = max(int(drag_steps), 1)
        self._navigation_timeout_ms = float(navigation_timeout_ms)
        self._settle_timeout_ms = max(float(settle_timeout_ms), 0.0)
        self._closed = False

        self._page: Page | None = None
        self._holds_driver = False
        try:
            playwright = _acquire_driver()
            self._holds_driver = True
            engine = getattr(playwright, browser)
            # Chromium animates wheel scrolling, so a scroll would still be moving
            # when perform() returned and the next capture would catch a half-
            # scrolled frame. Turning the animation off makes a wheel land at once;
            # _await_scroll_quiet below covers pages that animate scrolling
            # themselves, and engines where this flag does not exist.
            args = ["--disable-smooth-scrolling"] if browser == "chromium" else []
            self._browser = engine.launch(headless=headless, args=args)
            self._context = self._browser.new_context(
                viewport={"width": int(width), "height": int(height)},
                device_scale_factor=device_scale_factor,
            )
            self._page = self._context.new_page()
            if start_url is not None:
                self._page.goto(start_url, timeout=self._navigation_timeout_ms)
                # The same settle every other navigation gets. Without it the first
                # capture of the session can catch an application that has loaded its
                # document but not yet drawn anything - and an agent that starts by
                # reading a blank page has no control to act on, so it never acts,
                # never re-reads, and spends its whole budget on an empty screen.
                self._settle(self._page)
        except Exception as exc:
            # A half-built controller still owns an OS process; do not leak it.
            self.close()
            raise ControllerError(f"could not launch {browser}: {_brief(exc)}") from exc

    # -- context manager ---------------------------------------------------------------

    def __enter__(self) -> BrowserController:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- Controller protocol -----------------------------------------------------------

    def capture(self) -> Screenshot:
        """Grab the current viewport.

        ``width``/``height`` are the logical viewport and ``scale`` is MEASURED from
        the PNG that was actually produced rather than from the scale factor the
        browser was asked for, so the reported geometry and the bytes can never
        drift apart.
        """
        page = self._live_page()
        try:
            png = page.screenshot(caret="hide")
        except PlaywrightError as exc:
            raise ControllerError(f"screenshot failed: {_brief(exc)}") from exc

        width, height = self._logical_size(page)
        physical_width, physical_height = _coords.png_size(png)
        scale = _coords.scale_for(physical_width, width)
        height_scale = _coords.scale_for(physical_height, height)
        if abs(scale - height_scale) > 0.01 * scale:
            raise ControllerError(
                f"capture is not the viewport: {physical_width}x{physical_height} png "
                f"for a {width}x{height} logical viewport"
            )
        return Screenshot(png=png, width=width, height=height, scale=scale, captured_at=utcnow())

    def perform(self, action: Action) -> ActionResult:
        """Deliver one action as synthetic input, then let the page settle.

        Anything that stops the action being delivered - an off-screen point, an
        empty key chord, a navigation that fails - comes back as
        ``ActionResult(ok=False, error=...)``. Only a dead controller raises.
        """
        page = self._live_page()
        started = time.perf_counter()
        try:
            error = self._deliver(page, action)
        except PlaywrightTimeoutError as exc:
            error = f"{action.kind} timed out: {_brief(exc)}"
        except PlaywrightError as exc:
            if page.is_closed():
                raise ControllerError(f"page closed during {action.kind}") from exc
            error = f"{action.kind} failed: {_brief(exc)}"
        return ActionResult(
            ok=error is None, error=error, elapsed_ms=(time.perf_counter() - started) * 1000.0
        )

    def viewport(self) -> Box:
        """The page area in logical pixels, anchored at ``(0, 0)``."""
        page = self._live_page()
        size = page.viewport_size
        if size:
            self._viewport = Box(0, 0, int(size["width"]), int(size["height"]))
        return self._viewport

    def supports(self, action_kind: ActionKind) -> bool:
        """True for every kind in the contract, ``navigate`` included."""
        return action_kind in _SUPPORTED_ACTIONS

    def url(self) -> str | None:
        """The address the page is on, or ``None`` if it has none yet."""
        return self._live_page().url or None

    def describe(self) -> str:
        """One line for logs and prompts, e.g. ``playwright chromium 1280x800 @2x``."""
        view = self._viewport if self._closed else self.viewport()
        mode = "" if self._headless else " headed"
        return (
            f"playwright {self._browser_name} {view.w}x{view.h} @{self._requested_scale:g}x{mode}"
        )

    def close(self) -> None:
        """Shut the page, context, browser and Playwright driver down. Idempotent
        and never raises: closing a half-dead browser must not mask the real error."""
        self._closed = True
        for name in ("_page", "_context", "_browser"):
            handle = getattr(self, name, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:  # noqa: BLE001 - teardown must never raise
                    pass
                setattr(self, name, None)
        if self._holds_driver:
            self._holds_driver = False
            _release_driver()

    # -- internals ---------------------------------------------------------------------

    def _live_page(self) -> Page:
        if self._closed or self._page is None:
            raise ControllerError("browser controller is closed")
        if self._page.is_closed():
            raise ControllerError("browser page has been closed")
        return self._page

    def _logical_size(self, page: Page) -> tuple[int, int]:
        """The viewport in logical pixels, preferring what Playwright already knows."""
        size = page.viewport_size
        if size:
            return int(size["width"]), int(size["height"])
        width, height = page.evaluate("() => [window.innerWidth, window.innerHeight]")
        return int(width), int(height)

    def _deliver(self, page: Page, action: Action) -> str | None:
        """Perform ``action``; return ``None`` on success or a reason it was refused."""
        bounds = self.viewport()
        for point in _points_of(action):
            if not _coords.within_viewport(bounds, point):
                return f"point ({point.x}, {point.y}) is outside the {bounds.w}x{bounds.h} viewport"

        match action:
            case Click():
                if action.clicks < 1:
                    return f"click needs at least one click, got {action.clicks}"
                page.mouse.click(
                    action.point.x,
                    action.point.y,
                    button=action.button,
                    click_count=action.clicks,
                )
            case Move():
                page.mouse.move(action.point.x, action.point.y)
            case Drag():
                page.mouse.move(action.start.x, action.start.y)
                page.mouse.down()
                page.mouse.move(action.end.x, action.end.y, steps=self._drag_steps)
                page.mouse.up()
            case TypeText():
                page.keyboard.type(action.text, delay=self._type_delay_ms)
            case PressKey():
                if not action.keys:
                    return "press_key needs at least one key"
                return self._press_chord(page, action.keys)
            case Scroll():
                # Park the pointer first: a wheel event scrolls whatever is under it.
                page.mouse.move(action.point.x, action.point.y)
                page.mouse.wheel(action.dx, action.dy)
                self._await_scroll_quiet(page)
            case Wait():
                if action.ms < 0:
                    return f"wait needs a non-negative duration, got {action.ms}"
                page.wait_for_timeout(action.ms)
                return None  # an explicit wait is its own settle
            case Navigate():
                return self._navigate(page, action.url)
            case _:
                return f"action kind {action.kind!r} is not supported"

        self._settle(page)
        return None

    def _navigate(self, page: Page, url: str) -> str | None:
        """Load ``url``, retrying once if a previous navigation got in the way.

        An agent that tries a dead URL and then a good one would otherwise see the
        good one fail: the error page from the first is still committing, and
        Chromium reports that as this navigation being interrupted. The URL is
        fine, so the retry is the honest answer rather than a spurious refusal.
        """
        for attempt in (1, 2):
            try:
                page.goto(url, timeout=self._navigation_timeout_ms, wait_until="load")
            except PlaywrightTimeoutError as exc:
                self._settle(page)
                return f"navigate timed out: {_brief(exc)}"
            except PlaywrightError as exc:
                message = _brief(exc)
                if attempt == 1 and "interrupted by another navigation" in message:
                    continue
                # Let a failed navigation's error page finish committing, so the
                # next action does not trip over it.
                self._settle(page)
                return f"navigate failed: {message}"
            self._settle(page)
            return None
        return None

    def _press_chord(self, page: Page, keys: tuple[str, ...]) -> str | None:
        """Hold every key but the last, tap the last, then release in reverse order."""
        *held, final = keys
        pressed: list[str] = []
        try:
            for key in held:
                page.keyboard.down(key)
                pressed.append(key)
            page.keyboard.press(final)
        finally:
            for key in reversed(pressed):
                try:
                    page.keyboard.up(key)
                except PlaywrightError:  # already released, or the page went away
                    pass
        self._settle(page)
        return None

    def _await_scroll_quiet(self, page: Page) -> None:
        """Block until nothing on the page has scrolled for a short quiet period.

        A wheel event is delivered asynchronously and a page may animate the
        scroll it triggers, so the position right after ``mouse.wheel`` is a
        position in motion. Listening for scroll events in the capture phase
        catches any scroller, not just the window, and the deadline guarantees
        this returns even on a page that scrolls forever.
        """
        page.evaluate(_SCROLL_QUIET_JS, [_SCROLL_QUIET_MS, _SCROLL_QUIET_DEADLINE_MS])

    def _await_paint_quiet(self, page: Page) -> None:
        """Block until the document has stopped changing for a short quiet period.

        A page whose content arrives from a fetch is complete long before it is drawn,
        and a capture taken in between shows an agent an empty screen with nothing to
        act on. Watching for mutations settles that for any application rather than
        for one that happens to announce itself; a page that never stops changing is
        released by the deadline.

        Failures are swallowed on purpose: a context that vanished under the probe is
        a navigation, which the caller is already looping on, and an engine without
        ``MutationObserver`` should degrade to the old behaviour rather than break.
        """
        try:
            page.evaluate(_PAINT_QUIET_JS, [_PAINT_QUIET_MS, _PAINT_QUIET_DEADLINE_MS])
        except PlaywrightError:
            return

    def _settle(self, page: Page) -> None:
        """Give the page its moment to react, and to finish arriving if the action
        sent it somewhere.

        A click on a link sets off a navigation that is still committing when the
        click itself has been delivered, so an action that returned immediately
        would hand the next ``capture()`` a blank half-loaded frame. Waiting for
        ``document.readyState`` to reach ``complete`` is what actually catches
        that; a bare ``wait_for_load_state`` can return mid-commit.

        On a page that is already idle this costs one round trip. A page that
        never finishes loading must not wedge the controller, so running out of
        ``settle_timeout_ms`` means settling is done, not that anything failed.

        Arriving is not the same as being drawn. ``readyState`` reaching ``complete``
        says the DOCUMENT is here, and on a client-rendered application the document
        is an empty root element whose content appears when a fetch returns. So the
        wait ends on the document going quiet - see :data:`_PAINT_QUIET_JS` - rather
        than on it having loaded.
        """
        if self._settle_ms > 0:
            page.wait_for_timeout(self._settle_ms)

        deadline = time.monotonic() + self._settle_timeout_ms / 1000.0
        while True:
            try:
                if page.evaluate("() => document.readyState") == "complete":
                    self._await_paint_quiet(page)
                    return
            except PlaywrightError:
                # The context vanished under the probe: a navigation just
                # committed, which is exactly what there is to wait for.
                pass
            remaining_ms = (deadline - time.monotonic()) * 1000.0
            if remaining_ms <= 0:
                return
            try:
                page.wait_for_load_state("load", timeout=remaining_ms)
                # wait_for_load_state can return while a navigation is still
                # committing, so pause before re-probing rather than spinning.
                page.wait_for_timeout(min(25.0, max(remaining_ms, 1.0)))
            except PlaywrightError:
                return


def _points_of(action: Action) -> tuple[Point, ...]:
    """Every viewport coordinate an action will touch, for bounds checking."""
    if isinstance(action, Click | Move | Scroll):
        return (action.point,)
    if isinstance(action, Drag):
        return (action.start, action.end)
    return ()


def _brief(exc: BaseException) -> str:
    """Playwright errors carry a page of context; keep the first useful line."""
    text = str(exc).strip().splitlines()
    head = text[0] if text else exc.__class__.__name__
    return head[:200]


# --------------------------------------------------------------------------------------
# Offline teacher
# --------------------------------------------------------------------------------------

_KIND_VALUES = {kind.value for kind in ElementKind}

_GROUND_TRUTH_JS = """
() => {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const MAX_TEXT = 200;

  const ROLE_KINDS = {
    button: 'button', link: 'link', checkbox: 'checkbox', switch: 'checkbox',
    radio: 'radio', textbox: 'text_field', searchbox: 'text_field',
    combobox: 'menu', listbox: 'menu', menu: 'menu', menuitem: 'menu',
    tab: 'tab', row: 'row', img: 'image', heading: 'text',
  };
  // Only ever reached by an element carrying its own text node (see `ownText`
  // below), which is what keeps plain `div` in the list from reporting every
  // layout wrapper on a real page.
  const TEXT_TAGS = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span', 'label',
    'li', 'td', 'th', 'legend', 'strong', 'em', 'dt', 'dd', 'figcaption', 'caption',
    'div', 'section', 'article', 'blockquote', 'pre', 'code']);
  const BUTTON_INPUTS = new Set(['button', 'submit', 'reset', 'image']);
  const NON_TEXT_INPUTS = new Set(['checkbox', 'radio', 'file', 'range', 'color', 'hidden']);
  // Anything whose own label would otherwise be reported a second time as loose text.
  const LABELLED = 'a[href], button, select, textarea, label, summary,' +
    ' [role="button"], [role="link"], [role="tab"], [role="menuitem"]';

  const typeOf = (el) => (el.getAttribute('type') || 'text').trim().toLowerCase();

  const kindOf = (el, tag) => {
    const role = (el.getAttribute('role') || '').trim().toLowerCase();
    if (ROLE_KINDS[role]) return ROLE_KINDS[role];
    if (el.hasAttribute('contenteditable')) return 'text_field';
    if (tag === 'button' || tag === 'summary') return 'button';
    if (tag === 'input') {
      const type = typeOf(el);
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      if (BUTTON_INPUTS.has(type)) return 'button';
      return NON_TEXT_INPUTS.has(type) ? 'other' : 'text_field';
    }
    if (tag === 'textarea') return 'text_field';
    if (tag === 'a') return el.hasAttribute('href') ? 'link' : null;
    if (tag === 'select') return 'menu';
    if (tag === 'tr') return 'row';
    if (tag === 'img') return 'image';
    if (tag === 'svg') return 'icon';
    if (TEXT_TAGS.has(tag)) return 'text';
    return null;
  };

  const textOf = (el, tag) => {
    const aria = (el.getAttribute('aria-label') || '').trim();
    if (aria) return aria;
    if (tag === 'img') return (el.getAttribute('alt') || '').trim();
    if (tag === 'input') {
      const type = typeOf(el);
      if (BUTTON_INPUTS.has(type)) return (el.value || '').trim();
      if (NON_TEXT_INPUTS.has(type)) return (el.getAttribute('title') || '').trim();
      return (el.value || el.getAttribute('placeholder') || '').trim();
    }
    if (tag === 'select') {
      const chosen = el.selectedOptions && el.selectedOptions[0];
      return chosen ? chosen.text.trim() : '';
    }
    return (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
  };

  const elements = [];
  for (const el of document.querySelectorAll('*')) {
    const tag = el.tagName.toLowerCase();
    const kind = kindOf(el, tag);
    if (!kind) continue;

    const style = getComputedStyle(el);
    if (style.display === 'none' || style.visibility !== 'visible') continue;
    if (parseFloat(style.opacity || '1') <= 0.01) continue;

    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) continue;
    if (rect.right <= 0 || rect.bottom <= 0 || rect.left >= vw || rect.top >= vh) continue;

    if (kind === 'text') {
      // A button's caption is the button; do not also report it as floating text.
      if (el.parentElement && el.parentElement.closest(LABELLED)) continue;
      const ownText = Array.from(el.childNodes).some(
        (node) => node.nodeType === 3 && node.textContent.trim().length > 0);
      if (!ownText) continue;
    }

    elements.push({
      kind: kind,
      text: textOf(el, tag).slice(0, MAX_TEXT),
      x: rect.left, y: rect.top, w: rect.width, h: rect.height,
    });
  }
  return { viewport: [vw, vh], elements: elements };
}
"""


class BrowserGroundTruth:
    """Perfect element knowledge read from the DOM. AN OFFLINE TEACHER ONLY.

    **The agent must never see this.** Its whole job is to know things the agent is
    supposed to learn to see, so it exists for exactly two callers: the labeller
    that builds detector training data, and the evaluation harness that scores how
    close pixel-derived perception got to the truth.

    That is why it is a separate class you construct yourself::

        truth = BrowserGroundTruth(controller)   # visible at the call site

    and why :class:`BrowserController` has no method that hands one out. A
    perceiver, explorer, planner, skill runner or generated skill that reaches this
    class has broken the premise of the project. Code that legitimately needs it
    takes it as an explicit argument.

    Boxes are in LOGICAL pixels in the controller's own coordinate space - the same
    space clicks are expressed in - clipped to the viewport, so a label always
    describes a rectangle a detector could actually have seen. Elements that are
    ``display:none``, ``visibility:hidden``, transparent, zero-sized or entirely
    off-screen are filtered out. Elements are returned in reading order.

    Known limits: it reads the main frame's light DOM only, so content inside
    iframes or a shadow root is not reported, and it does not test whether one
    element is painted over another.
    """

    def __init__(self, controller: BrowserController) -> None:
        self._controller = controller

    def elements(self) -> list[Element]:
        """Every visible element of the page, ``source=dom``, ``confidence=1.0``."""
        data = self._read_dom()

        width, height = data["viewport"]
        viewport = Box(0, 0, int(width), int(height))
        elements: list[Element] = []
        for raw in data["elements"]:
            # getBoundingClientRect already reports CSS pixels, so the conversion is
            # at scale 1: what matters is that edges round the same way everywhere.
            box = _coords.clip_to_viewport(
                _coords.box_to_logical(raw["x"], raw["y"], raw["w"], raw["h"], 1.0), viewport
            )
            if box.area == 0:
                continue
            kind = ElementKind(raw["kind"]) if raw["kind"] in _KIND_VALUES else ElementKind.other
            text = raw["text"]
            elements.append(
                Element(
                    box=box,
                    kind=kind,
                    text=text,
                    confidence=1.0,
                    stable_id=_stable_id(kind, text, box),
                    source=ElementSource.dom,
                )
            )
        elements.sort(key=lambda el: (el.box.y, el.box.x))
        return elements

    def url(self) -> str:
        """The exact current URL, ``""`` when there is none."""
        return self._controller._live_page().url or ""

    def _read_dom(self) -> dict:
        """Run the reader script, once more if a navigation pulled the rug out.

        A page that is mid-navigation - a redirect chain, a click that has just
        committed - destroys the execution context under the script. That is a
        moment in time rather than a broken page, so wait for the new document and
        ask again. A second failure is real and is raised.
        """
        page = self._controller._live_page()
        try:
            return page.evaluate(_GROUND_TRUTH_JS)
        except PlaywrightError as exc:
            if "context was destroyed" not in str(exc) and "navigating" not in str(exc):
                raise ControllerError(f"could not read the DOM: {_brief(exc)}") from exc
        try:
            page.wait_for_load_state("load")
            return page.evaluate(_GROUND_TRUTH_JS)
        except PlaywrightError as exc:
            raise ControllerError(f"could not read the DOM: {_brief(exc)}") from exc


def _stable_id(kind: ElementKind, text: str, box: Box) -> str:
    """Identity for "the same element across observations of the same screen".

    Kind and text plus position rounded to a 16-pixel grid, so a label that shifts
    by a pixel or two keeps its identity while a different control never borrows it.
    """
    seed = f"{kind.value}|{text}|{box.x // 16}|{box.y // 16}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
