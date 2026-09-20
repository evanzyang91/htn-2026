"""A ``contracts.Controller`` driving a real browser page through Playwright.

Deliberately blind: it moves a pointer and presses keys at LOGICAL (CSS) pixel
coordinates and never resolves a selector for the agent. :class:`BrowserGroundTruth`
reads the DOM and is an OFFLINE TEACHER only - see its docstring.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, Playwright, Route, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from skillweaver.contracts import (
    Action,
    ActionKind,
    ActionResult,
    Back,
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
from skillweaver.controllers._quiesce import QUIESCE_JS
from skillweaver.controllers.chrome_launch import PLAINLY_LAUNCHED, ChromeProcess
from skillweaver.errors import ControllerError

BrowserName = Literal["chromium", "firefox", "webkit"]

SOMETIMES_ONLY_OVERLAYS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Special:BannerLoader"),
    re.compile(r"Special:RecordImpression"),
    re.compile(r"geoiplookup"),
)
"""Requests serving an overlay a page renders only SOMETIMES, aborted by default.

A measurement wants one page to be one screen, and everything downstream compares two
screens of the same URL. Measured on live ``en.wikipedia.org`` 2026-09-19: of thirty
plain ``/wiki/Main_Page`` loads, five rendered a 531px CentralNotice appeal and the two
groups fingerprint at 0.132, far below the 0.26 same-state cut; two of fifteen live
learning runs threw away a correct skill because of it. Ten loads with these patterns
aborted rendered it zero times.

REGEXES, not the Playwright globs this looks like: CentralNotice serves the appeal from
``index.php?title=Special:BannerLoader&...``, so the name is in the QUERY STRING and
``**/Special:BannerLoader*`` matches nothing. Force the appeal with ``?banner=<name>&force=1``
to re-check. Pass ``block=()`` to measure the overlay itself.
"""

REAL_CHROME_CHANNEL = "chrome"
"""Playwright's name for the Google Chrome INSTALLED ON THIS MACHINE, not the bundled
Chromium. Half of what it takes to be served by a site that refuses an automated browser;
the other half is ``launch_persistent_context(<dir>)``, a profile that SURVIVES the run.

Measured 2026-09-19 against live doordash.com: those two settings alone loaded the front
page, a city listing and a store menu with no interstitial, and nothing masks automation
to get there - ``navigator.webdriver`` stays true and a re-measurement without any such
argument carried the whole result. Access is not a guarantee: the same profile got the
interstitial hours later after many loads from one address and a profile shared between
concurrent runs. Give each run its own directory; a human-verification page FAILS the run
and a person clears it by hand, once, in that profile."""

_SUPPORTED_ACTIONS: frozenset[str] = frozenset(
    {"click", "move", "drag", "type_text", "press_key", "scroll", "wait", "navigate", "back"}
)
"""A browser can do every kind. ``navigate`` and ``back`` are the two a desktop
controller lacks, because both are session history and a desktop has none."""


class _ThreadDriver(threading.local):
    """The Playwright driver for ONE thread, shared by every controller on it.

    Playwright's sync API refuses a second ``sync_playwright().start()`` while the first
    is live, so two controllers on one thread have to share. Started on the first
    controller, stopped when the last one closes.
    """

    playwright: Playwright | None = None
    users: int = 0


_THREAD_DRIVER = _ThreadDriver()


def _acquire_driver() -> Playwright:
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


def _abort(route: Route) -> None:
    """Refuse one request. A handler that raises kills the page, and the request may
    already be gone, so aborting is best-effort."""
    try:
        route.abort()
    except Exception:  # noqa: BLE001 - a request that is already gone is blocked enough
        pass


_SCROLL_QUIET_MS = 80
"""How long nothing may scroll before a scroll counts as finished."""

_SCROLL_QUIET_DEADLINE_MS = 2_000
"""Ceiling, so an endlessly animating page cannot wedge the controller."""

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


class BrowserController:
    """Playwright-backed eyes and hands on one browser page.

    Headless by default. Use it as a context manager so the browser is always torn down.

    Args:
        headless: Run without a visible window.
        viewport: ``(width, height)`` in LOGICAL pixels.
        device_scale_factor: Physical pixels per logical pixel, reported back as
            ``Screenshot.scale``. Under ``user_data_dir`` it becomes a LAUNCH option
            rather than a per-context one: a persistent profile IS the browser, so one
            process holds exactly one scale.
        browser: Which Playwright engine to launch.
        settle_ms: Wait after each action for the page to react; ``0`` for fastest replay.
        type_delay_ms: Pause between characters, for handlers that debounce keystrokes.
        drag_steps: Intermediate pointer positions, so movement handlers see a gesture.
        navigation_timeout_ms: Ceiling on a single :class:`Navigate`.
        settle_timeout_ms: Ceiling on settling. Best-effort, so reaching it is not an error.
        user_data_dir: A PERSISTENT Chrome profile directory; drives the REAL Chrome out
            of it - see :data:`REAL_CHROME_CHANNEL`. Created if missing, never deleted,
            and EXCLUSIVE: Chrome locks a profile, so an open window fails the launch.
        attach: Start that Chrome as an ORDINARY PROCESS and attach over its debugging
            port - the only one of the three launch configurations a live DoorDash serves;
            see :data:`~skillweaver.controllers.chrome_launch.PLAINLY_LAUNCHED`. Requires
            ``user_data_dir``. The one behavioural difference is that ``viewport`` is
            applied to the adopted page, there being no context of ours to give it to.
        chrome_binary: Where Chrome is, for ``attach``, or ``None`` to search.
        start_url: Loaded once at construction.
        block: URL patterns aborted on the CONTEXT. A regex matches anywhere in the URL;
            a string is a Playwright path glob. ``None`` means
            :data:`SOMETIMES_ONLY_OVERLAYS`; ``()`` blocks nothing.

    Raises:
        ControllerError: the browser could not be launched. A missing real Chrome fails
            HERE, by name, rather than falling back to the Chromium a bot wall blocks.
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
        block: Sequence[str | re.Pattern[str]] | None = None,
        user_data_dir: str | Path | None = None,
        attach: bool = False,
        chrome_binary: str | Path | None = None,
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
        self._acted_ms = 0.0
        self._blocked: tuple[str | re.Pattern[str], ...] = (
            SOMETIMES_ONLY_OVERLAYS if block is None else tuple(block)
        )
        self._profile_dir = Path(user_data_dir).expanduser() if user_data_dir is not None else None
        if self._profile_dir is not None and browser != "chromium":
            raise ValueError(
                f"user_data_dir drives real Google Chrome, which is a chromium channel, "
                f"not {browser!r}"
            )
        self._attach = bool(attach)
        if self._attach and self._profile_dir is None:
            raise ValueError(
                "attach=True starts a real Chrome of its own and needs a user_data_dir "
                "to start it on; give this run its own directory"
            )
        self._closed = False

        self._page: Page | None = None
        self._chrome: ChromeProcess | None = None
        self._holds_driver = False
        try:
            playwright = _acquire_driver()
            self._holds_driver = True
            engine = getattr(playwright, browser)
            # Chromium animates wheel scrolling, so a scroll would still be moving when
            # perform() returned; _await_scroll_quiet covers pages that animate it
            # themselves and engines without this flag.
            args = ["--disable-smooth-scrolling"] if browser == "chromium" else []
            view = {"width": int(width), "height": int(height)}
            if self._attach:
                # Playwright does not launch here, so ``args`` cannot apply; the same
                # scroll setting is passed to the process instead.
                assert self._profile_dir is not None  # checked above
                self._chrome = ChromeProcess(
                    user_data_dir=self._profile_dir,
                    headless=headless,
                    binary=chrome_binary,
                    device_scale_factor=device_scale_factor,
                    window=(int(width), int(height)),
                )
                self._browser = engine.connect_over_cdp(self._chrome.endpoint)
                # Adopt what the browser already has; a new one would leave the demo
                # watching a tab nobody drives.
                contexts = self._browser.contexts
                self._context = contexts[0] if contexts else self._browser.new_context()
            elif self._profile_dir is not None:
                # The persistent context IS the launch: no Browser handle, and closing
                # the context is what shuts the process down.
                self._context = engine.launch_persistent_context(
                    str(self._profile_dir),
                    channel=REAL_CHROME_CHANNEL,
                    headless=headless,
                    viewport=view,
                    device_scale_factor=device_scale_factor,
                    args=args,
                )
            else:
                self._browser = engine.launch(headless=headless, args=args)
                self._context = self._browser.new_context(
                    viewport=view,
                    device_scale_factor=device_scale_factor,
                )
            # On the CONTEXT: the route then survives every navigation and covers popups.
            for pattern in self._blocked:
                self._context.route(pattern, _abort)
            # A persistent context already has a page; ADOPT it rather than opening a
            # second one nobody drives.
            existing = self._context.pages if self._profile_dir is not None else []
            self._page = existing[0] if existing else self._context.new_page()
            if self._attach:
                # The launch was an ordinary one, so the size goes on the page - which is
                # what every capture and coordinate is measured against anyway.
                self._page.set_viewport_size(view)
            if start_url is not None:
                self._page.goto(start_url, timeout=self._navigation_timeout_ms)
        except Exception as exc:
            # A half-built controller still owns an OS process; do not leak it.
            self.close()
            raise ControllerError(self._launch_failed(exc)) from exc

    # -- context manager ---------------------------------------------------------------

    @property
    def headless(self) -> bool:
        """Whether this browser runs without a visible window.

        Readable because the answer OUTLIVES the browser: the two modes render one page
        differently enough that a cross-mode comparison cannot succeed, and
        :func:`skillweaver.render_mode.mode_of` needs that as a fact, not as prose.
        """
        return self._headless

    @property
    def profile_dir(self) -> Path | None:
        """The persistent Chrome profile, or ``None`` for the bundled-Chromium launch.
        Readable for the same reason as :attr:`headless`: it changes which browser
        rendered a screen."""
        return self._profile_dir

    @property
    def attached(self) -> bool:
        """Whether the browser was started plainly and attached to. Readable for the same
        reason as :attr:`headless`: it is the difference between a screen a live site
        served and an interstitial it served instead."""
        return self._attach

    @property
    def cdp_endpoint(self) -> str | None:
        """``http://127.0.0.1:<port>`` for the attached browser, else ``None``.

        Readable so anything else speaking Chrome's debugging protocol can attach to the
        same process. The process stays owned here and dies with this controller.
        """
        return None if self._chrome is None else self._chrome.endpoint

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

        ``scale`` is MEASURED from the PNG actually produced rather than from what the
        browser was asked for, so geometry and bytes cannot drift apart.
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

        Anything that stops delivery comes back as ``ActionResult(ok=False, error=...)``;
        only a dead controller raises.
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
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._acted_ms += elapsed_ms
        return ActionResult(ok=error is None, error=error, elapsed_ms=elapsed_ms)

    @property
    def acted_ms(self) -> float:
        """Milliseconds spent inside :meth:`perform` so far: delivering input and then
        waiting for the page in :meth:`_settle`. Half of what a run's site time is; the
        DOM perceiver adds the other half - see ``DomPerceiver.site_ms``."""
        return self._acted_ms

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

    @property
    def blocked(self) -> tuple[str | re.Pattern[str], ...]:
        """The URL patterns this controller aborts. See :data:`SOMETIMES_ONLY_OVERLAYS`."""
        return self._blocked

    def evaluate(self, script: str) -> Any:
        """Run ``script`` in the page and return its JSON-able result. READ-ONLY.

        The one hook the DOM perception path needs, and deliberately the only thing it
        adds to this class: a click still goes through :meth:`perform` as a point. Not on
        the ``Controller`` Protocol - that is shared surface and a desktop has no page -
        so callers duck-type on its presence. A document mid-navigation destroys the
        execution context, which is a moment rather than a broken page, so the script is
        asked once more after the new document loads; a second failure is real.

        Raises:
            ControllerError: the script cannot be run, or fails twice.
        """
        page = self._live_page()
        try:
            return page.evaluate(script)
        except PlaywrightError as exc:
            if "context was destroyed" not in str(exc) and "navigating" not in str(exc):
                raise ControllerError(f"could not evaluate page script: {_brief(exc)}") from exc
        try:
            page.wait_for_load_state("load", timeout=self._navigation_timeout_ms)
            return page.evaluate(script)
        except PlaywrightError as exc:
            raise ControllerError(f"could not evaluate page script: {_brief(exc)}") from exc

    def quiesce(self, quiet_ms: float, cap_ms: float) -> float | None:
        """Wait until the page has been STILL for ``quiet_ms``, at most ``cap_ms``.

        READ-ONLY, duck-typed like :meth:`evaluate`, and the milliseconds it waited -
        or ``None`` when the wait was interrupted, which a navigating document does to
        any evaluation. It is ADVISORY either way: nothing that already happened may fail
        because the page would not hold still.

        NOT called from :meth:`_settle`, and it must not be. ``AGENTS.md`` says a quiet
        window taxes every action on every site, and measured here it is 164ms per call
        on a page with nothing to wait for. So the caller decides when a still page is
        worth that: the DOM perceiver, on the one observation that follows a policy's
        move (:meth:`~skillweaver.perception.dom.DomPerceiver.rest_after`).

        What it does and does not catch, measured through this class and
        ``DomPerceiver``, n=5 per page, 5 of 5 agreeing. A results page that streams a
        row every 90ms for ~900ms after ``load``: :meth:`_settle` hands over a frame with
        2 of 20 controls at ~165ms; after this, 20 of 20, for ~906ms. A page that sits
        idle and paints once at 700ms: 0 of 16 before and STILL 0 of 16 after, because
        ``quiet_ms`` of nothing happening is satisfied before the paint. It waits for a
        busy page to finish, not for an idle one to start - that second shape is what
        ``ctx.wait_for_text`` is for.
        """
        page = self._live_page()
        try:
            return float(page.evaluate(QUIESCE_JS, [quiet_ms, cap_ms]))
        except PlaywrightError:
            return None

    def describe(self) -> str:
        """One line for logs and prompts, e.g. ``playwright chromium 1280x800 @2x``.

        A persistent profile says ``chrome`` and an attached one ``chrome+cdp``, because
        being started plainly is what a live site is reading.
        """
        view = self._viewport if self._closed else self.viewport()
        mode = "" if self._headless else " headed"
        if self._attach:
            engine = f"{REAL_CHROME_CHANNEL}+cdp"
        elif self._profile_dir is not None:
            engine = REAL_CHROME_CHANNEL
        else:
            engine = self._browser_name
        return f"playwright {engine} {view.w}x{view.h} @{self._requested_scale:g}x{mode}"

    def close(self) -> None:
        """Shut the page, context, browser and driver down. Idempotent, never raises.

        A persistent profile directory is LEFT ON DISK: its cookies are what the next run
        needs. An ATTACHED browser's handles are only DROPPED - closing them would ask a
        browser we do not own to dismantle itself - and what ends it is killing the
        process this run started. A Chrome this run did not start is never touched.
        """
        self._closed = True
        handles = ("_browser",) if self._attach else ("_page", "_context", "_browser")
        for name in ("_page", "_context", "_browser"):
            handle = getattr(self, name, None)
            if handle is not None:
                if name in handles:
                    try:
                        handle.close()
                    except Exception:  # noqa: BLE001 - teardown must never raise
                        pass
                setattr(self, name, None)
        if self._chrome is not None:
            chrome, self._chrome = self._chrome, None
            chrome.close()
        if self._holds_driver:
            self._holds_driver = False
            _release_driver()

    # -- internals ---------------------------------------------------------------------

    def _launch_failed(self, exc: BaseException) -> str:
        """Why the browser did not open. A missing real Chrome is its own sentence: the
        alternative is a silent fallback to the Chromium a bot wall turns away."""
        if self._attach:
            return (
                f"could not start and attach to a plainly-launched Google Chrome on "
                f"profile {self._profile_dir}: {_brief(exc)}. This mode is "
                f"{PLAINLY_LAUNCHED}; there is no fallback to a framework-launched "
                f"browser, which is the configuration a bot wall refuses."
            )
        if self._profile_dir is None:
            return f"could not launch {self._browser_name}: {_brief(exc)}"
        return (
            f"could not launch real Google Chrome (channel {REAL_CHROME_CHANNEL!r}) on "
            f"profile {self._profile_dir}: {_brief(exc)}. Install Chrome, or close a "
            f"window already holding that profile; there is no fallback to bundled "
            f"Chromium, which is the build a bot wall refuses."
        )

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
            case Back():
                return self._go_back(page)
            case _:
                return f"action kind {action.kind!r} is not supported"

        self._settle(page)
        return None

    def _navigate(self, page: Page, url: str) -> str | None:
        """Load ``url``, retrying once if a previous navigation got in the way.

        A dead URL's error page is still committing when the next ``goto`` starts, and
        Chromium reports that as this navigation being interrupted.
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
                # Let the error page finish committing before the next action.
                self._settle(page)
                return f"navigate failed: {message}"
            self._settle(page)
            return None
        return None

    def _go_back(self, page: Page) -> str | None:
        """Pop one entry off the page's session history.

        Playwright answering ``None`` is the only authority on an empty stack - a page's
        ``history.length`` counts both directions - so an empty stack is REFUSED rather
        than reported as a move, which would tell a policy it went back onto this screen.
        """
        before = page.url
        try:
            response = page.go_back(timeout=self._navigation_timeout_ms, wait_until="load")
        except PlaywrightTimeoutError as exc:
            self._settle(page)
            return f"back timed out: {_brief(exc)}"
        except PlaywrightError as exc:
            self._settle(page)
            return f"back failed: {_brief(exc)}"
        self._settle(page)
        if response is None and page.url == before:
            return "there is nothing behind this page to go back to"
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
        """Block until nothing has scrolled for a short quiet period.

        A wheel event is asynchronous and a page may animate the scroll, so the position
        right after ``mouse.wheel`` is one in motion. Capture-phase listening catches any
        scroller, not just the window; the deadline covers a page that scrolls forever.
        """
        page.evaluate(_SCROLL_QUIET_JS, [_SCROLL_QUIET_MS, _SCROLL_QUIET_DEADLINE_MS])

    def _settle(self, page: Page) -> None:
        """Give the page its moment to react, and to finish arriving.

        Waiting for ``document.readyState`` to reach ``complete`` is what catches a
        navigation a click set off; a bare ``wait_for_load_state`` can return mid-commit.
        Running out of ``settle_timeout_ms`` means settling is done, not that it failed.
        """
        if self._settle_ms > 0:
            page.wait_for_timeout(self._settle_ms)

        deadline = time.monotonic() + self._settle_timeout_ms / 1000.0
        while True:
            try:
                if page.evaluate("() => document.readyState") == "complete":
                    return
            except PlaywrightError:
                # The context vanished: a navigation committed, which is what we await.
                pass
            remaining_ms = (deadline - time.monotonic()) * 1000.0
            if remaining_ms <= 0:
                return
            try:
                page.wait_for_load_state("load", timeout=remaining_ms)
                # It can return mid-commit, so pause before re-probing.
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

    **The agent must never see this.** It knows what the agent is supposed to learn to
    see, so it has exactly two callers: the detector's labeller and the evaluation
    harness. That is why it is constructed explicitly and why :class:`BrowserController`
    hands one out nowhere; code that legitimately needs it takes it as an argument.

    Boxes are LOGICAL pixels in the controller's own space, clipped to the viewport, in
    reading order; invisible, zero-sized and off-screen elements are dropped. Main frame,
    light DOM only, and it does not test whether one element is painted over another.
    """

    def __init__(self, controller: Any) -> None:
        """``controller`` is any browser controller: it is asked only for ``evaluate`` and
        ``url``, which the Playwright and the Browser Harness controllers both answer."""
        self._controller = controller

    def elements(self) -> list[Element]:
        """Every visible element of the page, ``source=dom``, ``confidence=1.0``."""
        data = self._read_dom()

        width, height = data["viewport"]
        viewport = Box(0, 0, int(width), int(height))
        elements: list[Element] = []
        for raw in data["elements"]:
            # getBoundingClientRect is already CSS pixels; scale 1 keeps edge rounding
            # identical to every other path.
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
        return self._controller.url() or ""

    def _read_dom(self) -> dict:
        """Run the reader script. The controller's ``evaluate`` already asks once more
        when a navigation pulled the rug out, which is a moment and not a broken page."""
        return self._controller.evaluate(_GROUND_TRUTH_JS)


def _stable_id(kind: ElementKind, text: str, box: Box) -> str:
    """Identity for "the same element across observations of the same screen".

    Position is rounded to a 16-pixel grid, so a label that shifts a pixel or two keeps
    its identity while a different control never borrows it.
    """
    seed = f"{kind.value}|{text}|{box.x // 16}|{box.y // 16}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
