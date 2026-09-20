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

A site that refuses an automated browser needs the REAL Chrome and a profile that
outlives the run - ``BrowserController(user_data_dir=...)``, and
:data:`REAL_CHROME_CHANNEL` for what that is and what it is measured to fix. A site that
refuses even THAT needs a Chrome the framework did not start:
``BrowserController(user_data_dir=..., attach=True)``, and
:data:`~skillweaver.controllers.chrome_launch.PLAINLY_LAUNCHED` for the three-way
measurement that says why.
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
from skillweaver.controllers.chrome_launch import PLAINLY_LAUNCHED, ChromeProcess
from skillweaver.errors import ControllerError

BrowserName = Literal["chromium", "firefox", "webkit"]

SOMETIMES_ONLY_OVERLAYS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Special:BannerLoader"),
    re.compile(r"Special:RecordImpression"),
    re.compile(r"geoiplookup"),
)
"""Requests that serve an overlay a page renders only SOMETIMES, aborted by default.

A measurement wants one page to be one screen. Everything downstream of this
controller compares two screens of the same URL - the admission gate against the
screen the recording started on, the critic against the screen before the move,
``find_route`` against where it thinks it already is - and every one of those
comparisons is a lie when a page is a full-width appeal on one load and not on the
next. The identity cannot be taught to forgive this one: see ``SAME_STATE_THRESHOLD``
in :mod:`skillweaver.perception.fingerprint`, which refuses a full-screen takeover on
purpose, because a screen that is two thirds gone is not that screen.

Measured on live ``en.wikipedia.org`` on 2026-09-19. Thirty plain loads of
``/wiki/Main_Page``, each in a fresh context and no blocking: FIVE rendered a
CentralNotice fundraising appeal 531px tall in an 800px viewport and twenty-five
rendered nothing, and the two groups fingerprint against each other at 0.132 - far
below the 0.26 same-state cut. That is the whole defect: of fifteen live learning runs
of one task, the two that met the appeal both threw away a correct skill, one because
the recorded starting screen could no longer be stood on (0.09) and one because it
never was (0.12). Ten loads with these patterns aborted rendered it zero times.

Waiting for a one-in-six event is no way to check this, so force it:
``?banner=<name>&force=1`` renders the appeal on every load. Ten forced loads with
nothing blocked were the appeal ten times out of ten and the ordinary screen zero
times out of ten; ten with the block on were the ordinary screen ten times out of ten.

**Regexes, not the glob patterns this looks like it should use.** CentralNotice serves
the appeal from ``meta.wikimedia.org/w/index.php?title=Special:BannerLoader&...`` -
the name is in the QUERY STRING, and a Playwright glob matches path segments, so
``**/Special:BannerLoader*`` matches nothing on a real Wikipedia load. On one forced
load the glob aborted 0 requests and the appeal rendered at 531px; the regex aborted 1
and it did not render at all.

This is not ad-blocking for its own sake and it hides nothing a run needs: the page is
fully readable and every link on it still works. Pass ``block=()`` to a controller
that is deliberately measuring the overlay itself, or patterns of your own - a regex,
or a `Playwright URL glob <https://playwright.dev/python/docs/network>`_ where the
thing to block really is a path - for another site's version of the same problem.
"""

REAL_CHROME_CHANNEL = "chrome"
"""Playwright's name for the Google Chrome INSTALLED ON THIS MACHINE, as opposed to the
Chromium build Playwright ships with. Half of what it takes to be served by a real site.

A REAL SITE CAN REFUSE AN AUTOMATED BROWSER OUTRIGHT, and doordash.com does: a plain
``chromium.launch()``, whose profile Playwright throws away after the run, gets
Cloudflare's "Verify you are human" on every URL, while an ordinary Chrome window on the
same machine loads the site. TWO ordinary browser settings are what close that gap:

1. this channel - the real Google Chrome build, not bundled Chromium;
2. ``launch_persistent_context(<dir>)`` - a profile directory that SURVIVES the run, so
   whatever a site stored on one run is still there on the next.

Measured on 2026-09-19 against live doordash.com, signed out, no address entered: the
front page came back titled ``DoorDash: Food, Grocery and Retail - Fast Same Day
Delivery``, a city listing page carried 32k characters of real stores and a store page a
real priced menu - no interstitial on any of them.

**That access is not a guarantee, and this mode does not make it one.** Later the same
day the same profile got the interstitial on every load, and a bare hand-probe with no
controller in it got the same - so what changed was the site's opinion of us, not the
launch. Two causes we could see, both our own doing: MANY loads from one address in a
few minutes, and SHARING one profile directory between concurrent runs, which fight over
the lock and leave the stored clearance unreliable. Give each run its own directory, and
do not hammer a site to find out whether it is still letting you in.

**Nothing here masks automation and nothing here may.** No
``--disable-blink-features=AutomationControlled``, no spoofed fingerprint, no
user-agent edit, no proxy, no retry-until-it-passes loop: re-measured WITHOUT any such
argument, the two settings above carried the whole result on their own, and
``navigator.webdriver`` stays true. If a human-verification page appears, the run fails
and says so, and a person clears it by hand, once, in the profile - which is exactly what
a profile that outlives the run is for.

What this mode is, then, is a browser LAUNCH configuration, and it is correct or not on
its own terms: it opens the real Chrome, on a profile that persists, with the viewport,
scale, blocking and start URL every other run gets. Whether a particular site then serves
that browser is the site's business."""

_SUPPORTED_ACTIONS: frozenset[str] = frozenset(
    {"click", "move", "drag", "type_text", "press_key", "scroll", "wait", "navigate", "back"}
)
"""Every action kind in the contract. A browser can do all of them, ``navigate`` and
``back`` included - those are the two a raw desktop controller does not have, because
both are session history and a desktop has none."""


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


def _abort(route: Route) -> None:
    """Refuse one request. A route handler that raises kills the page, and a
    request can be gone - the page navigated away - by the time this runs, so the
    abort is best-effort and a failure to abort is not an error."""
    try:
        route.abort()
    except Exception:  # noqa: BLE001 - a request that is already gone is blocked enough
        pass


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
            will report it back. It survives ``user_data_dir`` and is not dropped
            there, but it becomes a LAUNCH option rather than a per-context one: a
            persistent profile IS the browser, so one process can hold exactly one
            scale and there is no second context to give another. Measured on real
            Chrome, headed and headless alike, a 640x400 viewport at ``2.0`` captures
            1280x800 with ``devicePixelRatio`` 2, on the page the browser opened
            itself - which is the page this controller adopts.
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
        user_data_dir: A PERSISTENT Chrome profile directory. Given, the controller
            drives the REAL Google Chrome installed on this machine out of that
            directory instead of a bundled Chromium with a throwaway profile, which
            is what it takes to be served by a site that refuses an automated browser
            - see :data:`REAL_CHROME_CHANNEL`. The directory is created if it is
            missing, is NOT deleted on teardown - persisting cookies between runs is
            the whole point - and is exclusive: Chrome locks a profile, so a window
            already open on it makes the launch fail rather than share it. ``None``
            leaves today's launch path exactly as it was.
        attach: Start that real Chrome as an ORDINARY PROCESS and attach to it over
            Chrome's own debugging interface, rather than letting Playwright launch it.
            Requires ``user_data_dir``. This is the only one of the three launch
            configurations a live DoorDash serves, and what it changes is the flags
            Playwright adds when IT starts the browser - see
            :data:`~skillweaver.controllers.chrome_launch.PLAINLY_LAUNCHED` for the
            measurement, and for the boundary this mode must never be extended past.
            ``headless``, ``block``, ``start_url`` and ``device_scale_factor`` all behave
            as they do in the other modes; the ONE difference is that ``viewport`` is
            applied to the adopted page rather than given to a context that does not
            exist here, so a headed window is sized to match but the page is what is
            authoritative.
        chrome_binary: Where Google Chrome is, for ``attach``, or ``None`` to look in
            this platform's usual places.
        start_url: Loaded once at construction, if given.
        block: URL patterns whose requests are aborted, on the context, so they
            are blocked for every page and every navigation. A regex matches
            anywhere in the URL; a string is a Playwright path glob. ``None``
            means :data:`SOMETIMES_ONLY_OVERLAYS` - read that constant before
            changing it; ``()`` blocks nothing.

    Raises:
        ControllerError: if the browser cannot be launched. A ``user_data_dir`` whose
            real Chrome is missing fails HERE, by name; it never quietly falls back to
            the bundled Chromium, which is the build the bot wall blocks.
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
            # Chromium animates wheel scrolling, so a scroll would still be moving
            # when perform() returned and the next capture would catch a half-
            # scrolled frame. Turning the animation off makes a wheel land at once;
            # _await_scroll_quiet below covers pages that animate scrolling
            # themselves, and engines where this flag does not exist.
            args = ["--disable-smooth-scrolling"] if browser == "chromium" else []
            view = {"width": int(width), "height": int(height)}
            if self._attach:
                # Playwright does not start this browser: an ordinary Chrome process
                # does, and Playwright is handed the address it published. ``args``
                # above belongs to a launch that is not happening here, so the same
                # scroll setting is passed to the process instead - see
                # :meth:`ChromeProcess._spawn`.
                assert self._profile_dir is not None  # checked above
                self._chrome = ChromeProcess(
                    user_data_dir=self._profile_dir,
                    headless=headless,
                    binary=chrome_binary,
                    device_scale_factor=device_scale_factor,
                    window=(int(width), int(height)),
                )
                self._browser = engine.connect_over_cdp(self._chrome.endpoint)
                # Take the context and page the browser already has. Making more is how
                # a run ends up driving a blank tab while the demo watches another.
                contexts = self._browser.contexts
                self._context = contexts[0] if contexts else self._browser.new_context()
            elif self._profile_dir is not None:
                # The persistent context IS the launch: there is no Browser to make a
                # second context on, and ``self._browser`` stays None. Closing the
                # context is what shuts the process down.
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
            # On the CONTEXT, not the page: the route then survives every
            # navigation and covers a popup the page opens, which is where an
            # appeal reappears if the block is installed one page at a time.
            for pattern in self._blocked:
                self._context.route(pattern, _abort)
            # Real Chrome opens a window of its own, so a persistent context already
            # has a page. ADOPT it rather than opening a second one: the one nobody
            # drives would stay on screen for the whole demo.
            existing = self._context.pages if self._profile_dir is not None else []
            self._page = existing[0] if existing else self._context.new_page()
            if self._attach:
                # The viewport is a property of the LAUNCH in the other two modes. Here
                # the launch was an ordinary one, so the size is set on the page that
                # came back - which is what every capture and every coordinate is
                # measured against anyway.
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

        Readable because the answer OUTLIVES the browser: a screen recorded here is
        stored and compared later, and the two modes render one page differently
        enough that a cross-mode comparison cannot succeed. ``describe()`` says the
        same thing in prose for a prompt; this says it as a fact for
        :func:`skillweaver.render_mode.mode_of` to read, so nothing has to parse a
        sentence written for a human.
        """
        return self._headless

    @property
    def profile_dir(self) -> Path | None:
        """The persistent Chrome profile this controller drives, or ``None`` for the
        ordinary bundled-Chromium launch. Readable for the same reason as
        :attr:`headless`: it changes which browser rendered a screen, and a person
        asking why a site served them has to be able to see which one they got."""
        return self._profile_dir

    @property
    def attached(self) -> bool:
        """Whether this browser was started as an ordinary process and attached to,
        rather than launched by Playwright. Readable for the same reason as
        :attr:`headless` and :attr:`profile_dir`: it is the difference between a screen a
        live site served and an interstitial it served instead."""
        return self._attach

    @property
    def cdp_endpoint(self) -> str | None:
        """``http://127.0.0.1:<port>`` for the attached browser, or ``None`` in the
        other modes.

        Readable so this controller is not the only thing that can use the browser it
        opened: anything else that speaks Chrome's debugging protocol - a policy that
        reads the DOM rather than the pixels, a person with DevTools - attaches to the
        same process at this address. The process stays owned here and dies with this
        controller, so a second user of it is a guest for the run, not an owner.
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

    @property
    def blocked(self) -> tuple[str | re.Pattern[str], ...]:
        """The URL patterns this controller aborts. See :data:`SOMETIMES_ONLY_OVERLAYS`."""
        return self._blocked

    def evaluate(self, script: str) -> Any:
        """Run ``script`` in the page and return its JSON-able result. READ-ONLY.

        The one hook the DOM perception path needs
        (:class:`~skillweaver.perception.dom.DomPerceiver`), and deliberately the only
        thing that path adds to this class. It is not an action plane and must not
        become one: a click still goes through :meth:`perform` as a
        :class:`~skillweaver.contracts.Click` at a point, so Playwright, the launch path
        and every stored skill are untouched by the choice of eyes. See ``AGENTS.md``
        for the standing rule this serves and how far it is relaxed.

        It is not on the :class:`~skillweaver.contracts.Controller` Protocol - that is
        shared surface, and a desktop controller has no page to evaluate anything in -
        so callers duck-type on its presence and say so when it is absent.

        A document that is mid-navigation destroys the execution context under the
        script. That is a moment rather than a broken page, so the new document is
        waited for and the script asked once more, exactly as
        :class:`BrowserGroundTruth` does; a second failure is real.

        Raises:
            ControllerError: if the script cannot be run, or fails twice.
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

    def describe(self) -> str:
        """One line for logs and prompts, e.g. ``playwright chromium 1280x800 @2x``.

        A persistent profile says ``chrome`` rather than ``chromium``, because that is
        the build a page was actually served to, and an attached one says ``chrome+cdp``,
        because being started plainly is what a live site is reading.
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
        """Shut the page, context, browser and Playwright driver down. Idempotent
        and never raises: closing a half-dead browser must not mask the real error.

        A persistent profile directory is LEFT ON DISK. There is no browser handle to
        close in that mode - the context is the process - and the cookies in that
        directory are what the next run needs; deleting it would put the agent back
        behind the bot wall one run later.

        An ATTACHED browser is torn down the other way round. Closing its page or its
        context would be asking a browser this controller does not own to dismantle
        itself, so those handles are only dropped; what actually ends the browser is
        killing the process this run started, which happens last and happens even when
        the page is already wedged. A Chrome this run did not start is never touched.
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
        """Why the browser did not open, named precisely enough to act on.

        A missing real Chrome is its own sentence: the alternative would be to retry
        on bundled Chromium, and that is the build a site's bot wall turns away, so a
        silent fallback would trade a loud failure for a run that is quietly blocked.
        """
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

    def _go_back(self, page: Page) -> str | None:
        """Pop one entry off the page's session history.

        Playwright answers ``None`` when there was nothing behind the current page, and
        that is the only authority on the question - a page's own ``history.length``
        counts entries in both directions and never says where in the stack you are. So
        an empty stack is REFUSED here rather than reported as a move that happened,
        which is what keeps a policy from being told a back succeeded onto the same
        screen it was already on.
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
        """Block until nothing on the page has scrolled for a short quiet period.

        A wheel event is delivered asynchronously and a page may animate the
        scroll it triggers, so the position right after ``mouse.wheel`` is a
        position in motion. Listening for scroll events in the capture phase
        catches any scroller, not just the window, and the deadline guarantees
        this returns even on a page that scrolls forever.
        """
        page.evaluate(_SCROLL_QUIET_JS, [_SCROLL_QUIET_MS, _SCROLL_QUIET_DEADLINE_MS])

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
        """
        if self._settle_ms > 0:
            page.wait_for_timeout(self._settle_ms)

        deadline = time.monotonic() + self._settle_timeout_ms / 1000.0
        while True:
            try:
                if page.evaluate("() => document.readyState") == "complete":
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
