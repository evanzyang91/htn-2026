"""A ``contracts.Controller`` driving the person's OWN running Chrome, with no framework.

This is how ``jev_ultrafast/browser.py`` drives a browser, brought onto this project's
``Controller``: Browser Harness keeps one connection to the Chrome the person already
has open, and every capture, click and keystroke is a raw DevTools-protocol call down it.
Playwright is not imported, launches nothing and attaches to nothing.

Why it exists beside :class:`~skillweaver.controllers.browser.BrowserController`, whose
``attach`` mode already starts Chrome plainly: that mode still hands the page to
Playwright, on a PROFILE NOBODY HAS EVER USED. A site reads both. Here the browser is the
one a person browses with - its cookies, its history, its logins - and nothing is
injected into the page to drive it.

Which makes the thing to know before using it a warning rather than a feature: **this
drives the real, logged-in browser.** The run gets a tab of its own, opened in the
background and closed at the end, and touches no other tab - but that tab shares the
profile's sessions, so a task that says "buy" is addressed to a browser that can.
Nothing here defeats a human-verification page either; the rule in
:data:`~skillweaver.controllers.chrome_launch.PLAINLY_LAUNCHED` holds unchanged, and a
challenge still fails the run for a person to clear by hand.

Still blind and still POINT-based, like the Playwright controller: a click is a press and
a release at a logical pixel, and ``evaluate`` is the same read-only hook.
"""

from __future__ import annotations

import base64
import re
import sys
import time
from collections.abc import Sequence
from types import TracebackType
from typing import Any

from skillweaver.contracts import (
    Action,
    ActionKind,
    ActionResult,
    Back,
    Box,
    Click,
    Drag,
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
from skillweaver.errors import ControllerError
from skillweaver.logging_ import get_logger

log = get_logger(__name__)

__all__ = ["BLOCKED_URLS", "HarnessBrowserController"]

BLOCKED_URLS: tuple[str, ...] = (
    "*Special:BannerLoader*",
    "*Special:RecordImpression*",
    "*geoiplookup*",
)
"""``SOMETIMES_ONLY_OVERLAYS`` from :mod:`skillweaver.controllers.browser`, as the
wildcards ``Network.setBlockedURLs`` takes. A wildcard matches the whole URL, query
string included, which is where CentralNotice puts the name. Applied to this run's tab
only - the person's other tabs load whatever they load."""

_SUPPORTED_ACTIONS: frozenset[str] = frozenset(
    {"click", "move", "drag", "type_text", "press_key", "scroll", "wait", "navigate", "back"}
)

_MODIFIERS: dict[str, tuple[int, str, int]] = {
    "Alt": (1, "AltLeft", 18),
    "Control": (2, "ControlLeft", 17),
    "Meta": (4, "MetaLeft", 91),
    "Shift": (8, "ShiftLeft", 16),
}
"""Playwright's modifier names, as ``(CDP modifier bit, code, virtual key)``. Stored
skills are written in these names - ``ctx.ctl.press("Meta", "a")`` - so they are the
vocabulary this controller has to speak."""

_EDIT_COMMANDS: dict[str, str] = {
    "a": "selectAll",
    "c": "copy",
    "v": "paste",
    "x": "cut",
    "z": "undo",
}
"""What a Meta/Control chord MEANS, said out loud. A synthetic key event on macOS reaches
the page's listeners but not the menu that would have turned Cmd+A into a selection, so
without the command the chord in every typed move selects nothing and the field ends up
holding ``keycapskeycaps``. ``jev_ultrafast/browser.py`` passes the same command."""

_SCROLL_QUIET_JS = """(([quietMs, deadlineMs]) => new Promise((resolve) => {
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
}))([80, 2000])"""
"""The Playwright controller's scroll-quiet wait, as an expression."""

_CAPTURE_TIMEOUTS_S = (5.0, 20.0)
"""How long each ask for a frame may take. Two, because a heavy page in a background tab
sometimes never delivers the first frame and delivers the second at once - upstream
tolerates the loss because its policy reads no pictures, and this project cannot: the
pixel path, the critic and the inspector all look at the frame. For the same reason
upstream's 0.5s bound and its skip-frames backoff are NOT ported: they return ``None`` for
a frame, and ``Controller.capture`` has no ``None`` - every caller (``ComposedPerceiver``,
``DomPerceiver``, the explorer's and the inspector's wrappers) takes a ``Screenshot`` or a
``ControllerError``."""

_CAPTURE_PARAMS: dict[str, Any] = {"format": "png", "optimizeForSpeed": True}
"""``optimizeForSpeed`` picks the encoder's fast setting, and for PNG that is still
lossless. Measured 2026-09-20 on this controller's own background tab, headed Chrome,
1280x800, 8 interleaved pairs per page, median default -> fast: bbc.com 66 -> 43ms,
amazon.com 87 -> 48ms, Wikipedia Main Page 55 -> 36ms; same 1280x800 size and 0 of
1,024,000 pixels differing on all three, for 14-24% more bytes. That is not upstream's
~1000 -> ~40ms, which was its JPEG path on a page holding the compositor; it is one
encoder pass per observation. The OCR cache is keyed on these BYTES
(``CachingTextReader``), so what matters as much as the pixels is that the fast encoder
is deterministic - ``scripts/check_d_harness.py`` captures a still page twice and
compares."""

_NOT_BLANK_AND_COMPLETE = "location.href !== 'about:blank' && document.readyState === 'complete'"
_COMPLETE = "document.readyState === 'complete'"


class HarnessBrowserController:
    """Raw-CDP eyes and hands on one owned tab of the person's running Chrome.

    Args:
        viewport: ``(width, height)`` in LOGICAL pixels, set as a device-metrics override
            on the owned tab alone.
        settle_ms: Wait after each action for the page to react.
        navigation_timeout_ms: Ceiling on a single :class:`Navigate` or :class:`Back`.
        settle_timeout_ms: Ceiling on settling. Best-effort, so reaching it is no error.
        start_url: Loaded once at construction.
        block: URL wildcards refused on this tab. ``None`` means :data:`BLOCKED_URLS`.

    Raises:
        ControllerError: Browser Harness is not installed, or could not reach Chrome.
    """

    def __init__(
        self,
        *,
        viewport: tuple[int, int] = (1280, 800),
        settle_ms: float = 120.0,
        navigation_timeout_ms: float = 15_000.0,
        settle_timeout_ms: float = 3_000.0,
        start_url: str | None = None,
        block: Sequence[str] | None = None,
    ) -> None:
        width, height = viewport
        if width <= 0 or height <= 0:
            raise ValueError(f"viewport must be positive, got {viewport!r}")
        self._viewport = Box(0, 0, int(width), int(height))
        self._settle_ms = max(float(settle_ms), 0.0)
        self._navigation_timeout_ms = float(navigation_timeout_ms)
        self._settle_timeout_ms = max(float(settle_timeout_ms), 0.0)
        self._blocked: tuple[str, ...] = BLOCKED_URLS if block is None else tuple(block)
        self._target: str | None = None
        self._session: str | None = None
        self._closed = False
        self._acted_ms = 0.0
        try:
            from browser_harness.admin import ensure_daemon
            from browser_harness.helpers import cdp
        except ImportError as exc:
            raise ControllerError(
                "driving your own Chrome needs the browser-harness package; run `uv sync`"
            ) from exc
        self._cdp = cdp
        try:
            log.info(
                "harness.connect",
                note="if Chrome asks to allow remote debugging, allow it; the run waits",
            )
            ensure_daemon()
            # In the BACKGROUND: the person's visible tab stays theirs.
            self._target = cdp("Target.createTarget", url="about:blank", background=True)[
                "targetId"
            ]
            self._session = cdp("Target.attachToTarget", targetId=self._target, flatten=True)[
                "sessionId"
            ]
            self._call(
                "Emulation.setDeviceMetricsOverride",
                width=int(width),
                height=int(height),
                deviceScaleFactor=1,
                mobile=False,
            )
            # A hidden tab throttles animation frames and never opens a menu; this keeps
            # it rendering without taking the window's focus.
            self._call("Emulation.setFocusEmulationEnabled", enabled=True)
            if self._blocked:
                self._call("Network.enable")
                self._call("Network.setBlockedURLs", urls=list(self._blocked))
            if start_url is not None:
                error = self._navigate(start_url)
                if error is not None:
                    raise ControllerError(error)
        except ControllerError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise ControllerError(
                f"could not reach your running Chrome through Browser Harness: {_brief(exc)}. "
                "Run `uv run browser-harness --doctor`, and allow remote debugging in "
                "Chrome when it asks. There is no fallback to a framework-launched "
                "browser, which is the configuration a bot wall refuses."
            ) from exc

    # -- what a run records about its browser --------------------------------------------

    @property
    def headless(self) -> bool:
        """Always ``False``: this is a headed Chrome, and ``render_mode`` files it there."""
        return False

    @property
    def attached(self) -> bool:
        """Always ``True``: nothing here started the browser."""
        return True

    @property
    def blocked(self) -> tuple[str, ...]:
        """The URL wildcards refused on this tab."""
        return self._blocked

    def __enter__(self) -> HarnessBrowserController:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- Controller protocol -------------------------------------------------------------

    def capture(self) -> Screenshot:
        """Grab the owned tab's viewport. ``scale`` is measured from the PNG produced."""
        self._live()
        data: str | None = None
        failure: BaseException | None = None
        for timeout in _CAPTURE_TIMEOUTS_S:
            try:
                data = self._call(
                    "Page.captureScreenshot", _response_timeout=timeout, **_CAPTURE_PARAMS
                )["data"]
                break
            except (RuntimeError, TimeoutError, KeyError) as exc:
                failure = exc
        if data is None:
            raise ControllerError(f"screenshot failed: {_brief(failure)}")
        png = base64.b64decode(data)
        width, height = self._viewport.w, self._viewport.h
        physical_width, physical_height = _coords.png_size(png)
        scale = _coords.scale_for(physical_width, width)
        if abs(scale - _coords.scale_for(physical_height, height)) > 0.01 * scale:
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
        self._live()
        started = time.perf_counter()
        try:
            error = self._deliver(action)
        except TimeoutError as exc:
            error = f"{action.kind} timed out: {_brief(exc)}"
        except RuntimeError as exc:
            error = f"{action.kind} failed: {_brief(exc)}"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._acted_ms += elapsed_ms
        return ActionResult(ok=error is None, error=error, elapsed_ms=elapsed_ms)

    @property
    def acted_ms(self) -> float:
        """Milliseconds spent inside :meth:`perform` so far, as
        :attr:`BrowserController.acted_ms` counts them."""
        return self._acted_ms

    def quiesce(self, quiet_ms: float, cap_ms: float) -> float | None:
        """Wait until the page has been still for ``quiet_ms``, at most ``cap_ms``.
        The same script and the same ADVISORY contract as
        :meth:`BrowserController.quiesce`: ``None`` when the document went away."""
        self._live()
        try:
            return float(self._expression(f"({QUIESCE_JS})([{float(quiet_ms)}, {float(cap_ms)}])"))
        except (RuntimeError, TimeoutError, TypeError, ValueError):
            return None

    def viewport(self) -> Box:
        """The page area in logical pixels, anchored at ``(0, 0)``."""
        return self._viewport

    def supports(self, action_kind: ActionKind) -> bool:
        """True for every kind in the contract, ``navigate`` and ``back`` included."""
        return action_kind in _SUPPORTED_ACTIONS

    def url(self) -> str | None:
        """The address the tab is on, or ``None`` if it has none yet."""
        self._live()
        try:
            found = self._expression("location.href")
        except (RuntimeError, TimeoutError):
            return None
        return found if isinstance(found, str) and found != "about:blank" else None

    def evaluate(self, script: str) -> Any:
        """Run ``script`` - a function expression, as Playwright takes one - in the page
        and return its JSON-able result. READ-ONLY by the same rule as
        :meth:`BrowserController.evaluate`, and with the same one retry: a document
        mid-navigation is a moment, not a broken page.

        Raises:
            ControllerError: the script cannot be run, or fails twice.
        """
        self._live()
        expression = f"({script})()"
        try:
            return self._expression(expression)
        except (RuntimeError, TimeoutError):
            pass
        self._await_load(self._navigation_timeout_ms)
        try:
            return self._expression(expression)
        except (RuntimeError, TimeoutError) as exc:
            raise ControllerError(f"could not evaluate page script: {_brief(exc)}") from exc

    def describe(self) -> str:
        """One line for logs and prompts."""
        return f"browser-harness chrome+cdp {self._viewport.w}x{self._viewport.h} @1x headed"

    def close(self) -> None:
        """Close the tab this run opened. Idempotent, never raises, and touches nothing
        else: the browser is the person's and outlives every run."""
        self._closed = True
        target, self._target, self._session = self._target, None, None
        if target is not None:
            try:
                self._cdp("Target.closeTarget", targetId=target)
            except Exception:  # noqa: BLE001 - the tab or the browser is already gone
                pass

    # -- internals -----------------------------------------------------------------------

    def _live(self) -> None:
        if self._closed or self._session is None:
            raise ControllerError("browser controller is closed")

    def _call(self, method: str, **params: Any) -> dict[str, Any]:
        return self._cdp(method, session_id=self._session, **params)

    def _expression(self, expression: str) -> Any:
        """One ``Runtime.evaluate``, promises awaited.

        Raises:
            RuntimeError: the page threw, or the document went away underneath it.
        """
        response = self._call(
            "Runtime.evaluate",
            _response_timeout=max(self._navigation_timeout_ms / 1000.0, 5.0),
            expression=expression,
            returnByValue=True,
            awaitPromise=True,
        )
        details = response.get("exceptionDetails")
        if details:
            thrown = details.get("exception", {}).get("description") or details.get("text")
            raise RuntimeError(thrown or "the page script threw")
        return response.get("result", {}).get("value")

    def _deliver(self, action: Action) -> str | None:
        """Perform ``action``; return ``None`` on success or a reason it was refused."""
        for point in _points_of(action):
            if not _coords.within_viewport(self._viewport, point):
                return (
                    f"point ({point.x}, {point.y}) is outside the "
                    f"{self._viewport.w}x{self._viewport.h} viewport"
                )

        match action:
            case Click():
                if action.clicks < 1:
                    return f"click needs at least one click, got {action.clicks}"
                self._mouse("mouseMoved", action.point)
                # One press/release pair per click, counting up: what a double click IS.
                for count in range(1, action.clicks + 1):
                    for event in ("mousePressed", "mouseReleased"):
                        self._mouse(event, action.point, button=action.button, clickCount=count)
            case Move():
                self._mouse("mouseMoved", action.point)
            case Drag():
                self._mouse("mouseMoved", action.start)
                self._mouse("mousePressed", action.start, button="left", clickCount=1)
                steps = 12
                for step in range(1, steps + 1):
                    x = action.start.x + (action.end.x - action.start.x) * step / steps
                    y = action.start.y + (action.end.y - action.start.y) * step / steps
                    self._mouse("mouseMoved", Point(round(x), round(y)), button="left", buttons=1)
                self._mouse("mouseReleased", action.end, button="left", clickCount=1)
            case TypeText():
                self._call("Input.insertText", text=action.text)
            case PressKey():
                if not action.keys:
                    return "press_key needs at least one key"
                self._press_chord(action.keys)
            case Scroll():
                # A wheel event scrolls whatever is under the pointer it is given.
                self._mouse("mouseWheel", action.point, deltaX=action.dx, deltaY=action.dy)
                try:
                    self._expression(_SCROLL_QUIET_JS)
                except (RuntimeError, TimeoutError):
                    pass  # the scroll set off a navigation; settling below covers it
            case Wait():
                if action.ms < 0:
                    return f"wait needs a non-negative duration, got {action.ms}"
                time.sleep(action.ms / 1000.0)
                return None  # an explicit wait is its own settle
            case Navigate():
                return self._navigate(action.url)
            case Back():
                return self._go_back()
            case _:
                return f"action kind {action.kind!r} is not supported"

        self._settle()
        return None

    def _mouse(self, event: str, point: Point, **params: Any) -> None:
        self._call("Input.dispatchMouseEvent", type=event, x=point.x, y=point.y, **params)

    def _press_chord(self, keys: tuple[str, ...]) -> None:
        """Hold every key but the last, tap the last, then release in reverse order."""
        from browser_harness.helpers import _KEYS, _printable_key

        *held, final = keys
        bits = 0
        pressed: list[str] = []
        try:
            for key in held:
                bit, code, virtual = _MODIFIERS.get(key, (0, key, 0))
                bits |= bit
                self._key("rawKeyDown", key, code, virtual, bits)
                pressed.append(key)

            text = ""
            if final in _KEYS:
                virtual, code, text = _KEYS[final]
            elif final in _MODIFIERS:
                _, code, virtual = _MODIFIERS[final]
            elif len(final) == 1 and (resolved := _printable_key(final)) is not None:
                code, virtual, shifted = resolved
                text = final
                if shifted and not bits & 7:
                    bits |= 8
            else:
                code, virtual = final, 0
            shortcut = bool(bits & 7)
            extra: dict[str, Any] = {}
            if shortcut and final.lower() in _EDIT_COMMANDS and bits & (4 if _MAC else 2):
                extra["commands"] = [_EDIT_COMMANDS[final.lower()]]
            if text and not shortcut:
                # The text is what makes the key DO something: Enter's is "\r", and without
                # it the page hears a keydown and no keypress, so a form with no button -
                # GitHub's search, upstream's 120-steps-to-5 case - never submits.
                extra["text"] = text
            self._key("keyDown", final, code, virtual, bits, **extra)
            self._key("keyUp", final, code, virtual, bits)
        finally:
            for key in reversed(pressed):
                bit, code, virtual = _MODIFIERS.get(key, (0, key, 0))
                bits &= ~bit
                try:
                    self._key("keyUp", key, code, virtual, bits)
                except (RuntimeError, TimeoutError):  # the page went away
                    pass

    def _key(self, event: str, key: str, code: str, virtual: int, bits: int, **extra: Any) -> None:
        self._call(
            "Input.dispatchKeyEvent",
            type=event,
            key=key,
            code=code,
            modifiers=bits,
            windowsVirtualKeyCode=virtual,
            nativeVirtualKeyCode=virtual,
            **extra,
        )

    def _navigate(self, url: str) -> str | None:
        """Load ``url`` and wait for ITS load event, not the one ``about:blank`` already had.

        The tab is opened on ``about:blank``, a document that reads ``complete`` for ever,
        so a readiness poll that reaches it returns at once and the first observation is
        of a blank page - upstream measured exactly that (Amazon: blocked at 0 steps).
        Here the hole did NOT reproduce, measured 2026-09-20 on a plainly-launched Chrome
        through this same daemon: the first ``location.href`` read after ``Page.navigate``
        answered was already the new document in 55 of 55 loads from ``about:blank``
        (a local page served after 0 and 400ms, Wikipedia, bbc.com, amazon.com,
        github.com; headed and headless), because the call answers at COMMIT - 409ms for
        the 400ms page - and not at the request. ``Back`` likewise: the first read, 3-39ms
        after ``Page.navigateToHistoryEntry``, was the entry gone back to, 6 of 6. So the
        requirement below has never yet been what held a load; it is kept because it is
        free when the document has already changed, and because what it guards against is
        a run decided on nothing. It is asked only of a navigation that is meant to LEAVE
        ``about:blank``: not one aimed at it, and not an aborted one, where the blank tab
        is the page that is showing.
        """
        try:
            answer = self._call(
                "Page.navigate",
                _response_timeout=max(self._navigation_timeout_ms / 1000.0, 5.0),
                url=url,
            )
        except TimeoutError as exc:
            self._settle()
            return f"navigate timed out: {_brief(exc)}"
        except RuntimeError as exc:
            self._settle()
            return f"navigate failed: {_brief(exc)}"
        failed = answer.get("errorText")
        if failed and "ERR_ABORTED" not in failed:
            # ERR_ABORTED is what a download or a redirect-to-app reports; the page that
            # is showing is still the answer, and the next observation will see it.
            self._settle()
            return f"navigate failed: {failed}"
        leaves_blank = not failed and url.strip().lower() != "about:blank"
        if not self._await_load(self._navigation_timeout_ms, off_blank=leaves_blank):
            return f"navigate timed out: {url} had not finished loading"
        self._settle()
        return None

    def _go_back(self) -> str | None:
        """Pop one entry off the tab's session history.

        Chrome's own history is the authority on an empty stack - a page's
        ``history.length`` counts both directions - and the ``about:blank`` the tab was
        opened on is not a page to go back TO, so both are REFUSED rather than reported
        as a move, which would tell a policy it went back onto this screen.
        """
        try:
            history = self._call("Page.getNavigationHistory")
            index = int(history["currentIndex"])
            entries = history["entries"]
            if index <= 0 or entries[index - 1].get("url") == "about:blank":
                return "there is nothing behind this page to go back to"
            self._call("Page.navigateToHistoryEntry", entryId=entries[index - 1]["id"])
        except (RuntimeError, TimeoutError, KeyError, IndexError) as exc:
            self._settle()
            return f"back failed: {_brief(exc)}"
        if not self._await_load(self._navigation_timeout_ms):
            return "back timed out"
        self._settle()
        return None

    def _await_load(self, timeout_ms: float, *, off_blank: bool = False) -> bool:
        """Poll until the document is ``complete``; ``False`` if ``timeout_ms`` ran out.
        A failed probe is a navigation committing, which is what is being awaited.
        ``off_blank`` also requires the document to be something other than the
        ``about:blank`` the tab was opened on - see :meth:`_navigate`. Never for a
        ``Back`` or a settle: both can rightly end on a page that was there all along."""
        ready = _NOT_BLANK_AND_COMPLETE if off_blank else _COMPLETE
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            try:
                if self._expression(ready) is True:
                    return True
            except (RuntimeError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.025)

    def _settle(self) -> None:
        """Give the page its moment to react, and to finish arriving. Running out of
        ``settle_timeout_ms`` means settling is done, not that it failed."""
        if self._settle_ms > 0:
            time.sleep(self._settle_ms / 1000.0)
        self._await_load(self._settle_timeout_ms)


_MAC = sys.platform == "darwin"


def _points_of(action: Action) -> tuple[Point, ...]:
    """Every viewport coordinate an action will touch, for bounds checking."""
    if isinstance(action, Click | Move | Scroll):
        return (action.point,)
    if isinstance(action, Drag):
        return (action.start, action.end)
    return ()


def _brief(exc: BaseException | None) -> str:
    text = re.sub(r"\s+", " ", str(exc or "")).strip()
    return (text or (exc.__class__.__name__ if exc else "no answer"))[:200]
