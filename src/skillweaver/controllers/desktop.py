"""``DesktopController``: real screen capture and a real cursor on this machine.

This is what makes "general computer use" more than a claim about browsers. It
captures with `mss <https://github.com/BoboTiG/python-mss>`_ and acts with
`pyautogui <https://pyautogui.readthedocs.io>`_, and every coordinate it exchanges
with the rest of skillweaver is a LOGICAL pixel relative to the top-left of
:meth:`DesktopController.viewport`.

Why the scale factor is the whole story
---------------------------------------

A Retina Mac has two physical bitmap pixels per logical pixel in each direction, so
a 1512x982 display grabs as a 3024x1964 image while the pointer APIs still address
it as 1512x982. A capture reported at its *bitmap* size makes every detection come
back at twice its true offset, every click lands in the wrong half of the screen,
and the whole thing reads as a model that cannot aim - which is why this lives in
its own module with :mod:`skillweaver.controllers.scaling` next to it.

So :meth:`capture` never assumes. It grabs the display, compares the bitmap size it
got against the display's logical bounds, and reports the measured ratio as
``Screenshot.scale`` with ``width``/``height`` in logical pixels. A machine that
turns out to be 1x, an external monitor that really is 1x, or a backend configured
for nominal resolution all come out right without a special case.

macOS permissions, which fail silently
--------------------------------------

Screen capture and synthetic input are both gated behind permissions a human grants
by hand, and neither raises when it is missing:

* **Screen Recording** withheld: the capture succeeds and returns a uniform image
  (black, or just the desktop picture). :meth:`capture` detects a single-colour
  frame and raises :class:`~skillweaver.errors.PerceptionError` naming the
  permission and where to grant it, rather than handing perception a black rectangle
  to hallucinate over.
* **Accessibility** withheld: the pointer events post and simply do nothing.
  :meth:`perform` reads the cursor back after moving it and reports
  ``ActionResult(ok=False, error=...)`` naming that permission when it did not land.

Neither check tries to grant anything, and nothing in the default test run touches a
real screen, so importing or unit-testing this module never raises a permission
prompt. Set ``SKILLWEAVER_LIVE_DESKTOP=1`` to opt into the live tests.

Multiple displays
-----------------

``DesktopController(display=N)`` drives one display, numbered as ``mss`` numbers
them: ``1`` is the primary, ``2`` and up are the others, and ``0`` is the union of
them all. :meth:`viewport` reports that display's bounds in global logical pixels -
a display to the left of the primary has a negative ``x`` - while the coordinates of
an action stay relative to the viewport's top-left, as
``contracts.Controller.viewport`` requires. :func:`scaling.to_global` does the
translation in one place.

One backing scale factor is measured per capture and applies to the whole captured
region. That is exact for a single display, which is all the demo needs. Capturing
``display=0`` across a Retina laptop screen and a 1x external monitor would give a
bitmap the OS has already normalised to one resolution, so the single ratio is
still self-consistent, but element boxes on the lower-density display would be
about as accurate as its own pixels allow and no better. Drive each display with
its own controller when that matters.

Example::

    with DesktopController() as ctl:
        shot = ctl.capture()                       # logical size, measured scale
        ctl.perform(Click(Point(40, 12)))          # logical, viewport-relative
"""

from __future__ import annotations

import contextlib
import io
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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
from skillweaver.controllers import scaling
from skillweaver.errors import ControllerError, PerceptionError

__all__ = [
    "KEY_ALIASES",
    "LIVE_ENV_VAR",
    "SCROLL_PIXELS_PER_CLICK",
    "DesktopController",
    "MssGrabber",
    "Pointer",
    "PyAutoGuiPointer",
    "RawCapture",
    "ScreenGrabber",
]

LIVE_ENV_VAR = "SKILLWEAVER_LIVE_DESKTOP"
"""Set this to ``1`` to enable the tests that touch the real screen and cursor."""

_UNSUPPORTED_ACTIONS: frozenset[str] = frozenset({"navigate", "back"})
"""The action kinds a desktop cannot perform, named rather than inferred.

Both are session history: there is no address bar to load a URL from and no stack to
pop. Naming them is what makes a NEW action kind land here as unsupported-until-taught
rather than as silently supported, which is how a controller comes to claim a move it
would then have to fake.
"""

SCROLL_PIXELS_PER_CLICK = 40
"""Logical pixels of content movement one wheel 'click' is taken to be.

``Scroll`` speaks pixels because that is what a browser and a vision model speak,
but the macOS wheel event speaks lines, and how far a line scrolls is up to the
application receiving it. Forty is a typical line height and makes the conversion
predictable; a scroll is a nudge in a direction, not a measured displacement.
"""

SCREEN_RECORDING_HINT = (
    "screen capture returned a single flat colour, which on macOS almost always means "
    "the Screen Recording permission is missing - grant it to this terminal (or to the "
    "app hosting this process) in System Settings > Privacy & Security > Screen & System "
    "Audio Recording, then restart the process. If the screen really is one colour "
    "(a blank desktop, a sleeping display), pass detect_blank=False."
)

ACCESSIBILITY_HINT = (
    "the cursor did not move, which on macOS almost always means the Accessibility "
    "permission is missing - grant it to this terminal (or to the app hosting this "
    "process) in System Settings > Privacy & Security > Accessibility, then restart "
    "the process"
)

KEY_ALIASES: dict[str, str] = {
    # contracts.PressKey uses Playwright's vocabulary; pyautogui has its own.
    "enter": "enter",
    "return": "enter",
    "tab": "tab",
    "escape": "esc",
    "esc": "esc",
    "backspace": "backspace",
    "delete": "delete",
    "space": "space",
    " ": "space",
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "pageup": "pageup",
    "pagedown": "pagedown",
    "home": "home",
    "end": "end",
    "insert": "insert",
    "control": "ctrl",
    "ctrl": "ctrl",
    "meta": "command",
    "command": "command",
    "cmd": "command",
    "shift": "shift",
    "alt": "option",
    "option": "option",
    "capslock": "capslock",
}
"""Maps a lowercased contract key name to its pyautogui name.

Anything not listed falls through as-is once lowercased, which covers single
characters (``"a"``) and the function keys (``"F5"`` -> ``"f5"``). A name pyautogui
does not know is refused by :meth:`DesktopController.perform` rather than silently
dropped.
"""


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def _native_image_options(enabled: bool) -> Iterator[None]:
    """Temporarily ask the macOS capture backend for the display's real pixel density.

    ``mss`` defaults to ``kCGWindowImageNominalResolution``, which hands back a
    logical-sized bitmap - correct, but soft, and the detector and OCR downstream
    both want the full Retina detail. The flag is a module global that ``mss`` reads
    at grab time and documents as the supported way to turn scaling on, so this sets
    it for the duration of one grab and always puts it back. A no-op off macOS.
    """
    import sys

    if not enabled or sys.platform != "darwin":
        yield
        return

    from mss import darwin

    previous = darwin.IMAGE_OPTIONS
    darwin.IMAGE_OPTIONS = 0
    try:
        yield
    finally:
        darwin.IMAGE_OPTIONS = previous


@dataclass(frozen=True, slots=True)
class RawCapture:
    """One grabbed bitmap, straight from the screen backend.

    ``rgb`` is ``width * height * 3`` bytes of 8-bit RGB. ``width`` and ``height``
    are PHYSICAL pixels, which is what makes this type worth having: nothing else in
    the project is allowed to be.
    """

    rgb: bytes
    width: int
    height: int


@runtime_checkable
class ScreenGrabber(Protocol):
    """The screen-capture half of the controller, injectable so tests never grab a
    real screen."""

    def displays(self) -> list[Box]:
        """Every display's bounds in global LOGICAL pixels, ``mss``-ordered: index
        ``0`` is the union of them all and ``1`` is the primary.

        Raises:
            ControllerError: if the displays cannot be enumerated.
        """
        ...

    def grab(self, bounds: Box) -> RawCapture:
        """Capture the region ``bounds`` (global LOGICAL pixels) at whatever
        resolution the backend natively provides.

        Raises:
            ControllerError: if the capture fails.
        """
        ...

    def close(self) -> None:
        """Release the capture handle. Idempotent; never raises."""
        ...


@runtime_checkable
class Pointer(Protocol):
    """The input half of the controller, injectable so tests never move a real
    cursor. All coordinates are global LOGICAL pixels."""

    def position(self) -> tuple[int, int]: ...

    def move_to(self, x: int, y: int) -> None: ...

    def click(self, x: int, y: int, *, button: str, clicks: int) -> None: ...

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None: ...

    def type_text(self, text: str) -> None: ...

    def key_down(self, key: str) -> None: ...

    def key_up(self, key: str) -> None: ...

    def scroll(self, x: int, y: int, *, horizontal: int, vertical: int) -> None:
        """Scroll by wheel clicks at ``(x, y)``. Positive ``vertical`` scrolls the
        view UP and positive ``horizontal`` scrolls it LEFT, matching the underlying
        wheel-event convention rather than the ``Scroll`` action's."""
        ...

    def is_typable(self, text: str) -> str:
        """The characters of ``text`` this backend would silently drop, as a string;
        empty when it can type all of them."""
        ...

    def knows_key(self, key: str) -> bool:
        """Whether ``key`` is a key name this backend can press."""
        ...


class MssGrabber:
    """:class:`ScreenGrabber` backed by ``mss``.

    ``native_resolution=True`` (the default) asks the macOS backend for the full
    Retina bitmap instead of the nominal-resolution one it prefers, because the
    detector and the OCR downstream both want the sharper image. Either way
    :meth:`DesktopController.capture` measures the scale from what it actually got,
    so this is a quality knob and never a correctness one.
    """

    def __init__(self, *, native_resolution: bool = True) -> None:
        import mss

        self._native_resolution = native_resolution
        try:
            self._mss = mss.MSS()
        except Exception as exc:  # noqa: BLE001 - backend failures are all fatal here
            raise ControllerError(f"could not open a screen capture session: {exc}") from exc
        self._closed = False

    def displays(self) -> list[Box]:
        if self._closed:
            raise ControllerError("screen grabber is closed")
        try:
            monitors = self._mss.monitors
        except Exception as exc:  # noqa: BLE001
            raise ControllerError(f"could not enumerate displays: {exc}") from exc
        return [Box(m["left"], m["top"], m["width"], m["height"]) for m in monitors]

    def grab(self, bounds: Box) -> RawCapture:
        if self._closed:
            raise ControllerError("screen grabber is closed")
        region = {"left": bounds.x, "top": bounds.y, "width": bounds.w, "height": bounds.h}
        with _native_image_options(self._native_resolution):
            try:
                shot = self._mss.grab(region)
            except Exception as exc:  # noqa: BLE001
                raise ControllerError(f"screen capture failed for {bounds}: {exc}") from exc
        return RawCapture(rgb=bytes(shot.rgb), width=shot.width, height=shot.height)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._mss.close()
        except Exception:  # noqa: BLE001 - close never raises
            pass


class PyAutoGuiPointer:
    """:class:`Pointer` backed by ``pyautogui``, with its failsafe left on.

    Moving the real cursor into a screen corner raises pyautogui's
    ``FailSafeException`` and aborts whatever the agent was doing - the last-resort
    way for a human to take the machine back. :class:`DesktopController` turns that
    into ``ActionResult(ok=False, ...)`` rather than letting it escape, so the run
    stops cleanly instead of crashing.
    """

    def __init__(self, *, failsafe: bool = True, move_duration_s: float = 0.0) -> None:
        import pyautogui

        pyautogui.FAILSAFE = failsafe
        self._gui = pyautogui
        self._move_duration_s = move_duration_s

    def position(self) -> tuple[int, int]:
        point = self._gui.position()
        return (int(point.x), int(point.y))

    def move_to(self, x: int, y: int) -> None:
        self._gui.moveTo(x, y, duration=self._move_duration_s)

    def click(self, x: int, y: int, *, button: str, clicks: int) -> None:
        self._gui.click(x=x, y=y, clicks=clicks, button=button, interval=0.05)

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        self._gui.moveTo(start_x, start_y, duration=self._move_duration_s)
        self._gui.mouseDown(button="left")
        try:
            # A drag with no duration is often dropped: the OS sees a teleport rather
            # than a gesture, so give it a moment of travel to follow.
            self._gui.moveTo(end_x, end_y, duration=max(self._move_duration_s, 0.2))
        finally:
            self._gui.mouseUp(button="left")

    def type_text(self, text: str) -> None:
        self._gui.write(text, interval=0.01)

    def key_down(self, key: str) -> None:
        self._gui.keyDown(key)

    def key_up(self, key: str) -> None:
        self._gui.keyUp(key)

    def scroll(self, x: int, y: int, *, horizontal: int, vertical: int) -> None:
        if vertical:
            self._gui.scroll(vertical, x=x, y=y)
        if horizontal:
            self._gui.hscroll(horizontal, x=x, y=y)

    def is_typable(self, text: str) -> str:
        known = self._gui.KEYBOARD_KEYS
        return "".join(dict.fromkeys(c for c in text if c not in known))

    def knows_key(self, key: str) -> bool:
        return key in self._gui.KEYBOARD_KEYS


# --------------------------------------------------------------------------------------
# The controller
# --------------------------------------------------------------------------------------


class DesktopController:
    """A ``contracts.Controller`` over one real display of this machine.

    Args:
        display: Which display to drive, ``mss``-numbered (``1`` is the primary,
            ``0`` is every display unioned).
        settle_ms: How long to wait after an action for the UI to catch up. Applied
            to every action except :class:`~skillweaver.contracts.Wait`, which is
            already a wait.
        detect_blank: Whether :meth:`capture` should treat a single-colour frame as
            a missing Screen Recording permission. Turn it off only when the screen
            legitimately is one colour.
        verify_pointer: Whether :meth:`perform` should read the cursor back after
            moving it and fail the action when it did not land, which is how a
            missing Accessibility permission shows itself.
        native_resolution: Ask the capture backend for the full-density bitmap.
        failsafe: Leave pyautogui's corner failsafe on. Defaults to ``True`` and
            there is no good reason to change it.
        move_duration_s: Seconds a pointer move takes. ``0.0`` teleports, which is
            fastest and fine for most UIs.
        grabber / pointer: Backend overrides. Both default to the real ones; tests
            pass doubles so the suite never touches the screen.

    Raises:
        ControllerError: if the display cannot be found or the backends cannot start.
    """

    def __init__(
        self,
        *,
        display: int = 1,
        settle_ms: float = 60.0,
        detect_blank: bool = True,
        verify_pointer: bool = True,
        native_resolution: bool = True,
        failsafe: bool = True,
        move_duration_s: float = 0.0,
        grabber: ScreenGrabber | None = None,
        pointer: Pointer | None = None,
    ) -> None:
        self._display = display
        self._settle_ms = max(settle_ms, 0.0)
        self._detect_blank = detect_blank
        self._verify_pointer = verify_pointer
        self._closed = False
        self._scale: float | None = None

        self._grabber: ScreenGrabber = grabber or MssGrabber(native_resolution=native_resolution)
        try:
            displays = self._grabber.displays()
            if not 0 <= display < len(displays):
                raise ControllerError(
                    f"display {display} does not exist; this machine reports "
                    f"{max(len(displays) - 1, 0)} display(s), numbered 1..{len(displays) - 1} "
                    f"with 0 meaning all of them"
                )
            self._bounds = displays[display]
            self._pointer: Pointer = pointer or PyAutoGuiPointer(
                failsafe=failsafe, move_duration_s=move_duration_s
            )
        except Exception:
            self._grabber.close()
            raise

    # -- Controller protocol -----------------------------------------------------------

    def capture(self) -> Screenshot:
        """Grab the display, reported at its LOGICAL size with the measured scale.

        Raises:
            ControllerError: if the controller is closed or the grab fails.
            PerceptionError: if the frame is a single flat colour, which means the
                Screen Recording permission is almost certainly missing. The message
                names the permission and where to grant it.
        """
        self._ensure_open()
        bounds = self._bounds
        raw = self._grabber.grab(bounds)
        if raw.width <= 0 or raw.height <= 0:
            raise ControllerError(f"screen capture returned an empty bitmap for {bounds}")

        scale_x = scaling.scale_for(raw.width, bounds.w)
        scale_y = scaling.scale_for(raw.height, bounds.h)
        if abs(scale_x - scale_y) > 0.01:
            raise ControllerError(
                f"capture is not uniformly scaled: {raw.width}x{raw.height} physical for "
                f"{bounds.w}x{bounds.h} logical is {scale_x:.3f}x horizontally but "
                f"{scale_y:.3f}x vertically; refusing to guess which one a click should use"
            )
        self._scale = scale_x

        png = self._encode_png(raw)
        return Screenshot(
            png=png,
            width=bounds.w,
            height=bounds.h,
            scale=scale_x,
            captured_at=utcnow(),
        )

    def perform(self, action: Action) -> ActionResult:
        """Execute one action, then wait ``settle_ms`` for the UI to catch up.

        Every way an individual action can fail - an unsupported kind, a point
        outside the display, a key this backend cannot press, the failsafe firing -
        comes back as ``ActionResult(ok=False, error=...)``. Nothing is raised except
        when the controller itself is unusable, exactly as the ``Controller``
        docstring requires, so an exploring agent can try the next thing.

        Raises:
            ControllerError: only if the controller is closed.
        """
        self._ensure_open()
        started = time.perf_counter()

        def done(ok: bool, error: str | None = None) -> ActionResult:
            return ActionResult(
                ok=ok, error=error, elapsed_ms=(time.perf_counter() - started) * 1000.0
            )

        if not self.supports(action.kind):
            return done(False, f"a desktop controller cannot perform {action.kind!r}")

        refusal = self._out_of_bounds(action)
        if refusal is not None:
            return done(False, refusal)

        try:
            error = self._dispatch(action)
        except Exception as exc:  # noqa: BLE001 - every backend failure is a failed action
            return done(False, f"{type(exc).__name__}: {exc}")
        if error is not None:
            return done(False, error)

        if self._settle_ms and not isinstance(action, Wait):
            time.sleep(self._settle_ms / 1000.0)
        return done(True)

    def viewport(self) -> Box:
        """The driven display's bounds in global LOGICAL pixels.

        ``x``/``y`` are its origin on the desktop - non-zero for a secondary display
        - while action coordinates remain relative to this box's top-left.
        """
        return self._bounds

    def supports(self, action_kind: ActionKind) -> bool:
        """Every action kind but the two that need a session history.

        A desktop has no address bar to ``navigate`` with and no history stack to go
        ``back`` through. Both are named rather than the set being "everything except
        navigate", so a controller that cannot do a thing says so instead of a new
        action kind arriving here as supported by default.
        """
        return action_kind not in _UNSUPPORTED_ACTIONS

    def url(self) -> str | None:
        """Always ``None``. A desktop has no notion of a URL."""
        return None

    def describe(self) -> str:
        scale = f"{self._scale:g}" if self._scale is not None else "?"
        b = self._bounds
        origin = "" if (b.x, b.y) == (0, 0) else f" at ({b.x},{b.y})"
        return f"desktop display {self._display} {b.w}x{b.h} @{scale}x{origin}"

    def close(self) -> None:
        """Release the capture handle. Idempotent; never raises."""
        if self._closed:
            return
        self._closed = True
        try:
            self._grabber.close()
        except Exception:  # noqa: BLE001 - close never raises
            pass

    # -- context manager ---------------------------------------------------------------

    def __enter__(self) -> DesktopController:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- extras beyond the protocol ----------------------------------------------------

    def displays(self) -> list[Box]:
        """Every display's bounds in global LOGICAL pixels, ``mss``-ordered.

        Index ``0`` is the union of them all; ``1`` is the primary. Use it to pick
        the ``display`` argument for another controller.

        Raises:
            ControllerError: if the controller is closed or enumeration fails.
        """
        self._ensure_open()
        return self._grabber.displays()

    @property
    def scale(self) -> float | None:
        """The backing scale factor measured by the most recent :meth:`capture`, or
        ``None`` before the first one. Never assumed, always measured."""
        return self._scale

    # -- internals ---------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise ControllerError("desktop controller is closed")

    def _local_bounds(self) -> Box:
        """The viewport with its origin moved to ``(0, 0)``: what an action's
        coordinates are measured against."""
        return Box(0, 0, self._bounds.w, self._bounds.h)

    def _out_of_bounds(self, action: Action) -> str | None:
        """The abort guard: refuse any action whose target is off the display.

        Half of a mis-scaled coordinate's damage is that it still lands *somewhere* -
        on another window, on a menu bar, on a Delete button. Refusing beats clicking
        a place nobody asked for, and the refusal is reported, not raised, so the
        agent learns from it.
        """
        bounds = self._local_bounds()
        for label, point in _targets(action):
            if not bounds.contains(point):
                return (
                    f"{label} ({point.x}, {point.y}) is outside the {bounds.w}x{bounds.h} "
                    f"display; refusing to act somewhere unintended"
                )
        return None

    def _dispatch(self, action: Action) -> str | None:
        """Run one supported, in-bounds action. Returns an error string or ``None``."""
        match action:
            case Click():
                target = scaling.to_global(action.point, self._bounds)
                self._pointer.click(
                    target.x, target.y, button=action.button, clicks=max(action.clicks, 1)
                )
                return self._check_landed(target)
            case Move():
                target = scaling.to_global(action.point, self._bounds)
                self._pointer.move_to(target.x, target.y)
                return self._check_landed(target)
            case Drag():
                start = scaling.to_global(action.start, self._bounds)
                end = scaling.to_global(action.end, self._bounds)
                self._pointer.drag(start.x, start.y, end.x, end.y)
                return self._check_landed(end)
            case TypeText():
                dropped = self._pointer.is_typable(action.text)
                if dropped:
                    return (
                        f"cannot type {dropped!r}: this backend types one character per "
                        f"key press and has no key for those, so the text would arrive "
                        f"silently incomplete"
                    )
                self._pointer.type_text(action.text)
                return None
            case PressKey():
                return self._press(action.keys)
            case Scroll():
                target = scaling.to_global(action.point, self._bounds)
                self._pointer.scroll(
                    target.x,
                    target.y,
                    horizontal=_wheel_clicks(-action.dx),
                    vertical=_wheel_clicks(-action.dy),
                )
                return None
            case Wait():
                time.sleep(max(action.ms, 0) / 1000.0)
                return None
            case Navigate():  # pragma: no cover - refused by supports() before here
                return "a desktop controller cannot navigate"
            case Back():  # pragma: no cover - refused by supports() before here
                return "a desktop controller has no session history to go back through"
        return f"unhandled action kind {action.kind!r}"  # pragma: no cover

    def _press(self, keys: tuple[str, ...]) -> str | None:
        """Hold ``keys`` together in order, then release them in reverse.

        Reverse release is what makes a chord a chord: Cmd goes down, A goes down, A
        comes up, Cmd comes up. Releasing in order would lift the modifier first and
        turn Cmd+A into a bare 'a'.
        """
        if not keys:
            return "no keys to press"
        translated: list[str] = []
        for key in keys:
            name = KEY_ALIASES.get(key.lower(), key.lower())
            if not self._pointer.knows_key(name):
                return f"unknown key name {key!r}"
            translated.append(name)
        pressed: list[str] = []
        try:
            for name in translated:
                self._pointer.key_down(name)
                pressed.append(name)
        finally:
            for name in reversed(pressed):
                self._pointer.key_up(name)
        return None

    def _check_landed(self, target: Point) -> str | None:
        """Confirm the cursor actually reached ``target``.

        On macOS a process without the Accessibility permission posts input events
        that are accepted and then ignored - no exception, no effect. Reading the
        position back is the only way to notice. A couple of pixels of slack absorbs
        pointer acceleration and a human nudging the mouse at the same moment.
        """
        if not self._verify_pointer:
            return None
        try:
            x, y = self._pointer.position()
        except Exception:  # noqa: BLE001 - an unreadable cursor is not a failed action
            return None
        if abs(x - target.x) <= 2 and abs(y - target.y) <= 2:
            return None
        return (
            f"pointer is at ({x}, {y}) but the action targeted ({target.x}, {target.y}): "
            f"{ACCESSIBILITY_HINT}"
        )

    def _encode_png(self, raw: RawCapture) -> bytes:
        """Encode the raw RGB bitmap as PNG, refusing a frame that is one flat colour.

        ``getextrema`` is a C-speed min/max per channel, so the blank check costs
        nothing next to the grab itself.
        """
        from PIL import Image

        try:
            image = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
        except (ValueError, OSError) as exc:
            raise ControllerError(f"captured bitmap could not be read: {exc}") from exc

        if self._detect_blank:
            extrema = image.getextrema()
            if all(low == high for low, high in extrema):
                raise PerceptionError(SCREEN_RECORDING_HINT)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def _targets(action: Action) -> tuple[tuple[str, Point], ...]:
    """Every logical point an action aims at, labelled for the error message."""
    if isinstance(action, Click | Move | Scroll):
        return (("point", action.point),)
    if isinstance(action, Drag):
        return (("drag start", action.start), ("drag end", action.end))
    return ()


def _wheel_clicks(pixels: int) -> int:
    """Logical pixels of scrolling as whole wheel clicks, never rounding to nothing.

    The backend truncates to an integer, so a small scroll asked for in pixels would
    otherwise turn into no scroll at all and look like the page refusing to move.
    Any non-zero request becomes at least one click in the right direction.
    """
    if pixels == 0:
        return 0
    magnitude = max(1, round(abs(pixels) / SCROLL_PIXELS_PER_CLICK))
    return magnitude if pixels > 0 else -magnitude
