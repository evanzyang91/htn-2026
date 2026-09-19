"""``FakeController``: a scriptable state machine standing in for a browser or desktop.

A fake application is a set of named states plus a transition table::

    states = {
        "home":  FakeState(png_home, [login_button], "https://fake.test/"),
        "login": (png_login, [user_field], "https://fake.test/login"),   # a tuple works too
    }
    transitions = {
        "home": [(clicks(login_button), "login")],     # first matching rule wins
        "login": [(presses("Escape"), "home")],
    }
    ctl = FakeController(states, transitions, start="home")
    ctl.perform(Click(login_button.box.center))
    assert ctl.state == "login" and ctl.actions == [Click(login_button.box.center)]

An action matching no rule succeeds and changes nothing, like clicking blank space.
Nothing here sleeps, draws or touches a real screen.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

from PIL import Image, ImageDraw

from skillweaver.contracts import (
    Action,
    ActionKind,
    ActionResult,
    Box,
    Click,
    Drag,
    Element,
    ElementSource,
    Move,
    Navigate,
    Point,
    PressKey,
    Screenshot,
    Scroll,
    TypeText,
    utcnow,
)
from skillweaver.errors import ControllerError

ActionMatcher = Callable[[Action], bool]
"""A predicate over actions; a transition fires when its matcher returns true."""


@dataclass(frozen=True, slots=True)
class FakeState:
    """One screen of a fake application.

    ``png`` must be unique per state (the fake detector and reader look states up by
    it) and should be a real PNG so ``Screenshot.to_array()`` works - use
    :func:`render_png`. ``elements`` are what perception will "see" on this screen,
    boxes in LOGICAL pixels. ``url`` is what ``Controller.url()`` reports.
    """

    png: bytes
    elements: tuple[Element, ...]
    url: str | None = None


StateLike = FakeState | tuple[bytes, Sequence[Element], str | None]


def render_png(
    elements: Sequence[Element],
    width: int = 800,
    height: int = 600,
    scale: float = 1.0,
    background: tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    """Draw ``elements`` as labelled outlines and return real PNG bytes.

    ``width``/``height`` are LOGICAL pixels; the image is ``scale`` times larger, as
    a real capture would be. Deterministic for the same inputs.
    """
    image = Image.new("RGB", (round(width * scale), round(height * scale)), background)
    draw = ImageDraw.Draw(image)
    for el in elements:
        b = el.box
        x0, y0 = b.x * scale, b.y * scale
        x1, y1 = (b.x + b.w) * scale - 1, (b.y + b.h) * scale - 1
        draw.rectangle((x0, y0, x1, y1), outline=(40, 40, 40))
        if el.text:
            draw.text((x0 + 4, y0 + 4), el.text, fill=(0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# -- action matchers -------------------------------------------------------------------


def clicks(target: Element | Box, button: str = "left") -> ActionMatcher:
    """Matches a :class:`Click` with ``button`` landing inside ``target``'s box."""
    box = target.box if isinstance(target, Element) else target
    return lambda a: isinstance(a, Click) and a.button == button and box.contains(a.point)


def types(text: str | None = None, *, contains: str | None = None) -> ActionMatcher:
    """Matches a :class:`TypeText`: exactly ``text``, or containing ``contains``
    (case-insensitive), or any typing when both are ``None``."""

    def match(a: Action) -> bool:
        if not isinstance(a, TypeText):
            return False
        if text is not None:
            return a.text == text
        if contains is not None:
            return contains.lower() in a.text.lower()
        return True

    return match


def presses(*keys: str) -> ActionMatcher:
    """Matches a :class:`PressKey` with exactly this chord."""
    return lambda a: isinstance(a, PressKey) and a.keys == tuple(keys)


def navigates(url: str) -> ActionMatcher:
    """Matches a :class:`Navigate` to exactly ``url``."""
    return lambda a: isinstance(a, Navigate) and a.url == url


def kind_is(kind: ActionKind) -> ActionMatcher:
    """Matches any action of the given kind."""
    return lambda a: a.kind == kind


def any_action() -> ActionMatcher:
    """Matches every action."""
    return lambda a: True


# -- the controller --------------------------------------------------------------------


class FakeController:
    """A ``contracts.Controller`` driven by a state table instead of a screen.

    Args:
        states: State name to :class:`FakeState` or a ``(png, elements, url)`` tuple.
        transitions: State name to an ordered list of ``(matcher, next_state)``
            rules. The first rule whose matcher accepts the action fires. States
            absent from the table have no way out.
        start: Name of the initial state (and the state :meth:`reset` returns to).
        viewport: Logical size reported by :meth:`viewport` and used for captures.
        scale: ``Screenshot.scale`` to report (render the PNGs at the same scale).
        unsupported: Action kinds :meth:`supports` denies; performing one returns
            ``ActionResult(ok=False)`` and changes nothing.
        action_ms: The ``elapsed_ms`` reported for every action (no real waiting).

    Inspection, for assertions:
        ``state`` (current state name), ``actions`` (every action passed to
        ``perform``, in order, including failed ones), ``history`` (``(state_before,
        action, state_after)`` triples), ``captures`` (number of ``capture`` calls),
        ``closed``.

    Raises:
        ValueError: on an unknown state name or two states sharing the same PNG.
    """

    def __init__(
        self,
        states: Mapping[str, StateLike],
        transitions: Mapping[str, Sequence[tuple[ActionMatcher, str]]],
        start: str,
        *,
        viewport: Box = Box(0, 0, 800, 600),  # noqa: B008 - frozen value
        scale: float = 1.0,
        unsupported: Collection[ActionKind] = (),
        action_ms: float = 5.0,
    ) -> None:
        self.states: dict[str, FakeState] = {
            name: s if isinstance(s, FakeState) else FakeState(s[0], tuple(s[1]), s[2])
            for name, s in states.items()
        }
        self.transitions = {name: list(rules) for name, rules in transitions.items()}
        if start not in self.states:
            raise ValueError(f"start state {start!r} is not in states")
        for name, rules in self.transitions.items():
            for target in (name, *(target for _, target in rules)):
                if target not in self.states:
                    raise ValueError(f"transitions for {name!r} name unknown state {target!r}")
        pngs = [s.png for s in self.states.values()]
        if len(set(pngs)) != len(pngs):
            raise ValueError("every state needs a unique png (fakes look states up by it)")

        self._start = start
        self._viewport = viewport
        self._scale = scale
        self._unsupported = frozenset(unsupported)
        self._action_ms = action_ms
        self._fail_next: str | None = None
        self.state: str = start
        self.actions: list[Action] = []
        self.history: list[tuple[str, Action, str]] = []
        self.captures: int = 0
        self.closed: bool = False

    # -- scripting helpers (not part of the Controller protocol) -----------------------

    @property
    def current(self) -> FakeState:
        """The :class:`FakeState` currently on screen."""
        return self.states[self.state]

    def state_for_png(self, png: bytes) -> FakeState | None:
        """The state whose screenshot is ``png``, or ``None``."""
        return next((s for s in self.states.values() if s.png == png), None)

    def reset(self) -> None:
        """Return to the start state, as reloading the app would. Recorded actions
        and history are kept."""
        self.state = self._start

    def fail_next(self, error: str = "injected failure") -> None:
        """Make the next ``perform`` return ``ActionResult(ok=False, error=error)``
        without changing state."""
        self._fail_next = error

    # -- Controller protocol -----------------------------------------------------------

    def capture(self) -> Screenshot:
        self._ensure_open()
        self.captures += 1
        return Screenshot(
            png=self.current.png,
            width=self._viewport.w,
            height=self._viewport.h,
            scale=self._scale,
            captured_at=utcnow(),
        )

    def perform(self, action: Action) -> ActionResult:
        self._ensure_open()
        self.actions.append(action)
        before = self.state
        error = self._refusal(action)
        if error is None:
            for matcher, target in self.transitions.get(self.state, ()):
                if matcher(action):
                    self.state = target
                    break
        self.history.append((before, action, self.state))
        return ActionResult(ok=error is None, error=error, elapsed_ms=self._action_ms)

    def viewport(self) -> Box:
        return self._viewport

    def supports(self, action_kind: ActionKind) -> bool:
        return action_kind not in self._unsupported

    def url(self) -> str | None:
        return self.current.url

    def describe(self) -> str:
        v = self._viewport
        return f"fake controller {v.w}x{v.h} @{self._scale:g}x state={self.state}"

    def close(self) -> None:
        self.closed = True

    # -- internals ---------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self.closed:
            raise ControllerError("fake controller is closed")

    def _refusal(self, action: Action) -> str | None:
        if self._fail_next is not None:
            error, self._fail_next = self._fail_next, None
            return error
        if action.kind in self._unsupported:
            return f"action kind {action.kind!r} is not supported"
        bounds = Box(0, 0, self._viewport.w, self._viewport.h)
        for point in _points(action):
            if not bounds.contains(point):
                return f"point ({point.x}, {point.y}) is outside the viewport"
        return None


def _points(action: Action) -> tuple[Point, ...]:
    if isinstance(action, Click | Move | Scroll):
        return (action.point,)
    if isinstance(action, Drag):
        return (action.start, action.end)
    return ()


class FakeGroundTruth:
    """A ``contracts.GroundTruthSource`` over a :class:`FakeController`: the current
    state's declared elements, re-labelled ``source=dom`` with confidence ``1.0``.

    Offline teacher only - hand it to evaluation and labelling code, never to
    anything on the agent's action path.
    """

    def __init__(self, controller: FakeController) -> None:
        self._controller = controller
        self.calls = 0

    def elements(self) -> list[Element]:
        self.calls += 1
        return [
            Element(e.box, e.kind, e.text, 1.0, e.stable_id, ElementSource.dom)
            for e in self._controller.current.elements
        ]

    def url(self) -> str:
        self.calls += 1
        return self._controller.current.url or ""
