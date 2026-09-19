"""Tests for ``DesktopController`` and the pure conversions in ``scaling``.

Nothing here touches a real screen, a real cursor, a model or the network by
default: the controller takes its two backends as arguments and gets doubles. The
one test that does touch the screen is skipped unless ``SKILLWEAVER_LIVE_DESKTOP=1``
is set, so the default run never raises a macOS permission prompt.

The scaling tests state every expected number as a literal worked out by hand. They
deliberately never call the function under test to compute what it should have
returned - a test that says ``assert f(x) == f(x)`` passes just as happily when
``f`` doubles everything, which is the exact bug this module exists to catch.
"""

from __future__ import annotations

import io
import os

import pytest

from skillweaver.contracts import (
    Box,
    Click,
    Controller,
    Drag,
    Move,
    Navigate,
    Point,
    PressKey,
    Scroll,
    TypeText,
    Wait,
)
from skillweaver.controllers import scaling
from skillweaver.controllers.desktop import (
    LIVE_ENV_VAR,
    DesktopController,
    RawCapture,
)
from skillweaver.errors import ControllerError, PerceptionError

# --------------------------------------------------------------------------------------
# scaling: logical <-> physical
# --------------------------------------------------------------------------------------


def test_logical_to_physical_at_scale_one_is_the_identity():
    assert scaling.logical_to_physical(0, 1.0) == 0
    assert scaling.logical_to_physical(37, 1.0) == 37
    assert scaling.logical_to_physical(1511, 1.0) == 1511
    assert scaling.logical_to_physical(-20, 1.0) == -20


def test_logical_to_physical_at_scale_two_doubles():
    # Worked by hand: 0*2=0, 37*2=74, 1511*2=3022, -20*2=-40.
    assert scaling.logical_to_physical(0, 2.0) == 0
    assert scaling.logical_to_physical(37, 2.0) == 74
    assert scaling.logical_to_physical(1511, 2.0) == 3022
    assert scaling.logical_to_physical(-20, 2.0) == -40


def test_physical_to_logical_at_scale_one_is_the_identity():
    assert scaling.physical_to_logical(0, 1.0) == 0
    assert scaling.physical_to_logical(37, 1.0) == 37
    assert scaling.physical_to_logical(3022, 1.0) == 3022
    assert scaling.physical_to_logical(-40, 1.0) == -40


def test_physical_to_logical_at_scale_two_halves():
    # Worked by hand: 0/2=0, 74/2=37, 3022/2=1511, -40/2=-20.
    assert scaling.physical_to_logical(0, 2.0) == 0
    assert scaling.physical_to_logical(74, 2.0) == 37
    assert scaling.physical_to_logical(3022, 2.0) == 1511
    assert scaling.physical_to_logical(-40, 2.0) == -20


def test_physical_to_logical_rounds_half_up_not_bankers():
    # 5/2 = 2.5 and 7/2 = 3.5. Python's round() would give 2 and 4; half-up gives 3 and 4.
    assert scaling.physical_to_logical(5, 2.0) == 3
    assert scaling.physical_to_logical(7, 2.0) == 4
    # -5/2 = -2.5 -> floor(-2.0) = -2.
    assert scaling.physical_to_logical(-5, 2.0) == -2


@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("value", [0, 1, 2, 7, 40, 399, 982, 1511, -1, -37])
def test_scalar_round_trip_is_exact_at_integer_scales(value: int, scale: float):
    physical = scaling.logical_to_physical(value, scale)
    assert scaling.physical_to_logical(physical, scale) == value


@pytest.mark.parametrize("scale", [0.0, -1.0, -2.0, float("nan"), float("inf")])
def test_a_nonsense_scale_is_rejected_rather_than_producing_nonsense(scale: float):
    with pytest.raises(ValueError, match="scale"):
        scaling.logical_to_physical(10, scale)
    with pytest.raises(ValueError, match="scale"):
        scaling.physical_to_logical(10, scale)


def test_scale_for_measures_the_ratio_of_a_real_capture():
    assert scaling.scale_for(3024, 1512) == 2.0
    assert scaling.scale_for(1512, 1512) == 1.0
    assert scaling.scale_for(1920, 1280) == 1.5


@pytest.mark.parametrize(("physical", "logical"), [(0, 100), (100, 0), (-10, 100), (100, -10)])
def test_scale_for_rejects_impossible_sizes(physical: int, logical: int):
    with pytest.raises(ValueError, match="positive"):
        scaling.scale_for(physical, logical)


# --------------------------------------------------------------------------------------
# scaling: points
# --------------------------------------------------------------------------------------


def test_point_conversions_at_scale_one():
    assert scaling.point_to_physical(Point(12, 34), 1.0) == (12, 34)
    assert scaling.point_from_physical(12, 34, 1.0) == Point(12, 34)


def test_point_conversions_at_scale_two():
    # By hand: (12, 34) logical is (24, 68) physical on a Retina display.
    assert scaling.point_to_physical(Point(12, 34), 2.0) == (24, 68)
    assert scaling.point_from_physical(24, 68, 2.0) == Point(12, 34)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_point_round_trip_is_exact_at_integer_scales(scale: float):
    for point in (Point(0, 0), Point(1, 3), Point(755, 491), Point(1511, 981), Point(-8, -9)):
        physical = scaling.point_to_physical(point, scale)
        assert scaling.point_from_physical(*physical, scale) == point


# --------------------------------------------------------------------------------------
# scaling: boxes
# --------------------------------------------------------------------------------------


def test_box_to_physical_at_scale_one_is_the_identity():
    assert scaling.box_to_physical(Box(3, 4, 5, 6), 1.0) == (3, 4, 5, 6)


def test_box_to_physical_at_scale_two_doubles_every_edge():
    # By hand: left 3*2=6, top 4*2=8, right (3+5)*2=16 so w=10, bottom (4+6)*2=20 so h=12.
    assert scaling.box_to_physical(Box(3, 4, 5, 6), 2.0) == (6, 8, 10, 12)


def test_box_from_physical_at_scale_two_halves_every_edge():
    # By hand: left floor(6/2)=3, top floor(8/2)=4, right ceil(16/2)=8 so w=5,
    # bottom ceil(20/2)=10 so h=6.
    assert scaling.box_from_physical(6, 8, 10, 12, 2.0) == Box(3, 4, 5, 6)


def test_box_from_physical_grows_outwards_to_keep_covering_the_detection():
    # Physical x 5..10 is logical 2.5..5.0. Flooring the near edge and ceiling the far
    # one gives 2..5, which still covers every physical pixel that was detected.
    assert scaling.box_from_physical(5, 5, 5, 5, 2.0) == Box(2, 2, 3, 3)


def test_box_scaling_keeps_neighbouring_edges_flush():
    # Two logical boxes that touch at x=7 must still touch physically: scaling the
    # width on its own would let rounding open a gap between them.
    left = scaling.box_to_physical(Box(3, 0, 4, 10), 1.5)
    right = scaling.box_to_physical(Box(7, 0, 4, 10), 1.5)
    assert left[0] + left[2] == right[0]


@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize(
    "box", [Box(0, 0, 1, 1), Box(3, 4, 5, 6), Box(0, 0, 1512, 982), Box(-40, -20, 100, 50)]
)
def test_box_round_trip_is_exact_at_integer_scales(box: Box, scale: float):
    physical = scaling.box_to_physical(box, scale)
    assert scaling.box_from_physical(*physical, scale) == box


def test_a_degenerate_box_stays_degenerate():
    assert scaling.box_to_physical(Box(10, 10, 0, 0), 2.0) == (20, 20, 0, 0)
    assert scaling.box_to_physical(Box(10, 10, -5, -5), 2.0) == (20, 20, 0, 0)
    assert scaling.box_from_physical(20, 20, 0, 0, 2.0) == Box(10, 10, 0, 0)


# --------------------------------------------------------------------------------------
# scaling: clamping
# --------------------------------------------------------------------------------------

SCREEN = Box(0, 0, 1512, 982)


def test_clamp_point_leaves_an_interior_point_alone():
    assert scaling.clamp_point(Point(10, 20), SCREEN) == Point(10, 20)
    assert scaling.clamp_point(Point(0, 0), SCREEN) == Point(0, 0)
    assert scaling.clamp_point(Point(1511, 981), SCREEN) == Point(1511, 981)


def test_clamp_point_at_the_left_edge():
    assert scaling.clamp_point(Point(-1, 500), SCREEN) == Point(0, 500)
    assert scaling.clamp_point(Point(-9999, 500), SCREEN) == Point(0, 500)


def test_clamp_point_at_the_right_edge_stops_one_short_of_the_width():
    # 1512 is off the screen: the rightmost real pixel is 1511.
    assert scaling.clamp_point(Point(1512, 500), SCREEN) == Point(1511, 500)
    assert scaling.clamp_point(Point(9999, 500), SCREEN) == Point(1511, 500)


def test_clamp_point_at_the_top_edge():
    assert scaling.clamp_point(Point(700, -1), SCREEN) == Point(700, 0)
    assert scaling.clamp_point(Point(700, -9999), SCREEN) == Point(700, 0)


def test_clamp_point_at_the_bottom_edge_stops_one_short_of_the_height():
    assert scaling.clamp_point(Point(700, 982), SCREEN) == Point(700, 981)
    assert scaling.clamp_point(Point(700, 9999), SCREEN) == Point(700, 981)


def test_clamp_point_at_a_corner_clamps_both_axes():
    assert scaling.clamp_point(Point(-5, -5), SCREEN) == Point(0, 0)
    assert scaling.clamp_point(Point(5000, 5000), SCREEN) == Point(1511, 981)


def test_clamp_point_respects_a_display_that_is_left_of_the_primary_one():
    # A second display hung off to the left has a negative origin.
    bounds = Box(-1920, -100, 1920, 1080)
    assert scaling.clamp_point(Point(-5000, -5000), bounds) == Point(-1920, -100)
    assert scaling.clamp_point(Point(5000, 5000), bounds) == Point(-1, 979)


def test_clamp_point_collapses_a_degenerate_bound_onto_its_origin():
    assert scaling.clamp_point(Point(50, 50), Box(10, 20, 0, 0)) == Point(10, 20)


def test_clamp_box_leaves_an_interior_box_alone():
    bounds = Box(0, 0, 100, 50)
    assert scaling.clamp_box(Box(10, 10, 20, 20), bounds) == Box(10, 10, 20, 20)
    assert scaling.clamp_box(Box(0, 0, 100, 50), bounds) == Box(0, 0, 100, 50)


def test_clamp_box_trims_each_edge():
    bounds = Box(0, 0, 100, 50)
    # Over the top-left: x -10..20 becomes 0..20, y -10..20 becomes 0..20.
    assert scaling.clamp_box(Box(-10, -10, 30, 30), bounds) == Box(0, 0, 20, 20)
    # Over the bottom-right: x 90..120 becomes 90..100, y 40..70 becomes 40..50.
    assert scaling.clamp_box(Box(90, 40, 30, 30), bounds) == Box(90, 40, 10, 10)


def test_clamp_box_may_end_on_the_far_edge_because_that_edge_is_exclusive():
    # Unlike a point, a box is allowed to end at x=100 on a 100-wide bound.
    assert scaling.clamp_box(Box(50, 0, 999, 10), Box(0, 0, 100, 50)) == Box(50, 0, 50, 10)


def test_clamp_box_entirely_outside_comes_back_with_zero_area():
    bounds = Box(0, 0, 100, 50)
    assert scaling.clamp_box(Box(200, 200, 10, 10), bounds).area == 0
    assert scaling.clamp_box(Box(-50, -50, 10, 10), bounds).area == 0


# --------------------------------------------------------------------------------------
# scaling: viewport origin
# --------------------------------------------------------------------------------------


def test_to_global_adds_the_display_origin():
    second = Box(1512, 0, 1920, 1080)
    assert scaling.to_global(Point(0, 0), second) == Point(1512, 0)
    assert scaling.to_global(Point(10, 20), second) == Point(1522, 20)


def test_to_local_subtracts_it_again():
    second = Box(1512, 0, 1920, 1080)
    assert scaling.to_local(Point(1522, 20), second) == Point(10, 20)
    for point in (Point(0, 0), Point(10, 20), Point(1919, 1079)):
        assert scaling.to_local(scaling.to_global(point, second), second) == point


def test_the_primary_display_needs_no_translation():
    assert scaling.to_global(Point(10, 20), SCREEN) == Point(10, 20)


# --------------------------------------------------------------------------------------
# doubles for the controller
# --------------------------------------------------------------------------------------

KNOWN_KEYS = frozenset(
    "abcdefghijklmnopqrstuvwxyz0123456789 .,/-=[];'\\`"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+{}|:\"<>?~"
) | frozenset(
    {
        "enter", "tab", "esc", "backspace", "delete", "space", "up", "down", "left",
        "right", "pageup", "pagedown", "home", "end", "insert", "ctrl", "command",
        "shift", "option", "capslock", "f1", "f5",
    }
)  # fmt: skip


class FakePointer:
    """Records what it was asked to do instead of moving the real cursor.

    ``ignore_moves`` reproduces the macOS Accessibility failure exactly: the calls
    are accepted, nothing raises, and the cursor never actually goes anywhere.
    """

    def __init__(self, *, start: tuple[int, int] = (0, 0), ignore_moves: bool = False) -> None:
        self.calls: list[tuple] = []
        self._position = start
        self._ignore_moves = ignore_moves

    def position(self) -> tuple[int, int]:
        return self._position

    def _land(self, x: int, y: int) -> None:
        if not self._ignore_moves:
            self._position = (x, y)

    def move_to(self, x: int, y: int) -> None:
        self.calls.append(("move_to", x, y))
        self._land(x, y)

    def click(self, x: int, y: int, *, button: str, clicks: int) -> None:
        self.calls.append(("click", x, y, button, clicks))
        self._land(x, y)

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        self.calls.append(("drag", start_x, start_y, end_x, end_y))
        self._land(end_x, end_y)

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", text))

    def key_down(self, key: str) -> None:
        self.calls.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", key))

    def scroll(self, x: int, y: int, *, horizontal: int, vertical: int) -> None:
        self.calls.append(("scroll", x, y, horizontal, vertical))

    def is_typable(self, text: str) -> str:
        return "".join(dict.fromkeys(c for c in text if c not in KNOWN_KEYS))

    def knows_key(self, key: str) -> bool:
        return key in KNOWN_KEYS


class FakeGrabber:
    """Hands back a synthetic bitmap of a chosen physical size.

    ``physical`` defaults to ``scale`` times the display's logical bounds, which is
    what a real Retina grab does. Setting it explicitly lets a test hand the
    controller a mismatched or blank frame.
    """

    def __init__(
        self,
        bounds: Box = Box(0, 0, 1512, 982),  # noqa: B008 - frozen value
        *,
        extra: tuple[Box, ...] = (),
        scale: float = 2.0,
        physical: tuple[int, int] | None = None,
        uniform: bool = False,
    ) -> None:
        self._bounds = bounds
        self._extra = extra
        self._scale = scale
        self._physical = physical
        self._uniform = uniform
        self.grabs: list[Box] = []
        self.closed = 0

    def displays(self) -> list[Box]:
        union = self._bounds if not self._extra else _union((self._bounds, *self._extra))
        return [union, self._bounds, *self._extra]

    def grab(self, bounds: Box) -> RawCapture:
        self.grabs.append(bounds)
        width, height = self._physical or (
            round(bounds.w * self._scale),
            round(bounds.h * self._scale),
        )
        if self._uniform:
            rgb = bytes([17, 17, 17]) * (width * height)
        else:
            # A gradient: every row differs, so a blank-frame check must not fire.
            rgb = bytes(
                value
                for y in range(height)
                for x in range(width)
                for value in (x % 256, y % 256, (x + y) % 256)
            )
        return RawCapture(rgb=rgb, width=width, height=height)

    def close(self) -> None:
        self.closed += 1


def _union(boxes: tuple[Box, ...]) -> Box:
    left = min(b.x for b in boxes)
    top = min(b.y for b in boxes)
    right = max(b.x + b.w for b in boxes)
    bottom = max(b.y + b.h for b in boxes)
    return Box(left, top, right - left, bottom - top)


def make_controller(
    grabber: FakeGrabber | None = None,
    pointer: FakePointer | None = None,
    **kwargs,
) -> tuple[DesktopController, FakeGrabber, FakePointer]:
    """A controller wired to doubles, with the settle wait off so tests do not sleep."""
    grabber = grabber or FakeGrabber(Box(0, 0, 200, 100), scale=2.0)
    pointer = pointer or FakePointer()
    kwargs.setdefault("settle_ms", 0.0)
    return DesktopController(grabber=grabber, pointer=pointer, **kwargs), grabber, pointer


# --------------------------------------------------------------------------------------
# the controller: shape and support
# --------------------------------------------------------------------------------------


def test_it_satisfies_the_controller_protocol():
    ctl, _, _ = make_controller()
    assert isinstance(ctl, Controller)


def test_supports_every_action_kind_except_navigate():
    ctl, _, _ = make_controller()
    for kind in ("click", "move", "drag", "type_text", "press_key", "scroll", "wait"):
        assert ctl.supports(kind) is True, kind
    assert ctl.supports("navigate") is False


def test_navigate_is_refused_rather_than_raised_and_touches_nothing():
    ctl, _, pointer = make_controller()
    result = ctl.perform(Navigate("https://example.test/"))
    assert result.ok is False
    assert result.error is not None and "navigate" in result.error
    assert pointer.calls == []


def test_a_desktop_has_no_url():
    ctl, _, _ = make_controller()
    assert ctl.url() is None


def test_describe_says_it_does_not_know_the_scale_until_it_has_measured_one():
    ctl, _, _ = make_controller()
    assert ctl.describe() == "desktop display 1 200x100 @?x"
    ctl.capture()
    assert ctl.describe() == "desktop display 1 200x100 @2x"


# --------------------------------------------------------------------------------------
# the controller: the abort guard
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        Click(Point(200, 10)),  # x == width: one past the last real pixel
        Click(Point(10, 100)),  # y == height
        Click(Point(-1, 10)),
        Click(Point(10, -1)),
        Click(Point(5000, 5000)),
        Move(Point(200, 100)),
        Scroll(Point(-3, 10), dy=100),
        Drag(Point(10, 10), Point(500, 10)),
        Drag(Point(-1, 10), Point(20, 20)),
    ],
)
def test_an_out_of_bounds_action_is_refused_without_moving_the_cursor(action):
    ctl, _, pointer = make_controller()
    result = ctl.perform(action)
    assert result.ok is False
    assert result.error is not None and "outside" in result.error
    assert pointer.calls == []
    assert pointer.position() == (0, 0)


def test_the_last_real_pixel_is_still_in_bounds():
    ctl, _, pointer = make_controller()
    assert ctl.perform(Click(Point(199, 99))).ok is True
    assert pointer.calls == [("click", 199, 99, "left", 1)]


# --------------------------------------------------------------------------------------
# the controller: actions
# --------------------------------------------------------------------------------------


def test_a_click_is_delivered_at_the_requested_logical_point():
    ctl, _, pointer = make_controller()
    assert ctl.perform(Click(Point(40, 12), button="right", clicks=2)).ok is True
    assert pointer.calls == [("click", 40, 12, "right", 2)]


def test_a_click_on_a_second_display_is_offset_by_that_displays_origin():
    # The action says (10, 20); the cursor must go to (1522, 20) because the display
    # starts at x=1512. Getting this wrong clicks on the wrong screen entirely.
    grabber = FakeGrabber(Box(0, 0, 1512, 982), extra=(Box(1512, 0, 1920, 1080),))
    ctl, _, pointer = make_controller(grabber, display=2)
    assert ctl.viewport() == Box(1512, 0, 1920, 1080)
    assert ctl.perform(Click(Point(10, 20))).ok is True
    assert pointer.calls == [("click", 1522, 20, "left", 1)]


def test_a_move_reports_the_point_it_was_given():
    ctl, _, pointer = make_controller()
    assert ctl.perform(Move(Point(7, 8))).ok is True
    assert pointer.calls == [("move_to", 7, 8)]


def test_a_drag_carries_both_endpoints():
    ctl, _, pointer = make_controller()
    assert ctl.perform(Drag(Point(5, 6), Point(50, 60))).ok is True
    assert pointer.calls == [("drag", 5, 6, 50, 60)]


def test_typing_passes_the_text_through_verbatim():
    ctl, _, pointer = make_controller()
    assert ctl.perform(TypeText("Acme Corp.")).ok is True
    assert pointer.calls == [("type_text", "Acme Corp.")]


def test_typing_a_character_the_backend_would_drop_fails_loudly_instead():
    # pyautogui types one key per character and silently skips what it has no key
    # for. Half-typed text that reports success is worse than a refusal.
    ctl, _, pointer = make_controller()
    result = ctl.perform(TypeText("café – 日本"))
    assert result.ok is False
    assert result.error is not None and "cannot type" in result.error
    assert pointer.calls == []


def test_a_key_chord_presses_in_order_and_releases_in_reverse():
    # Cmd down, A down, A up, Cmd up. Releasing in order would lift the modifier
    # first and turn Cmd+A into a bare "a".
    ctl, _, pointer = make_controller()
    assert ctl.perform(PressKey(("Meta", "a"))).ok is True
    assert pointer.calls == [
        ("key_down", "command"),
        ("key_down", "a"),
        ("key_up", "a"),
        ("key_up", "command"),
    ]


@pytest.mark.parametrize(
    ("contract_name", "backend_name"),
    [
        ("Enter", "enter"),
        ("Escape", "esc"),
        ("ArrowDown", "down"),
        ("ArrowUp", "up"),
        ("Control", "ctrl"),
        ("Meta", "command"),
        ("Alt", "option"),
        ("Backspace", "backspace"),
        ("Tab", "tab"),
        ("F5", "f5"),
        ("a", "a"),
    ],
)
def test_contract_key_names_are_translated_to_backend_ones(contract_name, backend_name):
    ctl, _, pointer = make_controller()
    assert ctl.perform(PressKey((contract_name,))).ok is True
    assert pointer.calls == [("key_down", backend_name), ("key_up", backend_name)]


def test_an_unknown_key_is_refused_and_nothing_is_pressed():
    ctl, _, pointer = make_controller()
    result = ctl.perform(PressKey(("Enter", "Nonexistent")))
    assert result.ok is False
    assert result.error is not None and "Nonexistent" in result.error
    assert pointer.calls == []


def test_an_empty_chord_is_refused():
    ctl, _, _ = make_controller()
    assert ctl.perform(PressKey(())).ok is False


def test_scrolling_down_sends_the_wheel_the_other_way():
    # Scroll's positive dy means "reveal content further down"; the wheel event's
    # positive vertical means "scroll the view up". They are opposites.
    ctl, _, pointer = make_controller()
    assert ctl.perform(Scroll(Point(10, 10), dy=120)).ok is True
    assert pointer.calls == [("scroll", 10, 10, 0, -3)]


def test_scrolling_right_sends_the_wheel_the_other_way():
    ctl, _, pointer = make_controller()
    assert ctl.perform(Scroll(Point(10, 10), dx=80)).ok is True
    assert pointer.calls == [("scroll", 10, 10, -2, 0)]


def test_a_small_scroll_still_moves_by_one_click_instead_of_none():
    # 10px is a quarter of a click and would truncate to zero: a scroll that
    # silently does nothing looks like a page that refuses to move.
    ctl, _, pointer = make_controller()
    assert ctl.perform(Scroll(Point(10, 10), dy=10)).ok is True
    assert pointer.calls == [("scroll", 10, 10, 0, -1)]


def test_a_zero_scroll_does_nothing_at_all():
    ctl, _, pointer = make_controller()
    assert ctl.perform(Scroll(Point(10, 10))).ok is True
    assert pointer.calls == [("scroll", 10, 10, 0, 0)]


def test_wait_takes_at_least_as_long_as_it_was_asked_for():
    ctl, _, pointer = make_controller()
    result = ctl.perform(Wait(30))
    assert result.ok is True
    assert result.elapsed_ms >= 30.0
    assert pointer.calls == []


def test_a_backend_that_raises_becomes_a_failed_action_not_an_exception():
    class Exploding(FakePointer):
        def click(self, x, y, *, button, clicks):
            raise RuntimeError("the screen fell off")

    ctl, _, _ = make_controller(pointer=Exploding())
    result = ctl.perform(Click(Point(1, 1)))
    assert result.ok is False
    assert result.error is not None and "the screen fell off" in result.error


# --------------------------------------------------------------------------------------
# the controller: the silent-permission failures
# --------------------------------------------------------------------------------------


def test_a_cursor_that_never_moves_is_reported_as_a_missing_accessibility_grant():
    # macOS accepts the event and ignores it. Reading the position back is the only
    # way to notice, so this is the check that turns a silent no-op into an error.
    ctl, _, pointer = make_controller(pointer=FakePointer(ignore_moves=True))
    result = ctl.perform(Click(Point(40, 12)))
    assert result.ok is False
    assert result.error is not None and "Accessibility" in result.error


def test_the_pointer_check_can_be_turned_off():
    ctl, _, _ = make_controller(pointer=FakePointer(ignore_moves=True), verify_pointer=False)
    assert ctl.perform(Click(Point(40, 12))).ok is True


def test_a_blank_capture_names_the_screen_recording_permission():
    grabber = FakeGrabber(Box(0, 0, 200, 100), uniform=True)
    ctl, _, _ = make_controller(grabber)
    with pytest.raises(PerceptionError, match="Screen Recording"):
        ctl.capture()


def test_the_blank_check_can_be_turned_off_for_a_screen_that_really_is_one_colour():
    grabber = FakeGrabber(Box(0, 0, 200, 100), uniform=True)
    ctl, _, _ = make_controller(grabber, detect_blank=False)
    assert ctl.capture().width == 200


# --------------------------------------------------------------------------------------
# the controller: capture
# --------------------------------------------------------------------------------------


def test_capture_reports_logical_size_and_the_measured_scale():
    from PIL import Image

    grabber = FakeGrabber(Box(0, 0, 200, 100), scale=2.0)
    ctl, _, _ = make_controller(grabber)
    shot = ctl.capture()

    # Reported logical, stored physical: this is the whole point of the module.
    assert (shot.width, shot.height) == (200, 100)
    assert shot.scale == 2.0
    assert Image.open(io.BytesIO(shot.png)).size == (400, 200)


def test_capture_measures_a_one_to_one_display_as_one_to_one():
    grabber = FakeGrabber(Box(0, 0, 200, 100), scale=1.0)
    ctl, _, _ = make_controller(grabber)
    shot = ctl.capture()
    assert shot.scale == 1.0
    assert (shot.width, shot.height) == (200, 100)


def test_to_array_comes_back_in_logical_pixels_so_boxes_need_no_conversion():
    ctl, _, _ = make_controller(FakeGrabber(Box(0, 0, 200, 100), scale=2.0))
    shot = ctl.capture()
    assert shot.to_array().shape == (100, 200, 3)
    assert shot.to_array(logical=False).shape == (200, 400, 3)


def test_capture_grabs_the_chosen_displays_global_bounds():
    grabber = FakeGrabber(Box(0, 0, 1512, 982), extra=(Box(-1920, -100, 1920, 1080),))
    ctl, _, _ = make_controller(grabber, display=2)
    ctl.capture()
    assert grabber.grabs == [Box(-1920, -100, 1920, 1080)]


def test_a_capture_scaled_differently_on_each_axis_is_refused_rather_than_guessed():
    grabber = FakeGrabber(Box(0, 0, 200, 100), physical=(400, 100))
    ctl, _, _ = make_controller(grabber)
    with pytest.raises(ControllerError, match="not uniformly scaled"):
        ctl.capture()


def test_the_measured_scale_is_exposed_only_once_it_has_been_measured():
    ctl, _, _ = make_controller()
    assert ctl.scale is None
    ctl.capture()
    assert ctl.scale == 2.0


# --------------------------------------------------------------------------------------
# the controller: displays and lifecycle
# --------------------------------------------------------------------------------------


def test_displays_are_listed_mss_style_with_the_union_first():
    grabber = FakeGrabber(Box(0, 0, 1512, 982), extra=(Box(1512, 0, 1920, 1080),))
    ctl, _, _ = make_controller(grabber)
    assert ctl.displays() == [
        Box(0, 0, 3432, 1080),
        Box(0, 0, 1512, 982),
        Box(1512, 0, 1920, 1080),
    ]


def test_asking_for_a_display_that_does_not_exist_fails_at_construction():
    grabber = FakeGrabber(Box(0, 0, 200, 100))
    with pytest.raises(ControllerError, match="display 7 does not exist"):
        DesktopController(display=7, grabber=grabber, pointer=FakePointer())
    assert grabber.closed == 1, "the grabber must not be left holding the screen"


def test_the_context_manager_releases_the_screen():
    grabber = FakeGrabber(Box(0, 0, 200, 100))
    with DesktopController(grabber=grabber, pointer=FakePointer()) as ctl:
        assert ctl.viewport() == Box(0, 0, 200, 100)
    assert grabber.closed == 1


def test_close_is_idempotent_and_never_raises():
    ctl, grabber, _ = make_controller()
    ctl.close()
    ctl.close()
    assert grabber.closed == 1


def test_using_a_closed_controller_raises_rather_than_reporting_a_failed_action():
    ctl, _, _ = make_controller()
    ctl.close()
    with pytest.raises(ControllerError, match="closed"):
        ctl.capture()
    with pytest.raises(ControllerError, match="closed"):
        ctl.perform(Click(Point(1, 1)))


# --------------------------------------------------------------------------------------
# live: the real screen
# --------------------------------------------------------------------------------------

live_only = pytest.mark.skipif(
    os.environ.get(LIVE_ENV_VAR) != "1",
    reason=f"set {LIVE_ENV_VAR}=1 to capture this machine's real screen",
)


@live_only
def test_live_capture_of_the_real_screen():
    """Capture this machine for real and check the numbers hang together.

    Capture only: nothing here moves the cursor or presses a key, so running it
    needs the Screen Recording permission but not Accessibility.
    """
    from PIL import Image

    with DesktopController() as ctl:
        bounds = ctl.viewport()
        assert bounds.w >= 640 and bounds.h >= 400, f"implausible display bounds {bounds}"

        shot = ctl.capture()

        # Logical dimensions, matching what the pointer API addresses.
        assert (shot.width, shot.height) == (bounds.w, bounds.h)

        # A real backing factor, and the PNG really is that many times bigger.
        assert 1.0 <= shot.scale <= 3.0, f"implausible scale {shot.scale}"
        assert Image.open(io.BytesIO(shot.png)).size == (
            round(bounds.w * shot.scale),
            round(bounds.h * shot.scale),
        )

        # Real content, not a permission-denied flat rectangle.
        array = shot.to_array()
        assert array.shape == (bounds.h, bounds.w, 3)
        assert array.std() > 1.0, "the screen looks uniform; is Screen Recording granted?"
        assert len({tuple(px) for px in array.reshape(-1, 3)[::997]}) > 10

        print(f"\nlive capture: {ctl.describe()} -> png {len(shot.png)} bytes")
