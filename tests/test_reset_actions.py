"""An undo performed on the screen, and the three promises it has to keep.

The admission gate re-runs a candidate up to three times and resets between attempts,
so an undo that works four times in five does not make a flaky demo - it makes the
gate REJECT the skill, because the fourth attempt starts dirty and disagrees with the
first three. The happy path is therefore the least interesting thing here. What is
tested is idempotence on an already-clean world, convergence over a cart with several
lines, and a bound that reports ``failed`` rather than claiming success.

The world is :class:`CartWorld` below: a fake shop whose cart really does shrink one
line per click, so "remove every line" is a loop against state and not a script.
"""

from __future__ import annotations

from typing import Any

import pytest

from skillweaver.contracts import (
    Box,
    Click,
    Element,
    ElementKind,
    ElementSource,
    Point,
    PressKey,
)
from skillweaver.errors import PerceptionError
from skillweaver.orchestrator import ResetOutcome, reset_world
from skillweaver.reset_actions import (
    DEFAULT_MAX_ROUNDS,
    chain_resets,
    reset_step_to_dict,
    reset_steps_from,
    world_reset_from_actions,
)
from tests.fakes.controller import FakeController, FakeState, render_png
from tests.fakes.perception import SimpleElementIndex

# --------------------------------------------------------------------------------------
# A cart that really empties
# --------------------------------------------------------------------------------------

LINES = ("Pad Thai", "Green Curry", "Mango Sticky Rice")
EMPTY_MARKER = "Your cart is empty"
DIRTY_MARKER = "each"

# Where the unlabelled trash icon sits relative to the price text beside it. The
# offset is the whole point of dx/dy: it travels with the line when the page moves.
TRASH_DX, TRASH_DY = 600, 0


def _el(kind: ElementKind, text: str, box: Box, *, label: str = "") -> Element:
    """One element. ``label`` is the name only the DOM knows - an ``aria-label``."""
    return Element(box, kind, label or text, 1.0, None, ElementSource.merged)


class CartWorld:
    """A fake shop with a cart that shrinks by one line per click on a trash icon.

    Two screens the reset has to tell apart, which is the whole reason ``until_seen``
    exists: ``shop`` is the catalogue, which says neither marker, and ``cart`` is the
    cart, which says ``"$N each"`` per line or ``"Your cart is empty"``.

    ``painted`` is what a camera sees and ``labelled`` is what the DOM reports. They
    differ on one element ON PURPOSE: the trash control is icon-only and its name lives
    in an ``aria-label``, exactly as DoorDash's remove and quick-add controls do, so a
    test that aims at it ``via="dom"`` is exercising the real reason that option exists.

    ``top_offset`` pushes every cart row down, standing in for the notice that arrives
    at the top of a real page.
    """

    def __init__(self, lines: int = len(LINES), *, top_offset: int = 0) -> None:
        self.lines = lines
        self.top_offset = top_offset
        self.screen = "shop"
        self.clicks: list[Point] = []

    # -- the two screens ---------------------------------------------------------------

    def painted(self) -> tuple[Element, ...]:
        """What detection and OCR see: no name on the trash icon."""
        return self._elements(labelled=False)

    def labelled(self) -> tuple[Element, ...]:
        """What the DOM reports: the trash icon carries ``"Remove <dish>"``."""
        return self._elements(labelled=True)

    def _elements(self, *, labelled: bool) -> tuple[Element, ...]:
        out = [
            _el(ElementKind.button, "Shop", Box(10, 10, 80, 30)),
            _el(ElementKind.button, "Cart", Box(120, 10, 80, 30)),
        ]
        if self.screen == "shop":
            out.append(_el(ElementKind.text, "Today's menu", Box(10, 60, 300, 20)))
            return tuple(out)
        if self.lines == 0:
            out.append(_el(ElementKind.text, EMPTY_MARKER, Box(10, 60 + self.top_offset, 300, 20)))
            return tuple(out)
        for row in range(self.lines):
            y = 60 + self.top_offset + row * 80
            price = Box(10, y + 24, 200, 20)
            out.append(_el(ElementKind.text, LINES[row], Box(10, y, 200, 20)))
            out.append(_el(ElementKind.text, f"$12.50 {DIRTY_MARKER}", price))
            # The trash icon is placed FROM the price's centre, which is what makes
            # (TRASH_DX, TRASH_DY) the honest offset a reset step would be written with.
            icon = price.center
            out.append(
                _el(
                    ElementKind.icon,
                    "",
                    Box(icon.x + TRASH_DX - 10, icon.y + TRASH_DY - 10, 20, 20),
                    label=f"Remove {LINES[row]}" if labelled else "",
                )
            )
        return tuple(out)

    # -- the controller it drives -------------------------------------------------------

    def act(self, action: Any) -> None:
        """Apply one action: the Cart button switches screen, a trash icon removes."""
        if isinstance(action, Click):
            self.clicks.append(action.point)
            for element in self.painted():
                if not element.box.contains(action.point):
                    continue
                if element.text == "Cart":
                    self.screen = "cart"
                    return
                if element.text == "Shop":
                    self.screen = "shop"
                    return
            if self.screen == "cart" and self.lines and self._on_a_trash_icon(action.point):
                self.lines -= 1

    def _on_a_trash_icon(self, point: Point) -> bool:
        return any(
            element.box.contains(point)
            for element in self.labelled()
            if element.kind is ElementKind.icon
        )


class CartController:
    """A ``contracts.Controller`` over a :class:`CartWorld`. Nothing is scripted."""

    def __init__(self, world: CartWorld, *, refuse: str | None = None) -> None:
        self.world = world
        self.refuse = refuse
        self.performed: list[Any] = []

    def capture(self) -> Any:
        return FakeController(
            {"only": FakeState(render_png([]), (), "https://shop.test/")}, {}, "only"
        ).capture()

    def perform(self, action: Any) -> Any:
        from skillweaver.contracts import ActionResult

        self.performed.append(action)
        if self.refuse is not None:
            return ActionResult(ok=False, error=self.refuse)
        self.world.act(action)
        return ActionResult(ok=True)

    def viewport(self) -> Box:
        return Box(0, 0, 800, 600)

    def supports(self, action_kind: Any) -> bool:
        return True

    def url(self) -> str | None:
        return "https://shop.test/"

    def describe(self) -> str:
        return "cart world"

    def close(self) -> None:
        return None


class PaintedEyes:
    """A ``contracts.Perceiver`` returning what a camera would see. Counts calls."""

    def __init__(self, world: CartWorld, *, blind: bool = False) -> None:
        self.world = world
        self.blind = blind
        self.calls = 0

    def observe(self, controller: Any) -> Any:
        from skillweaver.contracts import Fingerprint, Observation, utcnow

        self.calls += 1
        if self.blind:
            raise PerceptionError("the eyes are broken")
        elements = self.world.painted()
        return Observation(
            screenshot=controller.capture(),
            elements=elements,
            index=SimpleElementIndex(elements),
            fingerprint=Fingerprint("cart"),
            url=controller.url(),
            taken_at=utcnow(),
        )


class LabelledDom:
    """A ``contracts.GroundTruthSource`` over the same world. Counts calls."""

    def __init__(self, world: CartWorld) -> None:
        self.world = world
        self.calls = 0

    def elements(self) -> list[Element]:
        self.calls += 1
        return list(self.world.labelled())

    def url(self) -> str:
        return "https://shop.test/"


def empty_the_cart(
    *, via: str = "dom", positive: bool = True, max_rounds: int = DEFAULT_MAX_ROUNDS
) -> list[dict[str, Any]]:
    """The undo under test: open the cart, then remove lines until it is empty."""
    remove: dict[str, Any] = {
        "kind": "click",
        "via": via,
        "max_rounds": max_rounds,
    }
    if via == "dom":
        remove["find"] = "Remove "
    else:
        # A camera cannot read the trash icon's name, so aim from the price beside it.
        remove |= {"find": DIRTY_MARKER, "dx": TRASH_DX, "dy": TRASH_DY}
    remove["until_seen" if positive else "until_gone"] = EMPTY_MARKER if positive else DIRTY_MARKER
    return [{"kind": "click", "find": "Cart", "anchor_kind": "button", "via": via}, remove]


def run_reset(world: CartWorld, steps: Any, **kwargs: Any) -> tuple[Any, CartController]:
    """Build and call the reset for ``world``. Returns the ``ResetReport`` and hands."""
    controller = CartController(world, refuse=kwargs.pop("refuse", None))
    eyes = kwargs.pop("eyes", None) or PaintedEyes(world)
    truth = kwargs.pop("truth", "default")
    reset = world_reset_from_actions(
        reset_steps_from(steps),
        controller=controller,
        perceiver=eyes,
        truth=LabelledDom(world) if truth == "default" else truth,
        **kwargs,
    )
    return reset_world(reset), controller


# --------------------------------------------------------------------------------------
# The three promises
# --------------------------------------------------------------------------------------


class TestTheUndoIsReliable:
    """Idempotence, convergence, and an honest report when neither is reachable.

    A reset that empties the cart four times out of five makes the gate reject the
    skill, because attempt four starts dirty. These are the properties that stop that.
    """

    def test_an_already_clean_world_is_a_no_op_that_succeeds(self) -> None:
        """Running the undo twice must cost nothing the second time.

        This is the case the gate actually hits: the first reset empties the cart, the
        candidate is re-run, and the second reset arrives at a world that is already
        where it wants it. Erroring there - or clicking anyway - is how an undo that
        "works" still rejects the skill.
        """
        world = CartWorld(lines=0)
        report, controller = run_reset(world, empty_the_cart())
        assert report.outcome is ResetOutcome.restored
        assert world.lines == 0
        removals = [a for a in controller.performed if isinstance(a, Click)][1:]
        assert removals == [], "a clean world must not be clicked at"

    def test_it_removes_every_line_however_many_there_are(self) -> None:
        """Convergence: a loop against the screen, not a fixed number of clicks."""
        for count in (1, 2, 3):
            world = CartWorld(lines=count)
            report, controller = run_reset(world, empty_the_cart())
            assert report.outcome is ResetOutcome.restored, f"{count} lines"
            assert world.lines == 0, f"{count} lines"
            # One click to open the cart, then exactly one per line.
            assert len(controller.performed) == count + 1

    def test_the_same_undo_runs_three_times_over_as_the_gate_would(self) -> None:
        """The gate resets BETWEEN attempts, so run it the way the gate does.

        Dirty the world again between resets, as a re-run of the candidate skill
        would, and insist every attempt reaches the same clean screen. An undo that
        only works on the first attempt is the exact failure this feature exists to
        prevent, and it would pass every test above.
        """
        world = CartWorld(lines=3)
        for attempt in range(3):
            world.lines = 3
            world.screen = "shop"
            report, _ = run_reset(world, empty_the_cart())
            assert report.restored, f"attempt {attempt + 1}: {report}"
            assert world.lines == 0, f"attempt {attempt + 1}"

    def test_a_bound_it_cannot_meet_is_reported_as_failed_not_as_success(self) -> None:
        """The honest answer when the undo does not finish: ``failed``.

        Not ``restored`` with a dirty cart, and not a traceback out of a learning
        step. The gate then refuses the candidate and says why, which is the outcome
        the captain asked for: move the skill, not the failure.
        """
        world = CartWorld(lines=3)
        report, _ = run_reset(world, empty_the_cart(max_rounds=1))
        assert report.outcome is ResetOutcome.failed
        assert report.restored is False
        assert "ResetDidNotConverge" in report.detail
        assert world.lines == 2, "it did what it could before giving up"

    def test_the_bound_is_actions_performed_and_not_screens_read(self) -> None:
        """``max_rounds=3`` empties a three-line cart: three actions, four readings."""
        world = CartWorld(lines=3)
        report, _ = run_reset(world, empty_the_cart(max_rounds=3))
        assert report.outcome is ResetOutcome.restored and world.lines == 0


class TestSayingWhatCleanLooksLike:
    """``until_seen`` is a positive test, and a negative one is not enough.

    A negative test passes on every screen that does not say the marker, INCLUDING a
    screen an earlier step failed to reach. That is not hypothetical: a sandbox reset
    whose first step missed the cart button ran its loop on the mail inbox, found no
    cart lines there and reported the world restored with three lines still in it.
    """

    def test_a_positive_condition_refuses_to_pass_on_the_wrong_screen(self) -> None:
        world = CartWorld(lines=3)
        # The undo without its "open the cart" step: it never leaves the catalogue.
        stranded = empty_the_cart()[1:]
        report, _ = run_reset(world, stranded)
        assert report.outcome is ResetOutcome.failed
        assert world.lines == 3, "nothing was undone, and the report must not say it was"

    def test_a_negative_condition_alone_is_the_way_that_goes_wrong(self) -> None:
        """Documented, not endorsed: this is why ``until_seen`` exists at all."""
        world = CartWorld(lines=3)
        stranded = empty_the_cart(positive=False)[1:]
        report, _ = run_reset(world, stranded)
        assert report.outcome is ResetOutcome.restored, "the negative test is satisfied"
        assert world.lines == 3, "and the cart is still full - hence the warning logged"

    def test_doing_nothing_on_a_negative_condition_alone_is_logged_as_unverified(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The reader is the only thing that can tell 'clean' from 'wrong screen'."""
        world = CartWorld(lines=3)
        with caplog.at_level("WARNING"):
            run_reset(world, empty_the_cart(positive=False)[1:])
        assert any("unverified" in record.getMessage() for record in caplog.records)

    def test_both_conditions_together_must_both_hold(self) -> None:
        world = CartWorld(lines=2)
        steps = empty_the_cart()
        steps[1]["until_gone"] = DIRTY_MARKER
        report, _ = run_reset(world, steps)
        assert report.outcome is ResetOutcome.restored and world.lines == 0


class TestAimingAtWhatThePageSays:
    """An anchor is looked up, not ranked, and it is named by content not position."""

    def test_an_anchor_that_is_not_on_the_page_fails_loudly(self) -> None:
        world = CartWorld(lines=1)
        report, _ = run_reset(world, [{"kind": "click", "find": "Checkout"}])
        assert report.outcome is ResetOutcome.failed
        assert "Checkout" in report.detail

    def test_an_optional_anchor_that_is_missing_is_a_no_op(self) -> None:
        world = CartWorld(lines=1)
        report, controller = run_reset(
            world, [{"kind": "click", "find": "Dismiss", "optional": True}]
        )
        assert report.outcome is ResetOutcome.restored
        assert controller.performed == []

    def test_a_row_pushed_down_the_page_is_still_that_row(self) -> None:
        """The offset travels with the anchor, so a notice at the top changes nothing.

        Same undo, same cart, 200 pixels of banner in front of it. A reset that named
        a grid position would click the banner; this one clicks the same trash icons.
        """
        plain, pushed = CartWorld(lines=3), CartWorld(lines=3, top_offset=200)
        for world in (plain, pushed):
            report, _ = run_reset(world, empty_the_cart(via="screen"))
            assert report.outcome is ResetOutcome.restored, f"offset={world.top_offset}"
            assert world.lines == 0, f"offset={world.top_offset}"
        assert [p.y for p in plain.clicks] != [p.y for p in pushed.clicks]


class TestWhichReadingOfTheScreen:
    """``via`` picks pixels or the DOM, and the choice is the real-site one."""

    def test_a_control_named_only_by_aria_label_is_reachable_through_the_dom(self) -> None:
        """The reason ``via="dom"`` exists: DoorDash's remove control has no text.

        The painted screen carries an unnamed icon; only the DOM says "Remove Pad
        Thai". Aiming at that name through pixels cannot work and must not pretend to.
        """
        world = CartWorld(lines=2)
        report, _ = run_reset(world, empty_the_cart(via="dom"))
        assert report.outcome is ResetOutcome.restored and world.lines == 0

        blind = CartWorld(lines=2)
        report, _ = run_reset(
            blind,
            empty_the_cart(via="screen")[:1]
            + [{"kind": "click", "find": "Remove ", "via": "screen", "until_seen": EMPTY_MARKER}],
        )
        assert report.outcome is ResetOutcome.failed, "pixels cannot read an aria-label"
        assert blind.lines == 2

    def test_asking_for_a_dom_this_world_has_none_of_fails_saying_so(self) -> None:
        """No silent fallback to pixels: that would change which reading is trusted."""
        world = CartWorld(lines=1)
        report, _ = run_reset(world, empty_the_cart(via="dom"), truth=None)
        assert report.outcome is ResetOutcome.failed
        assert "ground-truth" in report.detail and "via='screen'" in report.detail

    def test_reading_the_dom_does_not_pay_for_an_observation(self) -> None:
        """OCR dominates perception, and a reset that never looks must never pay."""
        world = CartWorld(lines=3)
        eyes = PaintedEyes(world)
        report, _ = run_reset(world, empty_the_cart(via="dom"), eyes=eyes)
        assert report.outcome is ResetOutcome.restored
        assert eyes.calls == 0

    def test_eyes_that_cannot_see_are_a_failed_reset_not_a_crash(self) -> None:
        world = CartWorld(lines=1)
        report, _ = run_reset(
            world, empty_the_cart(via="screen"), eyes=PaintedEyes(world, blind=True)
        )
        assert report.outcome is ResetOutcome.failed
        assert "could not read the screen" in report.detail


class TestWhenTheHandsFail:
    """An action that was not delivered is a failed reset, and says which step."""

    def test_a_refused_action_stops_the_reset(self) -> None:
        world = CartWorld(lines=1)
        report, _ = run_reset(world, empty_the_cart(), refuse="the click went nowhere")
        assert report.outcome is ResetOutcome.failed
        assert "the click went nowhere" in report.detail


# --------------------------------------------------------------------------------------
# Writing one down
# --------------------------------------------------------------------------------------


class TestTheWrittenForm:
    """A step list is typed by hand, so it has to reject a mistyped one clearly."""

    def test_a_step_is_an_action_plus_what_this_module_adds(self) -> None:
        (step,) = reset_steps_from('[{"kind": "click", "find": "Cart", "dx": 12}]')
        assert isinstance(step.action, Click) and step.find == "Cart" and step.dx == 12

    def test_the_actions_kind_is_never_mistaken_for_the_anchors(self) -> None:
        """``kind`` names the ACTION; the anchor's kind is ``anchor_kind``.

        One key cannot mean two things, and when it did, parsing a click whose anchor
        was a button rewrote the action itself.
        """
        (step,) = reset_steps_from('[{"kind": "click", "find": "Cart", "anchor_kind": "button"}]')
        assert isinstance(step.action, Click)
        assert step.anchor_kind is ElementKind.button

    def test_a_step_round_trips_through_its_written_form(self) -> None:
        written = empty_the_cart()
        steps = reset_steps_from(written)
        assert reset_steps_from([reset_step_to_dict(s) for s in steps]) == steps

    def test_only_what_was_asked_for_is_written_back(self) -> None:
        (step,) = reset_steps_from('[{"kind": "press_key", "keys": ["Escape"]}]')
        assert reset_step_to_dict(step) == {"kind": "press_key", "keys": ["Escape"]}
        assert isinstance(step.action, PressKey)

    def test_an_empty_list_is_a_real_answer(self) -> None:
        assert reset_steps_from("[]") == () and reset_steps_from("") == ()

    @pytest.mark.parametrize(
        ("written", "complaint"),
        [
            ('[{"kind": "navigate", "url": "u", "find": "x"}]', "cannot be used with"),
            ('[{"kind": "click", "dx": 5, "point": {"x": 1, "y": 1}}]', "need"),
            (
                '[{"kind": "click", "anchor_kind": "button", "point": {"x": 1, "y": 1}}]',
                "needs one",
            ),
            ('[{"kind": "click", "find": "x", "via": "guess"}]', "'screen' or 'dom'"),
            ('[{"kind": "click", "find": "x", "max_rounds": 0}]', "at least 1"),
            ('[{"kind": "click", "find": "x", "optional": true, "until_gone": "y"}]', "contradict"),
            ('[{"kind": "click", "find": "x", "anchor_kind": "widget"}]', "anchor_kind"),
            ('[{"kind": "fly", "to": "moon"}]', "unknown action kind"),
            ("not json at all", "JSON array"),
            ('{"kind": "click"}', "ARRAY"),
        ],
    )
    def test_a_malformed_step_says_what_is_wrong(self, written: str, complaint: str) -> None:
        with pytest.raises(ValueError, match=complaint):
            reset_steps_from(written)

    def test_the_complaint_names_which_step_it_is_about(self) -> None:
        good = '{"kind": "wait", "ms": 1}'
        with pytest.raises(ValueError, match="reset step 3"):
            reset_steps_from(f'[{good}, {good}, {{"kind": "fly"}}]')


class TestBothUndoKindsCoexist:
    """``--reset-url`` keeps working, and a task may name both."""

    def test_they_run_in_the_order_they_were_chained(self) -> None:
        order: list[str] = []
        chained = chain_resets(lambda: order.append("url"), lambda: order.append("steps"))
        assert chained is not None
        chained()
        assert order == ["url", "steps"]

    def test_nothing_configured_stays_none_rather_than_becoming_a_no_op(self) -> None:
        """``None`` is "this world cannot be put back" and must survive as that."""
        assert chain_resets(None, None) is None

    def test_one_reset_is_passed_through_unwrapped(self) -> None:
        def only() -> None:
            return None

        assert chain_resets(None, only) is only


class TestTheSessionBuildsTheUndoTheTaskAsksFor:
    """``_reset_for`` is where a written step list becomes something callable."""

    def test_a_task_naming_steps_gets_a_reset_that_performs_them(self) -> None:
        from skillweaver.orchestrator import _reset_for, task_spec

        world = CartWorld(lines=2)
        spec = task_spec("Order a Pad Thai", url="https://shop.test/", reset_steps=empty_the_cart())
        controller = CartController(world)
        restore = _reset_for(spec, controller, PaintedEyes(world))
        assert restore is not None

        # No BrowserController here, so no DOM: the steps say via="dom" and must fail
        # saying so rather than quietly reading pixels instead.
        report = reset_world(restore)
        assert report.outcome is ResetOutcome.failed
        assert "ground-truth" in report.detail

    def test_a_task_naming_neither_kind_still_gets_nothing(self) -> None:
        from skillweaver.orchestrator import _reset_for, task_spec

        world = CartWorld(lines=1)
        spec = task_spec("just look at it", url="https://shop.test/")
        assert _reset_for(spec, CartController(world), PaintedEyes(world)) is None

    def test_a_task_naming_both_kinds_runs_the_endpoint_first(self) -> None:
        from skillweaver.orchestrator import _reset_for, task_spec

        world = CartWorld(lines=1)
        spec = task_spec(
            "Order a Pad Thai",
            url="https://shop.test/",
            reset_url="https://shop.test/__reset",
            reset_steps=empty_the_cart(via="screen"),
        )
        fetched: list[str] = []
        restore = _reset_for(spec, CartController(world), PaintedEyes(world))
        assert restore is not None

        import skillweaver.orchestrator as orchestrator

        original = orchestrator.world_reset_from_url
        try:
            # Rebuild through a stubbed fetcher so the endpoint is never called for
            # real; what is being pinned is the ORDER, which is the reason they chain.
            orchestrator.world_reset_from_url = lambda url, **_: lambda: fetched.append(url)
            restore = _reset_for(spec, CartController(world), PaintedEyes(world))
        finally:
            orchestrator.world_reset_from_url = original
        assert restore is not None
        report = reset_world(restore)
        assert fetched == ["https://shop.test/__reset"]
        assert report.outcome is ResetOutcome.restored
