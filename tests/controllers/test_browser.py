"""Real Chromium against real fixture pages.

Nothing here is mocked: a headless browser loads ``tests/fixtures/pages/*.html``
over ``file://`` (no network), the controller drives it with synthetic input at
coordinates, and the pages report back what actually happened by writing it into
visible text. The assertions read that text through :class:`BrowserGroundTruth`,
which is exactly the offline-teacher role it exists for.

No test resolves a selector to act. The controller is only ever given pixels,
because that is the only thing the agent will ever have.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import pytest

from skillweaver.contracts import (
    Box,
    Click,
    Controller,
    Drag,
    ElementKind,
    ElementSource,
    GroundTruthSource,
    Move,
    Navigate,
    Point,
    PressKey,
    Screenshot,
    Scroll,
    TypeText,
    Wait,
)
from skillweaver.controllers import _coords
from skillweaver.controllers.browser import (
    SOMETIMES_ONLY_OVERLAYS,
    BrowserController,
    BrowserGroundTruth,
)
from skillweaver.errors import ControllerError

PAGES = Path(__file__).resolve().parent.parent / "fixtures" / "pages"

VIEWPORT = (900, 650)

# Boxes as declared by tests/fixtures/pages/clicks.html, in LOGICAL pixels.
ALPHA = Box(40, 40, 160, 50)
BETA = Box(240, 40, 160, 50)
FIELD = Box(40, 140, 300, 40)
PAD = Box(40, 320, 400, 200)


def page_url(name: str) -> str:
    """A ``file://`` URL for a fixture page. Keeps every test off the network."""
    path = PAGES / name
    assert path.is_file(), f"missing fixture page: {path}"
    return path.as_uri()


def readout(truth: BrowserGroundTruth, prefix: str) -> str:
    """The text of the fixture's readout line starting with ``prefix``.

    Reading the page through ground truth, rather than a selector, keeps even the
    assertions honest about which side of the pixel boundary they are on.
    """
    for element in truth.elements():
        if element.text.startswith(prefix):
            return element.text
    raise AssertionError(f"no readout starting with {prefix!r} on the page")


@pytest.fixture(scope="module")
def controller():
    """One headless browser for the whole module; each test navigates it afresh."""
    with BrowserController(viewport=VIEWPORT) as ctl:
        yield ctl


@pytest.fixture
def truth(controller: BrowserController) -> BrowserGroundTruth:
    return BrowserGroundTruth(controller)


@pytest.fixture
def clicks_page(controller: BrowserController) -> BrowserController:
    assert controller.perform(Navigate(page_url("clicks.html"))).ok
    return controller


@pytest.fixture
def scroll_page(controller: BrowserController) -> BrowserController:
    assert controller.perform(Navigate(page_url("scroll.html"))).ok
    return controller


@pytest.fixture
def elements_page(controller: BrowserController) -> BrowserController:
    assert controller.perform(Navigate(page_url("elements.html"))).ok
    return controller


# --------------------------------------------------------------------------------------
# _coords: pure arithmetic, no browser
# --------------------------------------------------------------------------------------


class TestCoords:
    def test_png_size_reads_the_header(self) -> None:
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (37, 91)).save(buffer, format="PNG")
        assert _coords.png_size(buffer.getvalue()) == (37, 91)

    @pytest.mark.parametrize("data", [b"", b"not a png at all, really truly not", b"\x89PNG"])
    def test_png_size_rejects_anything_else(self, data: bytes) -> None:
        with pytest.raises(ValueError):
            _coords.png_size(data)

    def test_scale_for_is_physical_over_logical(self) -> None:
        assert _coords.scale_for(1600, 800) == 2.0
        assert _coords.scale_for(1200, 800) == 1.5
        assert _coords.scale_for(800, 800) == 1.0

    @pytest.mark.parametrize(("physical", "logical"), [(0, 800), (800, 0), (-4, 800)])
    def test_scale_for_rejects_impossible_sizes(self, physical: int, logical: int) -> None:
        with pytest.raises(ValueError):
            _coords.scale_for(physical, logical)

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
    def test_check_scale_refuses_a_scale_that_would_collapse_coordinates(
        self, scale: float
    ) -> None:
        with pytest.raises(ValueError):
            _coords.check_scale(scale)

    def test_physical_size_matches_the_screenshot_contract(self) -> None:
        assert _coords.physical_size(800, 600, 2.0) == (1600, 1200)
        assert _coords.physical_size(801, 601, 1.5) == (1202, 902)

    def test_point_round_trips_through_retina_pixels(self) -> None:
        point = Point(123, 456)
        physical = _coords.point_to_physical(point, 2.0)
        assert physical == Point(246, 912)
        assert _coords.point_to_logical(physical.x, physical.y, 2.0) == point

    def test_box_round_trips_through_retina_pixels(self) -> None:
        box = Box(10, 20, 100, 50)
        physical = _coords.box_to_physical(box, 2.0)
        assert physical == Box(20, 40, 200, 100)
        assert _coords.box_to_logical(20, 40, 200, 100, 2.0) == box

    def test_adjacent_boxes_stay_adjacent_after_conversion(self) -> None:
        """Converting edges, not sizes, is what stops a seam appearing between
        two touching detections at a fractional scale."""
        left = _coords.box_to_logical(0, 0, 75, 10, 1.5)
        right = _coords.box_to_logical(75, 0, 75, 10, 1.5)
        assert left.x + left.w == right.x
        assert left.w + right.w == _coords.box_to_logical(0, 0, 150, 10, 1.5).w

    def test_within_viewport_excludes_the_far_edges(self) -> None:
        viewport = Box(0, 0, 800, 600)
        assert _coords.within_viewport(viewport, Point(0, 0))
        assert _coords.within_viewport(viewport, Point(799, 599))
        assert not _coords.within_viewport(viewport, Point(800, 300))
        assert not _coords.within_viewport(viewport, Point(300, 600))
        assert not _coords.within_viewport(viewport, Point(-1, 300))

    def test_within_viewport_ignores_the_viewport_offset(self) -> None:
        """Action coordinates are relative to the viewport's top-left, so a
        controller reporting an offset must not shift what counts as on screen."""
        offset = Box(100, 50, 800, 600)
        assert _coords.within_viewport(offset, Point(0, 0))
        assert not _coords.within_viewport(offset, Point(800, 0))

    def test_clip_to_viewport_trims_and_zeroes(self) -> None:
        viewport = Box(0, 0, 800, 600)
        assert _coords.clip_to_viewport(Box(-20, -10, 100, 60), viewport) == Box(0, 0, 80, 50)
        assert _coords.clip_to_viewport(Box(760, 0, 100, 60), viewport) == Box(760, 0, 40, 60)
        assert _coords.clip_to_viewport(Box(900, 0, 100, 60), viewport).area == 0


# --------------------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------------------


class TestCapture:
    @pytest.mark.parametrize("scale", [1.0, 2.0, 1.5])
    def test_screenshot_geometry_matches_the_logical_viewport(self, scale: float) -> None:
        """The single most load-bearing fact in the project: what capture() says
        about the frame has to be true of the bytes it hands over."""
        with BrowserController(viewport=(640, 480), device_scale_factor=scale) as ctl:
            ctl.perform(Navigate(page_url("elements.html")))
            shot = ctl.capture()

        assert isinstance(shot, Screenshot)
        assert (shot.width, shot.height) == (640, 480)
        assert shot.scale == pytest.approx(scale)
        assert _coords.png_size(shot.png) == _coords.physical_size(640, 480, scale)
        assert shot.captured_at.tzinfo is not None

    def test_to_array_is_indexable_by_logical_pixel(self) -> None:
        """``array[y, x]`` must BE ``Point(x, y)``; this is where a 2x bug shows up."""
        with BrowserController(viewport=(640, 480), device_scale_factor=2.0) as ctl:
            ctl.perform(Navigate(page_url("elements.html")))
            shot = ctl.capture()

        assert shot.to_array().shape == (480, 640, 3)
        assert shot.to_array(logical=False).shape == (960, 1280, 3)

    def test_capture_reflects_the_page_that_is_loaded(self, controller: BrowserController) -> None:
        controller.perform(Navigate(page_url("elements.html")))
        elements_png = controller.capture().png
        controller.perform(Navigate(page_url("scroll.html")))
        assert controller.capture().png != elements_png


# --------------------------------------------------------------------------------------
# pointer and keyboard
# --------------------------------------------------------------------------------------


class TestPointer:
    def test_a_coordinate_click_activates_the_element_under_it(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        result = clicks_page.perform(Click(ALPHA.center))

        assert result.ok and result.error is None
        assert result.elapsed_ms > 0
        assert readout(truth, "last:") == "last: alpha"
        # The page saw the click at exactly the coordinate we asked for.
        assert readout(truth, "clicks:") == f"clicks: 1 at {ALPHA.center.x},{ALPHA.center.y}"

    def test_neighbouring_boxes_are_told_apart(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        clicks_page.perform(Click(BETA.center))
        assert readout(truth, "last:") == "last: beta"

        clicks_page.perform(Click(ALPHA.center))
        assert readout(truth, "last:") == "last: alpha"

    def test_a_click_just_outside_a_box_misses_it(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        """Proof the click lands where it is aimed rather than near it."""
        clicks_page.perform(Click(Point(ALPHA.x - 2, ALPHA.y - 2)))
        assert readout(truth, "last:") == "last: none"

        clicks_page.perform(Click(Point(ALPHA.x + 1, ALPHA.y + 1)))
        assert readout(truth, "last:") == "last: alpha"

    def test_double_click_is_delivered_as_one_gesture(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        assert clicks_page.perform(Click(ALPHA.center, clicks=2)).ok
        assert readout(truth, "last:") == "last: alpha x2"

    def test_move_does_not_press_anything(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        assert clicks_page.perform(Move(ALPHA.center)).ok
        assert readout(truth, "clicks:") == "clicks: 0 at -,-"

    def test_drag_presses_at_the_start_and_releases_at_the_end(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        start, end = Point(PAD.x + 40, PAD.y + 40), Point(PAD.x + 300, PAD.y + 150)
        assert clicks_page.perform(Drag(start, end)).ok
        assert readout(truth, "drag:") == f"drag: {start.x},{start.y} -> {end.x},{end.y}"


class TestKeyboard:
    def test_typed_text_lands_in_the_field_that_was_clicked(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        assert clicks_page.perform(Click(FIELD.center)).ok
        assert clicks_page.perform(TypeText("invoice 42")).ok
        assert readout(truth, "echo:") == "echo: invoice 42"

    def test_a_key_press_reaches_the_focused_field(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        clicks_page.perform(Click(FIELD.center))
        clicks_page.perform(TypeText("abc"))
        assert clicks_page.perform(PressKey(("Backspace",))).ok
        assert readout(truth, "echo:") == "echo: ab"

        assert clicks_page.perform(PressKey(("Enter",))).ok
        assert readout(truth, "keys:") == "keys: Enter"

    def test_a_chord_holds_its_modifiers_down_together(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        clicks_page.perform(Click(FIELD.center))
        assert clicks_page.perform(PressKey(("Control", "Shift", "k"))).ok
        assert readout(truth, "keys:") == "keys: Control+Shift+k"

    def test_modifiers_are_released_again(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        """A modifier left stuck down would silently corrupt every later action."""
        clicks_page.perform(Click(FIELD.center))
        clicks_page.perform(PressKey(("Control", "a")))
        clicks_page.perform(PressKey(("x",)))
        assert readout(truth, "keys:") == "keys: x"

    def test_a_held_modifier_really_changes_what_the_other_key_does(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        """Not just "the page saw a Shift flag": Shift+Arrow must actually extend a
        selection, so the following keystroke replaces two characters.

        Shift+Arrow rather than the more obvious select-all, because select-all is
        Meta+A on macOS and Control+A elsewhere - this asserts the controller, not
        the host's keyboard conventions.
        """
        clicks_page.perform(Click(FIELD.center))
        clicks_page.perform(TypeText("abcd"))
        clicks_page.perform(PressKey(("Shift", "ArrowLeft")))
        clicks_page.perform(PressKey(("Shift", "ArrowLeft")))
        clicks_page.perform(TypeText("X"))
        assert readout(truth, "echo:") == "echo: abX"


class TestScroll:
    def test_scrolling_changes_what_is_visible(
        self, scroll_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        before = scroll_page.capture().png
        visible_before = {el.text for el in truth.elements()}
        assert "Top marker" in visible_before
        assert "Deep marker" not in visible_before

        assert scroll_page.perform(Scroll(Point(450, 325), dy=1700)).ok

        visible_after = {el.text for el in truth.elements()}
        assert "Deep marker" in visible_after
        assert "Top marker" not in visible_after
        assert scroll_page.capture().png != before

    def test_scrolling_moves_content_by_the_distance_asked_for(
        self, scroll_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        def marker_box(text: str) -> Box:
            return next(el.box for el in truth.elements() if el.text == text)

        before = marker_box("Middle marker")
        assert scroll_page.perform(Scroll(Point(450, 325), dy=300)).ok

        assert readout(truth, "scrollY:") == "scrollY: 300"
        # The marker is still fully on screen, so its box moved by exactly the
        # distance asked for - no clipping, no rounding, no half-finished scroll.
        assert marker_box("Middle marker") == Box(before.x, before.y - 300, before.w, before.h)

    def test_a_scroll_has_finished_when_perform_returns(
        self, scroll_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        """A wheel is delivered asynchronously, so a controller that returned early
        would hand the next capture() a half-scrolled frame."""
        scroll_page.perform(Scroll(Point(450, 325), dy=900))
        settled = readout(truth, "scrollY:")

        scroll_page.perform(Wait(400))
        assert readout(truth, "scrollY:") == settled

    def test_scrolling_back_up_restores_the_first_view(
        self, scroll_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        scroll_page.perform(Scroll(Point(450, 325), dy=600))
        assert scroll_page.perform(Scroll(Point(450, 325), dy=-600)).ok
        assert readout(truth, "scrollY:") == "scrollY: 0"
        assert "Top marker" in {el.text for el in truth.elements()}


# --------------------------------------------------------------------------------------
# failures are reported, not raised
# --------------------------------------------------------------------------------------


class TestRefusals:
    @pytest.mark.parametrize(
        "point",
        [Point(2000, 50), Point(50, 2000), Point(-1, 50), Point(50, -1), Point(900, 650)],
    )
    def test_a_click_outside_the_viewport_is_reported_not_raised(
        self, clicks_page: BrowserController, point: Point
    ) -> None:
        result = clicks_page.perform(Click(point))

        assert result.ok is False
        assert result.error is not None
        assert "outside" in result.error and "900x650" in result.error

    def test_the_controller_is_still_usable_after_a_refusal(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        assert clicks_page.perform(Click(Point(5000, 5000))).ok is False
        assert clicks_page.perform(Click(ALPHA.center)).ok
        assert readout(truth, "last:") == "last: alpha"

    def test_a_drag_leaving_the_viewport_is_refused(self, clicks_page: BrowserController) -> None:
        result = clicks_page.perform(Drag(Point(100, 100), Point(100, 5000)))
        assert result.ok is False
        assert "outside" in (result.error or "")

    def test_a_scroll_anchored_outside_the_viewport_is_refused(
        self, clicks_page: BrowserController
    ) -> None:
        assert clicks_page.perform(Scroll(Point(-10, 10), dy=100)).ok is False

    def test_an_empty_key_chord_is_refused(self, clicks_page: BrowserController) -> None:
        result = clicks_page.perform(PressKey(()))
        assert result.ok is False
        assert "at least one key" in (result.error or "")

    def test_a_click_count_below_one_is_refused(self, clicks_page: BrowserController) -> None:
        assert clicks_page.perform(Click(ALPHA.center, clicks=0)).ok is False

    def test_a_negative_wait_is_refused(self, clicks_page: BrowserController) -> None:
        assert clicks_page.perform(Wait(-5)).ok is False

    def test_navigating_nowhere_is_reported(self, controller: BrowserController) -> None:
        result = controller.perform(Navigate("http://localhost:9/definitely-not-here"))

        assert result.ok is False
        assert result.error is not None
        assert "\n" not in result.error  # kept to one readable line for logs


# --------------------------------------------------------------------------------------
# ground truth: the offline teacher
# --------------------------------------------------------------------------------------

# Exactly what tests/fixtures/pages/elements.html plants, as text -> (kind, box).
PLANTED = {
    "Invoices": (ElementKind.text, Box(20, 20, 300, 40)),
    "Create invoice": (ElementKind.button, Box(20, 80, 140, 40)),
    "Search invoices": (ElementKind.text_field, Box(20, 140, 240, 36)),
    "Only unpaid": (ElementKind.checkbox, Box(20, 200, 24, 24)),
    "This month": (ElementKind.radio, Box(80, 200, 24, 24)),
    "Billing help": (ElementKind.link, Box(20, 250, 180, 24)),
    "Currency": (ElementKind.menu, Box(20, 300, 180, 32)),
    "Overdue": (ElementKind.tab, Box(20, 350, 120, 36)),
    "Company logo": (ElementKind.image, Box(20, 400, 100, 60)),
}


class TestBlockedRequests:
    """A page that renders an overlay only SOMETIMES must not be two screens.

    ``overlay.html`` is that page offline: it is 420px taller when the script it
    fetches arrives, and its readout line says which of the two it became. Live,
    the script is Wikipedia's ``Special:BannerLoader`` - see
    :data:`~skillweaver.controllers.browser.SOMETIMES_ONLY_OVERLAYS` for the
    measurement that put the default there.
    """

    def test_the_overlay_renders_when_nothing_is_blocked(self) -> None:
        with BrowserController(viewport=VIEWPORT, block=()) as ctl:
            assert ctl.perform(Navigate(page_url("overlay.html"))).ok
            assert readout(BrowserGroundTruth(ctl), "overlay:") == "overlay: shown"

    def test_a_blocked_request_cannot_render_its_overlay(self) -> None:
        with BrowserController(viewport=VIEWPORT, block=("**/overlay-banner.js",)) as ctl:
            assert ctl.perform(Navigate(page_url("overlay.html"))).ok
            assert readout(BrowserGroundTruth(ctl), "overlay:") == "overlay: absent"

    def test_the_block_survives_navigating_away_and_back(self) -> None:
        """The route is installed on the context, so one navigation cannot shed it."""
        with BrowserController(viewport=VIEWPORT, block=("**/overlay-banner.js",)) as ctl:
            assert ctl.perform(Navigate(page_url("overlay.html"))).ok
            assert ctl.perform(Navigate(page_url("link.html"))).ok
            assert ctl.perform(Navigate(page_url("overlay.html"))).ok
            assert readout(BrowserGroundTruth(ctl), "overlay:") == "overlay: absent"

    def test_blocking_leaves_the_rest_of_the_page_working(self) -> None:
        """Only the overlay goes. A block that cost the run its page would be worse
        than the screen it was tidying up."""
        with BrowserController(viewport=VIEWPORT, block=("**/overlay-banner.js",)) as ctl:
            assert ctl.perform(Navigate(page_url("overlay.html"))).ok
            texts = [el.text for el in BrowserGroundTruth(ctl).elements()]
            assert "overlay: absent" in texts

    def test_the_default_blocks_the_three_measured_wikipedia_endpoints(self) -> None:
        """The shipped default, pinned. Losing one of these is a demo that fails on
        stage roughly one load in five, which is how it was found."""
        with BrowserController(viewport=(320, 240)) as ctl:
            assert ctl.blocked == SOMETIMES_ONLY_OVERLAYS
        assert [p.pattern for p in SOMETIMES_ONLY_OVERLAYS] == [
            "Special:BannerLoader",
            "Special:RecordImpression",
            "geoiplookup",
        ]

    def test_the_default_patterns_match_the_url_wikipedia_actually_serves(self) -> None:
        """A glob cannot: CentralNotice puts the page name in the QUERY STRING, and a
        Playwright glob matches path segments. Blocking it with ``**/Special:...*``
        aborted nothing on a live forced appeal, and the banner rendered at 531px."""
        served = (
            "https://meta.wikimedia.org/w/index.php?title=Special:BannerLoader"
            "&campaign=WMF_FR_FY2627_en6C_dsk_0701&banner=B2627_091718_en6C_dsk_p1_lg"
            "&uselang=en&debug=false"
        )
        assert any(p.search(served) for p in SOMETIMES_ONLY_OVERLAYS)
        assert not fnmatch(served, "**/Special:BannerLoader*")

    def test_blocking_can_be_turned_off_entirely(self) -> None:
        with BrowserController(viewport=(320, 240), block=()) as ctl:
            assert ctl.blocked == ()


class TestGroundTruth:
    def test_it_finds_every_planted_element_with_the_right_kind_and_box(
        self, elements_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        found = {el.text: (el.kind, el.box) for el in truth.elements()}
        assert found == PLANTED

    def test_it_leaves_out_what_cannot_be_seen(
        self, elements_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        """display:none, visibility:hidden and below-the-fold, all planted in the
        fixture so their absence is a real assertion and not a coincidence."""
        texts = {el.text for el in truth.elements()}
        assert "Never rendered" not in texts
        assert "Hidden from view" not in texts
        assert "Below the fold" not in texts

    def test_every_element_is_labelled_as_dom_ground_truth(
        self, elements_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        elements = truth.elements()
        assert elements
        assert all(el.source is ElementSource.dom for el in elements)
        assert all(el.confidence == 1.0 for el in elements)
        assert all(el.stable_id for el in elements)

    def test_elements_come_back_in_reading_order(
        self, elements_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        corners = [(el.box.y, el.box.x) for el in truth.elements()]
        assert corners == sorted(corners)

    def test_stable_ids_survive_a_reload_and_differ_between_elements(
        self, elements_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        first = {el.text: el.stable_id for el in truth.elements()}
        elements_page.perform(Navigate(page_url("elements.html")))
        assert {el.text: el.stable_id for el in truth.elements()} == first
        assert len(set(first.values())) == len(first)

    def test_boxes_are_clipped_to_what_is_on_screen(self, truth: BrowserGroundTruth) -> None:
        """A tall element must be labelled with the rectangle a detector could
        actually have seen, never one running off the frame."""
        with BrowserController(viewport=(400, 300)) as ctl:
            ctl.perform(Navigate(page_url("elements.html")))
            viewport = ctl.viewport()
            elements = BrowserGroundTruth(ctl).elements()

        assert elements
        for el in elements:
            assert el.box.x >= 0 and el.box.y >= 0
            assert el.box.x + el.box.w <= viewport.w
            assert el.box.y + el.box.h <= viewport.h

    def test_its_boxes_are_in_the_same_space_the_controller_clicks_in(
        self, clicks_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        """The whole system rests on this: a box the teacher reports, clicked at
        its center, activates that element."""
        box = next(el.box for el in truth.elements() if el.text == "Alpha")
        assert box == ALPHA

        assert clicks_page.perform(Click(box.center)).ok
        assert readout(truth, "last:") == "last: alpha"

    def test_a_click_that_navigates_leaves_a_loaded_page_behind(
        self, controller: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        """A click sets a navigation going after the click itself is delivered, so
        a controller that returned too early would hand the agent's next capture a
        blank half-committed frame."""
        assert controller.perform(Navigate(page_url("link.html"))).ok
        box = next(el.box for el in truth.elements() if el.text == "Go to elements")

        assert controller.perform(Click(box.center)).ok

        assert controller.url() == page_url("elements.html")
        assert "Create invoice" in {el.text for el in truth.elements()}

    def test_it_reports_the_current_url(
        self, elements_page: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        assert truth.url() == page_url("elements.html")

    def test_the_controller_never_hands_one_out(self) -> None:
        """It is an offline teacher. If a controller could produce one, an agent
        on the action path could reach it by accident."""
        surface = {name for name in dir(BrowserController) if not name.startswith("_")}
        assert surface == {
            "blocked",  # URL patterns it refuses - configuration, not a reading of the page
            "capture",
            "close",
            "describe",
            "perform",
            "supports",
            "url",
            "viewport",
        }


# --------------------------------------------------------------------------------------
# the rest of the protocol
# --------------------------------------------------------------------------------------


class TestProtocol:
    def test_both_classes_satisfy_the_protocols_they_claim(
        self, controller: BrowserController, truth: BrowserGroundTruth
    ) -> None:
        assert isinstance(controller, Controller)
        assert isinstance(truth, GroundTruthSource)

    def test_it_supports_every_action_kind_including_navigate(
        self, controller: BrowserController
    ) -> None:
        for kind in ("click", "move", "drag", "type_text", "press_key", "scroll", "wait"):
            assert controller.supports(kind) is True
        assert controller.supports("navigate") is True
        assert controller.supports("teleport") is False  # type: ignore[arg-type]

    def test_viewport_is_the_logical_page_anchored_at_the_origin(
        self, controller: BrowserController
    ) -> None:
        assert controller.viewport() == Box(0, 0, *VIEWPORT)

    def test_url_follows_navigation(self, controller: BrowserController) -> None:
        controller.perform(Navigate(page_url("scroll.html")))
        assert controller.url() == page_url("scroll.html")

    def test_describe_is_one_useful_line(self, controller: BrowserController) -> None:
        assert controller.describe() == "playwright chromium 900x650 @1x"

    def test_describe_says_when_a_window_is_open(self) -> None:
        with BrowserController(viewport=(320, 240), device_scale_factor=2.0) as ctl:
            assert ctl.describe() == "playwright chromium 320x240 @2x"

    def test_wait_costs_the_time_it_says(self, controller: BrowserController) -> None:
        result = controller.perform(Wait(250))
        assert result.ok
        assert result.elapsed_ms >= 240

    def test_the_context_manager_closes_the_browser(self) -> None:
        with BrowserController(viewport=(320, 240)) as ctl:
            assert ctl.capture().width == 320
        with pytest.raises(ControllerError):
            ctl.capture()

    def test_the_context_manager_closes_after_a_failure(self) -> None:
        ctl = BrowserController(viewport=(320, 240))
        with pytest.raises(ZeroDivisionError):
            with ctl:
                raise ZeroDivisionError("something inside the block failed")
        with pytest.raises(ControllerError):
            ctl.capture()

    def test_close_is_idempotent_and_a_closed_controller_refuses_work(self) -> None:
        ctl = BrowserController(viewport=(320, 240))
        ctl.close()
        ctl.close()  # must not raise

        with pytest.raises(ControllerError):
            ctl.capture()
        with pytest.raises(ControllerError):
            ctl.perform(Click(Point(1, 1)))
        with pytest.raises(ControllerError):
            BrowserGroundTruth(ctl).elements()

    def test_start_url_is_loaded_at_construction(self) -> None:
        with BrowserController(viewport=(320, 240), start_url=page_url("scroll.html")) as ctl:
            assert ctl.url() == page_url("scroll.html")

    @pytest.mark.parametrize("viewport", [(0, 600), (800, -1)])
    def test_an_impossible_viewport_is_refused_before_launching(
        self, viewport: tuple[int, int]
    ) -> None:
        with pytest.raises(ValueError):
            BrowserController(viewport=viewport)
