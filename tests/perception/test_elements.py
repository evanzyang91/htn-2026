"""The element index, the source merge, stable ids, cropping and screenshot conversion.

These are the pure, fast half of perception: no model, no browser, no network. The
elements under test are *planted* - either written out explicitly here, or taken from the
ground-truth DOM boxes that ``tests/fixtures/shots/generate.py`` recorded next to the
committed screenshots - so every expectation is exact. ``test_ocr.py`` covers the same
fixtures with the real reader.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from skillweaver import contracts
from skillweaver.contracts import Box, Element, ElementKind, ElementSource, Point, Screenshot
from skillweaver.errors import PerceptionError
from skillweaver.perception import crop as cropping
from skillweaver.perception import screenshot as shots
from skillweaver.perception.elements import (
    MERGED_TEXT_LIMIT,
    STABLE_ID_GRID,
    ElementIndex,
    build_index,
    merge_elements,
    normalize_text,
    overlap_ratio,
    reading_order,
    stable_id,
    with_stable_ids,
)

SHOTS = Path(__file__).resolve().parents[1] / "fixtures" / "shots"


def _expectations() -> dict:
    return json.loads((SHOTS / "expectations.json").read_text())


def _shot_record(name: str) -> dict:
    for record in _expectations()["shots"]:
        if record["name"] == name:
            return record
    raise AssertionError(f"no fixture named {name}")


def _planted(name: str) -> list[Element]:
    """The fixture page's ground-truth elements: exactly what is on the screenshot."""
    record = _shot_record(name)
    return [
        Element(
            box=Box(*el["box"]),
            kind=ElementKind(el["kind"]),
            text=el["text"],
            confidence=1.0,
            source=ElementSource.dom,
        )
        for el in record["elements"]
    ]


def _load(name: str) -> Screenshot:
    record = _shot_record(name)
    return shots.load_screenshot(
        SHOTS / record["png"],
        scale=record["scale"],
        width=record["width"],
        height=record["height"],
    )


@pytest.fixture
def invoices() -> list[Element]:
    return _planted("invoices@1x")


@pytest.fixture
def index(invoices: list[Element]) -> ElementIndex:
    return ElementIndex(invoices)


def _element(
    box: Box,
    kind: ElementKind = ElementKind.other,
    text: str = "",
    confidence: float = 0.9,
    source: ElementSource = ElementSource.yolo,
) -> Element:
    return Element(box=box, kind=kind, text=text, confidence=confidence, source=source)


# --------------------------------------------------------------------------------------
# Fixture sanity: the planted data has to be what the tests below assume
# --------------------------------------------------------------------------------------


def test_fixture_plants_the_page_we_think_it_does(invoices: list[Element]) -> None:
    texts = [e.text for e in invoices]
    assert "Open Invoices" in texts
    assert "New Invoice" in texts
    assert {e.kind for e in invoices} >= {
        ElementKind.text,
        ElementKind.text_field,
        ElementKind.button,
        ElementKind.row,
        ElementKind.link,
    }


# --------------------------------------------------------------------------------------
# normalize_text and reading order
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Open   Invoices  ", "open invoices"),
        ("SEARCH:", "search"),
        ("[ Search ]", "search"),
        ("billing_settings", "billing settings"),
        ("Send\nReminders", "send reminders"),
        ("", ""),
        ("...", ""),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_reading_order_groups_a_row_whose_tops_differ() -> None:
    # The invoices toolbar: a 44px field and two 44px buttons, then a label a pixel off.
    field = _element(Box(40, 132, 300, 44), ElementKind.text_field, "Search invoices")
    button = _element(Box(360, 133, 120, 42), ElementKind.button, "Search")
    heading = _element(Box(40, 32, 300, 40), ElementKind.text, "Open Invoices")
    below = _element(Box(40, 220, 610, 48), ElementKind.row, "INV-1042")
    ordered = reading_order([below, button, heading, field])
    assert [e.text for e in ordered] == ["Open Invoices", "Search invoices", "Search", "INV-1042"]


def test_reading_order_of_the_real_page_starts_at_the_heading(index: ElementIndex) -> None:
    assert index.all()[0].text == "Open Invoices"
    assert index.all()[-1].text in {"Send Reminders", "3 invoices overdue"}


# --------------------------------------------------------------------------------------
# ElementIndex: the Protocol surface
# --------------------------------------------------------------------------------------


def test_index_satisfies_the_contract_protocol(index: ElementIndex) -> None:
    assert isinstance(index, contracts.ElementIndex)


def test_build_index_is_the_same_thing(invoices: list[Element]) -> None:
    assert build_index(invoices).all() == ElementIndex(invoices).all()


def test_all_returns_every_planted_element(index: ElementIndex, invoices: list[Element]) -> None:
    assert len(index.all()) == len(invoices)
    assert set(index.all()) == set(invoices)


def test_by_kind_returns_only_that_kind_in_reading_order(index: ElementIndex) -> None:
    buttons = index.by_kind(ElementKind.button)
    assert [e.text for e in buttons] == ["Search", "New Invoice", "Send Reminders"]
    assert index.by_kind(ElementKind.radio) == []


def test_every_query_returns_a_list_never_none() -> None:
    empty = ElementIndex([])
    assert empty.all() == []
    assert empty.by_kind(ElementKind.button) == []
    assert empty.find_text("anything") == []
    assert empty.nearest(Point(10, 10)) == []
    assert empty.containing(Point(10, 10)) == []
    assert empty.best("a blue button") == []


# --- find_text ------------------------------------------------------------------------


def test_find_text_is_case_and_whitespace_insensitive(index: ElementIndex) -> None:
    for query in ("New Invoice", "new invoice", "  NEW   INVOICE ", "new invoice:"):
        assert index.find_text(query)[0].text == "New Invoice"


def test_find_text_ranks_exact_above_prefix_above_substring(index: ElementIndex) -> None:
    # "Search" is the button's whole text and the prefix of the field's placeholder.
    hits = index.find_text("Search")
    assert [e.kind for e in hits[:2]] == [ElementKind.button, ElementKind.text_field]

    # "invoices" is a substring of several; the tightest container ranks first, so the
    # 13-character heading beats the 15-character placeholder.
    substring_hits = index.find_text("invoices")
    assert substring_hits[0].text == "Open Invoices"
    assert {e.text for e in substring_hits} >= {"Open Invoices", "Search invoices"}


def test_find_text_finds_a_fragment_inside_a_longer_row(index: ElementIndex) -> None:
    hits = index.find_text("Initech")
    assert hits and "Initech" in hits[0].text
    assert hits[0].kind is ElementKind.row


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Downlaod report", "Download report"),  # transposed pair
        ("Dowload report", "Download report"),  # dropped character
        ("Download rep0rt", "Download report"),  # o -> 0, the classic OCR slip
        ("NewInvoice", "New Invoice"),  # OCR ate the space
        ("Send Rerninders", "Send Reminders"),  # m -> rn, the other classic
        ("BillingSettings", "Billing settings"),
        ("acme corporation biling", "Acme Corporation billing"),
    ],
)
def test_find_text_tolerates_injected_ocr_noise(
    index: ElementIndex, query: str, expected: str
) -> None:
    hits = index.find_text(query)
    assert hits, f"{query!r} found nothing"
    assert hits[0].text == expected


@pytest.mark.parametrize("position", [1, 3, -2])
def test_find_text_tolerates_one_wrong_character_anywhere(
    index: ElementIndex, invoices: list[Element], position: int
) -> None:
    """Mutate one character of every reasonably long label and still find it."""
    for element in invoices:
        text = element.text
        if len(text) < 9:
            continue
        i = position % len(text)
        if not text[i].isalpha():
            continue
        wrong = "x" if text[i].lower() != "x" else "q"
        noisy = text[:i] + wrong + text[i + 1 :]
        hits = index.find_text(noisy)
        assert hits, f"{noisy!r} (from {text!r}) found nothing"
        assert hits[0].text == text, f"{noisy!r} found {hits[0].text!r}, wanted {text!r}"


def test_find_text_without_fuzzy_rejects_noise_but_keeps_literal(index: ElementIndex) -> None:
    assert index.find_text("Dowload report", fuzzy=False) == []
    assert index.find_text("Download report", fuzzy=False)[0].text == "Download report"
    assert index.find_text("download", fuzzy=False)[0].text == "Download report"


def test_find_text_restricts_to_a_kind(index: ElementIndex) -> None:
    assert index.find_text("Search", kind=ElementKind.text_field)[0].kind is ElementKind.text_field
    assert index.find_text("Search", kind=ElementKind.link) == []


def test_find_text_returns_empty_for_a_blank_or_absent_query(index: ElementIndex) -> None:
    assert index.find_text("") == []
    assert index.find_text("   ") == []
    assert index.find_text("quarterly depreciation schedule") == []


# --- nearest / containing -------------------------------------------------------------


def test_nearest_starts_inside_and_walks_outward(index: ElementIndex) -> None:
    # Centre of the "Search" button at Box(360, 132, 120, 44).
    point = Point(420, 154)
    hits = index.nearest(point)
    assert hits[0].text == "Search", "the point is inside it, so distance 0"
    assert len(hits) == len(index.all()), "nearest ranks everything, it does not filter"

    def distance(element: Element) -> float:
        box = element.box
        dx = max(box.x - point.x, 0, point.x - (box.x + box.w - 1))
        dy = max(box.y - point.y, 0, point.y - (box.y + box.h - 1))
        return (dx**2 + dy**2) ** 0.5

    distances = [distance(e) for e in hits]
    assert distances == sorted(distances)


def test_nearest_restricts_to_a_kind(index: ElementIndex) -> None:
    hits = index.nearest(Point(420, 154), kind=ElementKind.link)
    assert [e.text for e in hits] == ["Billing settings", "Download report"]


def test_nearest_prefers_the_smaller_box_when_both_contain_the_point() -> None:
    row = _element(Box(0, 0, 400, 60), ElementKind.row, "row")
    button = _element(Box(300, 10, 80, 40), ElementKind.button, "Open")
    index = ElementIndex([row, button])
    assert index.nearest(Point(340, 30))[0].text == "Open"


def test_containing_returns_most_specific_first() -> None:
    row = _element(Box(0, 0, 400, 60), ElementKind.row, "row")
    button = _element(Box(300, 10, 80, 40), ElementKind.button, "Open")
    icon = _element(Box(310, 20, 16, 16), ElementKind.icon, "")
    index = ElementIndex([row, button, icon])
    assert [e.kind for e in index.containing(Point(315, 25))] == [
        ElementKind.icon,
        ElementKind.button,
        ElementKind.row,
    ]
    assert index.containing(Point(5, 5)) == [row]
    assert index.containing(Point(900, 900)) == []


def test_containing_excludes_the_right_and_bottom_edge() -> None:
    index = ElementIndex([_element(Box(10, 10, 10, 10), ElementKind.button, "b")])
    assert index.containing(Point(10, 10))
    assert index.containing(Point(19, 19))
    assert index.containing(Point(20, 20)) == []


# --- best -----------------------------------------------------------------------------


def test_best_uses_the_kind_word_to_pick_between_lookalikes(index: ElementIndex) -> None:
    assert index.best("search field")[0].kind is ElementKind.text_field
    assert index.best("search button")[0].kind is ElementKind.button


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("blue New Invoice button", "New Invoice"),
        ("the Send Reminders button", "Send Reminders"),
        ("download report link", "Download report"),
        ("Globex row", "INV-1042 Globex $1,280.00"),
        ("Open Invoices heading", "Open Invoices"),
        # Position words cannot be evaluated from an index, so they are ignored rather
        # than allowed to sink every candidate: this falls back to "the first heading".
        ("the heading at the top", "Open Invoices"),
    ],
)
def test_best_matches_a_free_form_description(
    index: ElementIndex, description: str, expected: str
) -> None:
    hits = index.best(description)
    assert hits, f"{description!r} matched nothing"
    assert hits[0].text == expected


def test_best_with_only_a_kind_word_lists_that_kind(index: ElementIndex) -> None:
    hits = index.best("a button")
    assert hits
    assert all(e.kind is ElementKind.button for e in hits)


def test_best_matches_nothing_when_the_named_text_is_absent(index: ElementIndex) -> None:
    assert index.best("Kubernetes namespace field") == []


# --- by_id ----------------------------------------------------------------------------


def test_by_id_finds_the_element_that_carries_the_id(invoices: list[Element]) -> None:
    index = ElementIndex(with_stable_ids(invoices))
    target = index.all()[3]
    assert target.stable_id is not None
    assert index.by_id(target.stable_id) == [target]
    assert index.by_id("deadbeefdeadbeef") == []


# --------------------------------------------------------------------------------------
# stable_id
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dx", [-4, -1, 0, 3, 5])
@pytest.mark.parametrize("dy", [-3, 0, 2, 4])
def test_stable_id_survives_a_few_pixels_of_drift(
    invoices: list[Element], dx: int, dy: int
) -> None:
    """A whole page nudged by a few pixels is the same page."""
    for element in invoices:
        shifted = cropping.translate_elements([element], dx, dy)[0]
        assert stable_id(shifted) == stable_id(element), f"{element.text!r} moved by ({dx},{dy})"


def test_stable_id_changes_when_the_text_changes(invoices: list[Element]) -> None:
    import dataclasses

    for element in invoices:
        if not element.text:
            continue
        renamed = dataclasses.replace(element, text=element.text + " (overdue)")
        assert stable_id(renamed) != stable_id(element)


def test_stable_id_ignores_cosmetic_text_differences() -> None:
    a = _element(Box(100, 100, 80, 30), ElementKind.button, "Search")
    b = _element(Box(100, 100, 80, 30), ElementKind.button, "  SEARCH:  ")
    assert stable_id(a) == stable_id(b)


def test_stable_id_changes_with_kind_and_with_a_real_move() -> None:
    import dataclasses

    button = _element(Box(100, 100, 80, 30), ElementKind.button, "Search")
    assert stable_id(dataclasses.replace(button, kind=ElementKind.link)) != stable_id(button)
    far = cropping.translate_elements([button], 0, 6 * STABLE_ID_GRID)[0]
    assert stable_id(far) != stable_id(button)


def test_stable_id_is_deterministic_and_short() -> None:
    button = _element(Box(100, 100, 80, 30), ElementKind.button, "Search")
    first = stable_id(button)
    assert first == stable_id(button)
    assert len(first) == 16 and first.isalnum()


def test_with_stable_ids_fills_gaps_and_respects_existing_ids() -> None:
    import dataclasses

    plain = _element(Box(10, 10, 20, 20), ElementKind.button, "Go")
    known = dataclasses.replace(plain, stable_id="from-the-dom")
    filled = with_stable_ids([plain, known])
    assert filled[0].stable_id == stable_id(plain)
    assert filled[1].stable_id == "from-the-dom"
    assert with_stable_ids([known], overwrite=True)[0].stable_id == stable_id(known)


def test_stable_id_rejects_a_nonsense_grid() -> None:
    with pytest.raises(ValueError):
        stable_id(_element(Box(0, 0, 1, 1)), grid=0)


# --------------------------------------------------------------------------------------
# merge_elements
# --------------------------------------------------------------------------------------


def test_overlap_ratio_answers_containment_where_iou_does_not() -> None:
    button = Box(360, 132, 120, 44)
    label = Box(390, 145, 60, 18)
    assert button.iou(label) < 0.3
    assert overlap_ratio(button, label) == pytest.approx(1.0)
    assert overlap_ratio(button, Box(0, 0, 10, 10)) == 0.0


def test_merge_collapses_a_yolo_box_and_the_ocr_line_inside_it() -> None:
    """The case the whole function exists for: one button seen by two producers."""
    detected = _element(Box(360, 132, 120, 44), ElementKind.button, "", 0.82, ElementSource.yolo)
    read = _element(Box(390, 145, 60, 18), ElementKind.text, "Search", 0.97, ElementSource.ocr)
    merged = merge_elements([detected], [read])

    assert len(merged) == 1
    only = merged[0]
    assert only.kind is ElementKind.button, "the specific kind wins"
    assert only.text == "Search", "the text comes from OCR"
    assert only.box == detected.box, "the click target is the button, not the glyphs"
    assert only.confidence == pytest.approx(0.97)
    assert only.source is ElementSource.merged
    assert only.stable_id == stable_id(only)


def test_merge_collapses_a_near_duplicate_box_by_iou() -> None:
    a = _element(Box(40, 132, 300, 44), ElementKind.text_field, "", 0.7, ElementSource.yolo)
    b = _element(Box(42, 134, 296, 42), ElementKind.text_field, "", 0.9, ElementSource.yolo)
    assert a.box.iou(b.box) > 0.55
    merged = merge_elements([a, b])
    assert len(merged) == 1
    assert merged[0].confidence == pytest.approx(0.9)
    assert merged[0].source is ElementSource.merged


def test_a_fused_row_says_every_line_printed_inside_it() -> None:
    """A list row is several lines and any of them is how a person would name it.

    Keeping only the longest would leave a message row saying its body preview and
    nothing else, so "the message from Billing titled 'Invoice 4471 is ready'" could
    match neither the sender nor the subject that are plainly on screen.
    """
    row = _element(Box(250, 300, 950, 80), ElementKind.button, "", 0.8, ElementSource.yolo)
    sender = _element(Box(258, 306, 90, 18), ElementKind.text, "Billing", 0.95, ElementSource.ocr)
    subject = _element(
        Box(258, 326, 300, 20), ElementKind.text, "Invoice 4471 is ready", 0.95, ElementSource.ocr
    )
    preview = _element(
        Box(258, 350, 600, 18),
        ElementKind.text,
        "Invoice 4471 for the March hosting term is available.",
        0.9,
        ElementSource.ocr,
    )

    merged = merge_elements([row], [sender, subject, preview])

    fused = next(e for e in merged if e.box == row.box)
    assert fused.kind is ElementKind.button, "the click target is still the row"
    assert fused.text == (
        "Billing Invoice 4471 is ready Invoice 4471 for the March hosting term is available."
    )
    index = ElementIndex(merged)
    assert index.find_text("Billing"), "the sender is findable"
    assert index.find_text("Invoice 4471 is ready"), "the subject is findable"
    # Every line is also kept where it is printed. Several lines is what makes this a
    # container rather than a labelled control, and what is clickable in a container is
    # usually one of its lines rather than the middle of it.
    assert {e.box for e in merged if e.source is ElementSource.ocr} == {
        sender.box,
        subject.box,
        preview.box,
    }


def test_a_fused_control_does_not_repeat_a_line_it_reads_twice() -> None:
    button = _element(Box(40, 40, 120, 40), ElementKind.button, "", 0.8, ElementSource.yolo)
    once = _element(Box(48, 50, 60, 18), ElementKind.text, "Save", 0.9, ElementSource.ocr)
    again = _element(Box(50, 52, 58, 16), ElementKind.text, " save ", 0.8, ElementSource.ocr)
    assert merge_elements([button], [once, again])[0].text == "Save"


def test_ground_truth_text_is_not_padded_out_with_ocrs_guess_at_the_same_words() -> None:
    """Only the most trusted source present contributes, so a control never says its
    own label twice."""
    button = _element(Box(40, 40, 120, 40), ElementKind.button, "Sign In", 0.9, ElementSource.dom)
    guess = _element(Box(48, 50, 60, 18), ElementKind.text, "SignIn", 0.7, ElementSource.ocr)
    assert merge_elements([button], [guess])[0].text == "Sign In"


def test_a_very_wordy_control_is_cut_after_the_identifying_lines() -> None:
    """Reading order puts the short identifying lines first, so a cut loses the tail
    of the body copy rather than the sender."""
    row = _element(Box(0, 0, 900, 80), ElementKind.row, "", 0.8, ElementSource.yolo)
    who = _element(Box(8, 4, 80, 16), ElementKind.text, "Northwind IT", 0.9, ElementSource.ocr)
    body = _element(Box(8, 40, 860, 18), ElementKind.text, "word " * 60, 0.9, ElementSource.ocr)

    text = merge_elements([row], [who, body])[0].text

    assert text.startswith("Northwind IT ")
    assert len(text) <= MERGED_TEXT_LIMIT
    assert text.endswith("…")


def test_a_paragraph_does_not_fuzzily_answer_a_two_word_query() -> None:
    """A sentence containing one of the words is not the control you asked for.

    On the live sandbox the settings page explains "Changes are not applied until you
    confirm them", and a search for "Save changes" matched that paragraph - because a
    bare "changes" scores 0.74 against the query. The agent clicked the explanation,
    never scrolled to the save bar, and the task failed with the button on the page.
    """
    prose = _element(
        Box(289, 109, 466, 21),
        ElementKind.text,
        "Preferences for this console. Changes are not applied until you confirm them.",
    )
    button = _element(Box(817, 705, 120, 32), ElementKind.button, "Savechanges")
    index = ElementIndex([prose, button])

    hits = index.find_text("Save changes")

    assert [e.kind for e in hits] == [ElementKind.button], (
        "only the button, and the OCR-run-together label still matches fuzzily"
    )


def test_a_single_word_query_still_matches_one_word_inside_a_line() -> None:
    """The narrowing is about multi-word queries: a one-word query is still answered
    by one word, which is what absorbs OCR mangling a word in the middle of a line."""
    row = _element(Box(0, 0, 400, 20), ElementKind.row, "Subrnit order now")
    assert ElementIndex([row]).find_text("Submit")


def test_a_glyph_inside_a_field_does_not_adopt_the_fields_label() -> None:
    """ "Tightest container" has to mean tightest CONTAINER.

    The magnifier drawn inside the mail search box is fourteen pixels wide, and by a
    symmetric overlap measure it "contains" the placeholder printed beside it. Giving
    it the words left the search box with no text at all, and nothing on screen could
    be found by asking for it.
    """
    field = _element(Box(1026, 61, 239, 33), ElementKind.text_field, "", 0.8, ElementSource.yolo)
    glyph = _element(Box(1037, 70, 14, 15), ElementKind.icon, "", 0.7, ElementSource.yolo)
    label = _element(
        Box(1040, 68, 90, 16), ElementKind.text, "Search mail", 0.95, ElementSource.ocr
    )

    merged = merge_elements([field, glyph], [label])
    hit = ElementIndex(merged).find_text("Search mail")[0]

    assert hit.kind is ElementKind.text_field and hit.box == field.box


def test_a_word_on_a_large_surface_stays_pointable_where_it_is_printed() -> None:
    """The bulk-status menu on the live sandbox, which is where this came from.

    Its items were not detected as boxes at all, so every item's text fell into the
    wide table row drawn underneath it. With only the row to offer, clicking "Paused"
    meant clicking the middle of a record - which sets no status and is not what
    anybody asked for. The word is kept where it is printed as well.
    """
    row = _element(Box(0, 268, 1222, 45), ElementKind.row, "", 0.80, ElementSource.yolo)
    item = _element(Box(320, 280, 62, 18), ElementKind.text, "Paused", 0.95, ElementSource.ocr)

    merged = merge_elements([row], [item])
    hit = ElementIndex(merged).find_text("Paused")[0]

    assert hit.box == item.box, "the click lands on the word, not the middle of the row"
    assert any(e.box == row.box and "Paused" in e.text for e in merged), (
        "and the row still says what is printed on it, so a search for the row finds it"
    )


def test_a_card_whose_only_clickable_part_is_its_title_can_still_be_opened() -> None:
    """A board card, which is where this came from.

    Its title opens the ticket and the rest of the card does nothing at all, so an
    agent that could only click the card's centre could not open one. Several lines is
    what makes something a container rather than a labelled control, whatever its size
    relative to any one line.
    """
    card = _element(Box(33, 186, 258, 75), ElementKind.button, "", 0.85, ElementSource.yolo)
    title = _element(
        Box(41, 192, 200, 20), ElementKind.text, "Set up nightly export", 0.95, ElementSource.ocr
    )
    who = _element(Box(41, 222, 90, 18), ElementKind.text, "Mira Solano", 0.95, ElementSource.ocr)

    merged = merge_elements([card], [title, who])
    hit = ElementIndex(merged).find_text("Set up nightly export")[0]

    assert hit.box == title.box, "the click lands on the title, which is what opens it"


def test_a_button_is_not_split_from_its_own_label() -> None:
    """The rule is about surfaces far larger than the word, not about every container:
    a button is a few times the area of its label and stays one element."""
    button = _element(Box(8, 64, 182, 36), ElementKind.button, "", 0.85, ElementSource.yolo)
    label = _element(Box(60, 72, 78, 18), ElementKind.text, "Compose", 0.95, ElementSource.ocr)

    merged = merge_elements([button], [label])

    assert len(merged) == 1
    assert merged[0].box == button.box and merged[0].text == "Compose"


def test_a_label_joins_the_innermost_control_that_contains_it() -> None:
    """An overlay's text must not be claimed by whatever it was drawn on top of.

    A label menu opened over a message list puts "Travel" inside both the menu item
    and the row behind it. The word is the menu item's; a merge that gave it to the
    row left nothing on screen to click, and the agent could not apply a label that
    was plainly visible.
    """
    row = _element(Box(240, 113, 970, 79), ElementKind.button, "", 0.80, ElementSource.yolo)
    item = _element(Box(375, 131, 177, 31), ElementKind.button, "", 0.80, ElementSource.yolo)
    label = _element(Box(390, 138, 60, 16), ElementKind.text, "Travel", 0.95, ElementSource.ocr)

    merged = merge_elements([row, item], [label])

    by_box = {e.box: e.text for e in merged}
    assert by_box[item.box] == "Travel", "the menu item keeps its own label"
    assert by_box[row.box] == "", "the row behind it does not take the word"


def test_merge_keeps_two_genuinely_separate_elements() -> None:
    search = _element(Box(360, 132, 120, 44), ElementKind.button, "Search", 0.8)
    new = _element(Box(500, 132, 150, 44), ElementKind.button, "New Invoice", 0.8)
    assert search.box.iou(new.box) == 0.0
    merged = merge_elements([search, new])
    assert [e.text for e in merged] == ["Search", "New Invoice"]


def test_merge_keeps_a_checkbox_that_sits_inside_a_row() -> None:
    """Containment alone must not swallow a second real click target."""
    row = _element(Box(0, 300, 400, 40), ElementKind.row, "Keep me signed in", 0.8)
    checkbox = _element(Box(56, 312, 22, 22), ElementKind.checkbox, "", 0.9)
    merged = merge_elements([row, checkbox])
    assert {e.kind for e in merged} == {ElementKind.row, ElementKind.checkbox}


def test_merge_prefers_higher_confidence_at_equal_specificity() -> None:
    weak = _element(Box(40, 132, 300, 44), ElementKind.button, "Sign in", 0.55, ElementSource.yolo)
    strong = _element(Box(41, 133, 300, 44), ElementKind.button, "Sign In", 0.93, ElementSource.dom)
    merged = merge_elements([weak], [strong])
    assert len(merged) == 1
    assert merged[0].box == strong.box
    assert merged[0].text == "Sign In", "ground truth is trusted over YOLO for the text"


def test_merge_passes_a_lone_element_through_with_its_own_source() -> None:
    lone = _element(Box(40, 432, 190, 28), ElementKind.text, "Download report", 0.9)
    lone = Element(
        box=lone.box, kind=lone.kind, text=lone.text, confidence=0.9, source=ElementSource.ocr
    )
    merged = merge_elements([lone])
    assert len(merged) == 1
    assert merged[0].source is ElementSource.ocr, "nothing was fused, so nothing is 'merged'"
    assert merged[0].stable_id == stable_id(lone)


def test_merge_of_nothing_is_an_empty_list() -> None:
    assert merge_elements() == []
    assert merge_elements([], []) == []


def test_merge_returns_reading_order() -> None:
    late = _element(Box(40, 496, 190, 48), ElementKind.button, "Send Reminders")
    early = _element(Box(40, 32, 300, 40), ElementKind.text, "Open Invoices")
    assert [e.text for e in merge_elements([late, early])] == ["Open Invoices", "Send Reminders"]


def test_merge_of_the_whole_planted_page_changes_nothing() -> None:
    """A page whose elements are already distinct must survive a merge untouched."""
    planted = _planted("invoices@1x")
    merged = merge_elements(planted)
    assert len(merged) == len(planted)
    assert [e.text for e in merged] == [e.text for e in reading_order(planted)]


# --------------------------------------------------------------------------------------
# crop
# --------------------------------------------------------------------------------------


def test_clamp_box_trims_each_edge() -> None:
    assert cropping.clamp_box(Box(-20, -30, 100, 100), 800, 600) == Box(0, 0, 80, 70)
    assert cropping.clamp_box(Box(700, 550, 300, 200), 800, 600) == Box(700, 550, 100, 50)
    assert cropping.clamp_box(Box(10, 10, 20, 20), 800, 600) == Box(10, 10, 20, 20)
    assert cropping.clamp_box(Box(900, 700, 50, 50), 800, 600).area == 0


def test_crop_cuts_exactly_the_requested_pixels() -> None:
    shot = _load("invoices@1x")
    region = Box(40, 132, 300, 44)
    result = cropping.crop(shot, region)

    assert result.box == region
    assert not result.clamped
    assert (result.screenshot.width, result.screenshot.height) == (300, 44)
    expected = shot.to_array(logical=False)[132:176, 40:340]
    assert np.array_equal(result.screenshot.to_array(logical=False), expected)


def test_crop_translates_coordinates_both_ways() -> None:
    shot = _load("invoices@1x")
    result = cropping.crop(shot, Box(40, 132, 300, 44))

    # A control found 10px into the crop is 10px past the crop's origin on the screen.
    local = _element(Box(10, 4, 60, 20), ElementKind.button, "Go")
    back = result.to_parent_elements([local])[0]
    assert back.box == Box(50, 136, 60, 20)
    assert back.stable_id == local.stable_id, "translating does not make it a new element"

    assert result.to_parent_point(Point(0, 0)) == Point(40, 132)
    assert result.to_local_box(back.box) == local.box
    assert result.to_local_point(result.to_parent_point(Point(7, 9))) == Point(7, 9)
    assert result.offset == Point(40, 132)


def test_crop_clamps_at_an_edge_and_says_so() -> None:
    shot = _load("invoices@1x")
    result = cropping.crop(shot, Box(700, 550, 300, 200))

    assert result.box == Box(700, 550, 100, 50)
    assert result.requested == Box(700, 550, 300, 200)
    assert result.clamped
    assert (result.screenshot.width, result.screenshot.height) == (100, 50)
    # The offset is still the CLAMPED origin, so translation stays correct at an edge.
    assert result.to_parent_point(Point(0, 0)) == Point(700, 550)
    expected = shot.to_array(logical=False)[550:600, 700:800]
    assert np.array_equal(result.screenshot.to_array(logical=False), expected)


def test_crop_clamps_a_negative_origin() -> None:
    shot = _load("invoices@1x")
    result = cropping.crop(shot, Box(-20, -30, 100, 100))
    assert result.box == Box(0, 0, 80, 70)
    assert result.to_parent_box(Box(5, 5, 10, 10)) == Box(5, 5, 10, 10)


def test_crop_keeps_scale_so_a_retina_crop_stays_logical() -> None:
    shot = _load("invoices@2x")
    region = Box(40, 132, 300, 44)
    result = cropping.crop(shot, region)

    assert result.screenshot.scale == 2.0
    assert (result.screenshot.width, result.screenshot.height) == (300, 44)
    assert shots.physical_size(result.screenshot) == (600, 88)
    expected = shot.to_array(logical=False)[264:352, 80:680]
    assert np.array_equal(result.screenshot.to_array(logical=False), expected)


def test_crop_with_a_margin_grows_then_clamps() -> None:
    shot = _load("invoices@1x")
    assert cropping.crop(shot, Box(100, 100, 50, 50), margin=10).box == Box(90, 90, 70, 70)
    assert cropping.crop(shot, Box(0, 0, 50, 50), margin=10).box == Box(0, 0, 60, 60)


def test_crop_around_covers_every_element_given() -> None:
    shot = _load("invoices@1x")
    planted = _planted("invoices@1x")
    buttons = [e for e in planted if e.kind is ElementKind.button and e.box.y < 200]
    result = cropping.crop_around(shot, buttons, margin=4)
    for button in buttons:
        assert result.box.contains(button.box.center)


def test_crop_outside_the_screenshot_is_an_error() -> None:
    shot = _load("invoices@1x")
    with pytest.raises(PerceptionError, match="does not overlap"):
        cropping.crop(shot, Box(2000, 2000, 50, 50))
    with pytest.raises(PerceptionError):
        cropping.crop_around(shot, [])


def test_grow_box_never_goes_negative() -> None:
    assert cropping.grow_box(Box(10, 10, 20, 20), 5) == Box(5, 5, 30, 30)
    assert cropping.grow_box(Box(10, 10, 20, 20), -20).area == 0


# --------------------------------------------------------------------------------------
# screenshot conversions
# --------------------------------------------------------------------------------------


def test_load_reports_logical_size_derived_from_scale() -> None:
    record = _shot_record("invoices@2x")
    derived = shots.load_screenshot(SHOTS / record["png"], scale=2.0)
    assert (derived.width, derived.height) == (800, 600)
    assert shots.physical_size(derived) == (1600, 1200)
    assert shots.decode_png(derived.png).shape[:2] == (1200, 1600)


def test_to_array_is_logical_by_default_and_physical_on_request() -> None:
    shot = _load("invoices@2x")
    assert shots.to_array(shot).shape[:2] == (600, 800)
    assert shots.to_array(shot, logical=False).shape[:2] == (1200, 1600)


def test_array_roundtrip_preserves_scale_and_logical_size() -> None:
    shot = _load("invoices@2x")
    physical = shot.to_array(logical=False)
    rebuilt = shots.from_array(physical, scale=shot.scale)
    assert (rebuilt.width, rebuilt.height) == (shot.width, shot.height)
    assert rebuilt.scale == shot.scale
    assert np.array_equal(rebuilt.to_array(logical=False), physical)


def test_png_roundtrip_preserves_scale_and_logical_size(tmp_path: Path) -> None:
    shot = _load("invoices@2x")
    written = shots.save_screenshot(shot, tmp_path / "nested" / "shot.png")
    reloaded = shots.load_screenshot(written, scale=shot.scale)
    assert (reloaded.width, reloaded.height, reloaded.scale) == (
        shot.width,
        shot.height,
        shot.scale,
    )
    assert reloaded.png == shot.png


def test_from_png_can_be_told_the_true_logical_size() -> None:
    shot = _load("invoices@1x")
    odd = shots.from_png(shot.png, scale=1.5, width=533, height=400)
    assert (odd.width, odd.height, odd.scale) == (533, 400, 1.5)


def test_rescale_keeps_the_logical_frame_and_changes_the_pixels() -> None:
    shot = _load("invoices@2x")
    flat = shots.rescale(shot, 1.0)
    assert (flat.width, flat.height) == (shot.width, shot.height)
    assert flat.scale == 1.0
    assert shots.physical_size(flat) == (800, 600)
    assert shots.decode_png(flat.png).shape[:2] == (600, 800)
    assert flat.captured_at == shot.captured_at
    assert shots.rescale(flat, 1.0) is flat


def test_resize_changes_the_logical_frame_and_keeps_the_density() -> None:
    shot = _load("invoices@2x")
    half = shots.resize(shot, width=400)
    assert (half.width, half.height) == (400, 300), "aspect ratio is preserved"
    assert half.scale == 2.0
    assert shots.physical_size(half) == (800, 600)

    forced = shots.resize(shot, width=200, height=200)
    assert (forced.width, forced.height) == (200, 200)


def test_conversions_reject_nonsense() -> None:
    shot = _load("invoices@1x")
    with pytest.raises(PerceptionError, match="positive"):
        shots.from_png(shot.png, scale=0)
    with pytest.raises(PerceptionError, match="positive"):
        shots.rescale(shot, -1.0)
    with pytest.raises(PerceptionError, match="width, height"):
        shots.resize(shot)
    with pytest.raises(PerceptionError, match="could not be decoded"):
        shots.decode_png(b"this is not a png")
    with pytest.raises(PerceptionError, match="could not be read"):
        shots.load_screenshot(SHOTS / "no-such-fixture.png")
    with pytest.raises(PerceptionError, match="cannot encode"):
        shots.encode_png(np.zeros((4, 4, 2), dtype=np.uint8))


def test_a_screenshot_built_from_an_array_decodes_back_to_it() -> None:
    array = np.zeros((40, 60, 3), dtype=np.uint8)
    array[10:20, 15:25] = (255, 0, 0)
    shot = shots.from_array(array, scale=2.0, captured_at=datetime(2026, 9, 19, tzinfo=UTC))
    assert (shot.width, shot.height) == (30, 20)
    assert shot.captured_at == datetime(2026, 9, 19, tzinfo=UTC)
    assert np.array_equal(shots.decode_png(shot.png), array)
    assert shots.logical_size(shot) == (30, 20)
