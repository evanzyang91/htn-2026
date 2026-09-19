"""The real OCR reader against the real committed screenshots.

Unlike the rest of the suite these tests do load a model - the bundled PP-OCRv4 ONNX
files that ship inside ``rapidocr-onnxruntime`` - but they still touch no network and no
API. That is the point: OCR has to be exercised on actual pixels, because the failures
that matter (text not recovered, boxes off by ``scale``) are invisible against a fake.

The fixtures and their expectations come from ``tests/fixtures/shots/generate.py``;
``expect_text`` lists strings OCR must recover, each with the ground-truth box of the
element that prints it, so a wrong coordinate fails just as loudly as wrong text.

The engine is loaded once for the whole module and every screenshot is read once.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from skillweaver import contracts
from skillweaver.contracts import Box, Element, ElementKind, ElementSource, Screenshot
from skillweaver.errors import PerceptionError
from skillweaver.perception import crop as cropping
from skillweaver.perception import screenshot as shots
from skillweaver.perception.elements import ElementIndex, normalize_text, overlap_ratio, stable_id
from skillweaver.perception.ocr import (
    DEFAULT_MIN_CONFIDENCE,
    CachingTextReader,
    PerceptionCounters,
    RapidOcrReader,
    content_key,
)

SHOTS = Path(__file__).resolve().parents[1] / "fixtures" / "shots"
RECORDS: dict[str, dict] = {
    record["name"]: record
    for record in json.loads((SHOTS / "expectations.json").read_text())["shots"]
}
NAMES = sorted(RECORDS)


def _load(name: str) -> Screenshot:
    record = RECORDS[name]
    return shots.load_screenshot(
        SHOTS / record["png"],
        scale=record["scale"],
        width=record["width"],
        height=record["height"],
    )


@pytest.fixture(scope="module")
def reader() -> RapidOcrReader:
    """One engine for the module: loading it is the expensive part, reading is not."""
    return RapidOcrReader()


@pytest.fixture(scope="module")
def read(reader: RapidOcrReader) -> dict[str, list[Element]]:
    """Every fixture read once, keyed by fixture name."""
    return {name: reader.read(_load(name)) for name in NAMES}


# --------------------------------------------------------------------------------------
# Recovering the committed expectations
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_ocr_recovers_every_expected_string_at_the_right_place(
    name: str, read: dict[str, list[Element]]
) -> None:
    """Each expected string is findable AND its box lands on the element that prints it.

    Lookup goes through :class:`ElementIndex` because that is how the pipeline asks -
    and because real OCR runs words together ("sendreminders"), which the index's fuzzy
    matching is there to absorb. The coordinate half of the assertion is what catches a
    Retina screenshot read without dividing by ``scale``.
    """
    index = ElementIndex(read[name])
    for expectation in RECORDS[name]["expect_text"]:
        wanted = expectation["text"]
        truth = Box(*expectation["near"])
        hits = index.find_text(wanted)
        assert hits, f"{name}: OCR could not find {wanted!r} in {[e.text for e in read[name]]}"
        assert overlap_ratio(hits[0].box, truth) >= 0.5, (
            f"{name}: {wanted!r} came back at {hits[0].box} but is drawn at {truth}"
        )


@pytest.mark.parametrize("name", NAMES)
def test_ocr_reads_the_clean_strings_verbatim(name: str, read: dict[str, list[Element]]) -> None:
    """Strings the recognizer got exactly right at generation time it must still get right.

    This is the half of the expectations that fuzzy matching cannot paper over, so it
    pins down raw OCR quality rather than the index's tolerance.
    """
    haystack = " | ".join(e.text.casefold() for e in read[name])
    exact = [e["text"] for e in RECORDS[name]["expect_text"] if e["verbatim"]]
    assert exact, "a fixture with no verbatim expectations is not testing OCR quality"
    for wanted in exact:
        assert wanted.casefold() in haystack, f"{name}: {wanted!r} not in {haystack}"


@pytest.mark.parametrize("name", NAMES)
def test_ocr_elements_honor_the_text_reader_contract(
    name: str, read: dict[str, list[Element]]
) -> None:
    found = read[name]
    record = RECORDS[name]
    assert found, f"{name}: the fixture has text, so OCR returning nothing is a failure"
    for element in found:
        assert element.kind is ElementKind.text
        assert element.source is ElementSource.ocr
        assert element.text.strip() == element.text and element.text
        assert DEFAULT_MIN_CONFIDENCE <= element.confidence <= 1.0
        assert element.stable_id == stable_id(element)
        # Boxes are LOGICAL pixels, so they fit the logical frame - not the PNG.
        assert 0 <= element.box.x and element.box.x + element.box.w <= record["width"]
        assert 0 <= element.box.y and element.box.y + element.box.h <= record["height"]
        assert element.box.area > 0


@pytest.mark.parametrize("name", NAMES)
def test_ocr_returns_reading_order(name: str, read: dict[str, list[Element]]) -> None:
    found = read[name]
    tops = [e.box.y for e in found]
    assert tops == sorted(tops) or found[0].box.y == min(tops)
    first = normalize_text(found[0].text)
    assert first in {"open invoices", "sign in to ledger"}


# --------------------------------------------------------------------------------------
# The scale conversion, which is the expensive mistake to get wrong
# --------------------------------------------------------------------------------------


def test_a_retina_screenshot_reads_at_the_same_logical_coordinates_as_a_1x_one(
    read: dict[str, list[Element]],
) -> None:
    """The same page at 1x and 2x must produce the same LOGICAL boxes.

    The 2x PNG is 1600x1200 and the 1x PNG is 800x600, so any code that forgot to divide
    by ``Screenshot.scale`` doubles every coordinate here and this fails by ~300 pixels.
    """
    one = ElementIndex(read["invoices@1x"]).find_text("Open Invoices")[0]
    two = ElementIndex(read["invoices@2x"]).find_text("Open Invoices")[0]
    assert abs(one.box.center.x - two.box.center.x) <= 6
    assert abs(one.box.center.y - two.box.center.y) <= 6
    assert two.box.x + two.box.w <= 800, "a 2x box leaked physical pixels"


def test_rescaling_a_retina_screenshot_does_not_move_the_text(reader: RapidOcrReader) -> None:
    """``rescale`` keeps the logical frame, so boxes found before and after agree."""
    retina = _load("invoices@2x")
    flattened = shots.rescale(retina, 1.0)
    assert (flattened.width, flattened.height) == (retina.width, retina.height)

    before = ElementIndex(reader.read(retina)).find_text("Download report")[0]
    after = ElementIndex(reader.read(flattened)).find_text("Download report")[0]
    assert overlap_ratio(before.box, after.box) >= 0.6


# --------------------------------------------------------------------------------------
# OCR inside a crop, which is what re-detection in a region actually looks like
# --------------------------------------------------------------------------------------


def test_reading_a_crop_and_translating_back_lands_on_the_original_element(
    reader: RapidOcrReader,
) -> None:
    shot = _load("invoices@1x")
    truth = Box(
        *next(
            e["near"]
            for e in RECORDS["invoices@1x"]["expect_text"]
            if e["text"] == "Send Reminders"
        )
    )
    region = cropping.crop(shot, truth, margin=6)

    local = ElementIndex(reader.read(region.screenshot)).find_text("Send Reminders")
    assert local, "the crop contains the button, so its label must be readable"
    # In the crop's own frame the box is near the origin, nowhere near the real y.
    assert local[0].box.y < 40

    on_screen = region.to_parent_elements(local)[0]
    assert overlap_ratio(on_screen.box, truth) >= 0.5, (
        f"translated to {on_screen.box}, drawn at {truth}"
    )


def test_reading_a_retina_crop_also_lands_correctly(reader: RapidOcrReader) -> None:
    """Crop plus scale together: two conversions that must compose, not cancel."""
    shot = _load("invoices@2x")
    truth = Box(
        *next(
            e["near"] for e in RECORDS["invoices@2x"]["expect_text"] if e["text"] == "New Invoice"
        )
    )
    region = cropping.crop(shot, truth, margin=8)
    assert region.screenshot.scale == 2.0

    local = ElementIndex(reader.read(region.screenshot)).find_text("New Invoice")
    assert local
    on_screen = region.to_parent_elements(local)[0]
    assert overlap_ratio(on_screen.box, truth) >= 0.5


# --------------------------------------------------------------------------------------
# Loading behavior and failure behavior
# --------------------------------------------------------------------------------------


def test_the_reader_satisfies_the_text_reader_protocol() -> None:
    assert isinstance(RapidOcrReader(), contracts.TextReader)


def test_importing_the_module_does_not_load_onnx_runtime() -> None:
    """Importing perception must stay cheap; the model is a first-``read`` cost.

    Checked in a subprocess because this process has certainly loaded the engine by now.
    """
    probe = (
        "import sys, skillweaver.perception.ocr as m;"
        "assert 'onnxruntime' not in sys.modules, 'onnxruntime imported eagerly';"
        "assert 'rapidocr_onnxruntime' not in sys.modules, 'rapidocr imported eagerly';"
        "r = m.RapidOcrReader();"
        "assert r.loaded is False;"
        "print('cheap')"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "cheap" in done.stdout


def test_a_missing_engine_raises_perception_error_instead_of_reading_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Degrading to an empty list would let an agent click blindly, so it must raise."""
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    with pytest.raises(PerceptionError, match="not installed"):
        RapidOcrReader().read(_load("login@1x"))


def test_an_unloadable_model_raises_perception_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("model file is missing")

    stub = types.ModuleType("rapidocr_onnxruntime")
    stub.RapidOCR = explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", stub)
    with pytest.raises(PerceptionError, match="could not be loaded"):
        RapidOcrReader().read(_load("login@1x"))


def test_failed_inference_raises_perception_error() -> None:
    def explode(_image: object) -> object:
        raise RuntimeError("onnxruntime segfault-adjacent nonsense")

    with pytest.raises(PerceptionError, match="inference failed"):
        RapidOcrReader(engine=explode).read(_load("login@1x"))


def test_a_blank_screen_reads_as_an_empty_list() -> None:
    """No text is an empty list; that is different from OCR being broken, which raises."""
    assert RapidOcrReader(engine=lambda _image: (None, None)).read(_load("login@1x")) == []
    assert RapidOcrReader(engine=lambda _image: ([], [])).read(_load("login@1x")) == []


def test_unreadable_rows_are_skipped_rather_than_crashing() -> None:
    engine = lambda _image: (  # noqa: E731
        [
            "nonsense",
            [[[0, 0], [10, 0], [10, 10], [0, 10]]],
            [[[0, 0], [10, 0], [10, 10], [0, 10]], "   ", 0.99],
            [[[0, 0], [10, 0], [10, 10], [0, 10]], "ok", "not-a-number"],
            [[[0, 0], [20, 0], [20, 12], [0, 12]], "Keep", 0.91],
        ],
        None,
    )
    found = RapidOcrReader(engine=engine).read(_load("login@1x"))
    assert [e.text for e in found] == ["Keep"]


def test_low_confidence_recognitions_are_dropped() -> None:
    engine = lambda _image: (  # noqa: E731
        [
            [[[0, 0], [20, 0], [20, 12], [0, 12]], "sure", 0.95],
            [[[0, 20], [20, 20], [20, 32], [0, 32]], "guess", 0.21],
        ],
        None,
    )
    assert [e.text for e in RapidOcrReader(engine=engine).read(_load("login@1x"))] == ["sure"]
    lenient = RapidOcrReader(engine=engine, min_confidence=0.1)
    assert [e.text for e in lenient.read(_load("login@1x"))] == ["sure", "guess"]


def test_boxes_from_a_retina_capture_are_halved() -> None:
    """The one line of arithmetic this module exists to get right, in isolation."""
    engine = lambda _image: (  # noqa: E731
        [[[[80, 264], [680, 264], [680, 352], [80, 352]], "Search invoices", 0.9]],
        None,
    )
    found = RapidOcrReader(engine=engine).read(_load("invoices@2x"))
    assert len(found) == 1
    assert found[0].box == Box(40, 132, 300, 44)


def test_a_box_running_off_the_edge_is_clamped_into_the_logical_frame() -> None:
    engine = lambda _image: (  # noqa: E731
        [[[[-10, -10], [900, -10], [900, 700], [-10, 700]], "overflow", 0.9]],
        None,
    )
    found = RapidOcrReader(engine=engine).read(_load("invoices@1x"))
    assert found[0].box == Box(0, 0, 800, 600)


def test_an_undecodable_screenshot_raises_perception_error() -> None:
    broken = Screenshot(
        png=b"not a png",
        width=10,
        height=10,
        scale=1.0,
        captured_at=_load("login@1x").captured_at,
    )
    with pytest.raises(PerceptionError):
        RapidOcrReader(engine=lambda _image: ([], None)).read(broken)


def test_the_engine_is_built_once_and_only_when_needed() -> None:
    calls: list[object] = []

    def engine(image: object) -> object:
        calls.append(image)
        return ([], None)

    reader = RapidOcrReader(engine=engine)
    assert reader.loaded, "an injected engine needs no loading"
    reader.read(_load("login@1x"))
    reader.read(_load("login@1x"))
    assert len(calls) == 2


# --------------------------------------------------------------------------------------
# Not reading the same pixels twice
# --------------------------------------------------------------------------------------


def test_the_caching_reader_satisfies_the_text_reader_protocol() -> None:
    assert isinstance(CachingTextReader(RapidOcrReader()), contracts.TextReader)


def test_the_same_frame_is_read_once_and_then_remembered(reader: RapidOcrReader) -> None:
    """The saving this cache exists for, against the real engine and real pixels."""
    counters = PerceptionCounters()
    caching = CachingTextReader(reader, counters=counters)
    shot = _load("invoices@1x")

    first = caching.read(shot)
    repeats = [caching.read(_load("invoices@1x")) for _ in range(3)]

    assert first, "the fixture has text; a cache test on an empty read proves nothing"
    assert all(again == first for again in repeats), "a hit must be the read it replaces"
    assert (counters.ocr_reads, counters.ocr_hits) == (1, 3)


@pytest.mark.parametrize("other", ["login@1x", "invoices@2x"])
def test_a_changed_page_is_never_served_a_previous_read(
    reader: RapidOcrReader, read: dict[str, list[Element]], other: str
) -> None:
    """The bar the whole optimization is held to, with the real engine.

    A stale read is far worse than a slow one: a skill cannot tell one from the other,
    it just clicks. ``invoices@2x`` is the sharper case - the SAME page as the 1x
    fixture, whose elements live at the same logical coordinates but whose bytes and
    declared scale differ - so a cache keyed on anything looser than the pixels would
    hand the 2x frame the 1x read and be none the wiser.
    """
    counters = PerceptionCounters()
    caching = CachingTextReader(reader, counters=counters)

    caching.read(_load("invoices@1x"))
    served = caching.read(_load(other))

    assert counters.ocr_reads == 2, f"{other} must be read, not served from the cache"
    assert counters.ocr_hits == 0
    assert served == read[other], "the changed page gets its own true read"


def test_the_same_bytes_at_a_different_scale_are_a_different_frame() -> None:
    """Geometry is part of the identity: the same PNG at 2x has boxes at half the
    position, and serving one read for the other is the doubled-coordinate bug."""
    engine = lambda _image: (  # noqa: E731
        [[[[80, 264], [680, 264], [680, 352], [80, 352]], "Search invoices", 0.9]],
        None,
    )
    counters = PerceptionCounters()
    caching = CachingTextReader(RapidOcrReader(engine=engine), counters=counters)
    png = _load("invoices@2x").png

    at_1x = caching.read(shots.from_png(png, scale=1.0))
    at_2x = caching.read(shots.from_png(png, scale=2.0))

    assert counters.ocr_reads == 2 and counters.ocr_hits == 0
    assert at_1x[0].box == Box(80, 264, 600, 88)
    assert at_2x[0].box == Box(40, 132, 300, 44)


def test_a_cached_read_cannot_be_corrupted_by_whoever_was_served_it() -> None:
    """Every hit is a fresh list, so a caller that sorts or trims it harms nobody."""
    caching = CachingTextReader(RapidOcrReader(engine=_one_word_engine("Keep")))
    shot = _load("login@1x")

    served = caching.read(shot)
    served.clear()

    assert [e.text for e in caching.read(shot)] == ["Keep"]


def test_a_failed_read_is_not_remembered_as_an_answer() -> None:
    """A transient engine fault must not become a permanent blind spot."""
    attempts: list[int] = []

    def flaky(_image: object) -> object:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("onnxruntime had a bad day")
        return ([[[[0, 0], [20, 0], [20, 12], [0, 12]], "Recovered", 0.9]], None)

    caching = CachingTextReader(RapidOcrReader(engine=flaky))
    with pytest.raises(PerceptionError):
        caching.read(_load("login@1x"))

    assert [e.text for e in caching.read(_load("login@1x"))] == ["Recovered"]


def test_the_cache_is_bounded_and_evicts_the_least_recently_used_frame() -> None:
    reads: list[int] = []

    def engine(_image: object) -> object:
        reads.append(1)
        return ([], None)

    caching = CachingTextReader(RapidOcrReader(engine=engine), capacity=2)
    a, b, c = (shots.resize(_load("login@1x"), width=w) for w in (640, 320, 160))

    caching.read(a)
    caching.read(b)
    caching.read(a)  # a is now the most recently used, so b is next out
    caching.read(c)

    assert len(reads) == 3
    caching.read(a)
    assert len(reads) == 3, "a was kept"
    caching.read(b)
    assert len(reads) == 4, "b was evicted"


def test_capacity_zero_reads_every_frame_and_still_counts() -> None:
    """How a caller turns the optimization off to measure against it."""
    counters = PerceptionCounters()
    caching = CachingTextReader(
        RapidOcrReader(engine=_one_word_engine("Hi")), capacity=0, counters=counters
    )
    for _ in range(3):
        caching.read(_load("login@1x"))
    assert (counters.ocr_reads, counters.ocr_hits) == (3, 0)


def test_clearing_forgets_the_frames_but_keeps_the_tally() -> None:
    counters = PerceptionCounters()
    caching = CachingTextReader(RapidOcrReader(engine=_one_word_engine("Hi")), counters=counters)
    caching.read(_load("login@1x"))
    caching.read(_load("login@1x"))
    caching.clear()
    caching.read(_load("login@1x"))
    assert (counters.ocr_reads, counters.ocr_hits) == (2, 1)


def test_content_key_is_the_pixels_and_the_geometry() -> None:
    shot = _load("invoices@1x")
    same = shots.from_png(shot.png, scale=1.0, width=shot.width, height=shot.height)
    assert content_key(shot) == content_key(same)
    assert content_key(shot) != content_key(_load("login@1x"))
    assert content_key(shot) != content_key(shots.rescale(shot, 2.0))


def _one_word_engine(word: str):
    """A stub RapidOCR that always recognizes ``word`` once, wherever it is asked."""
    return lambda _image: ([[[[0, 0], [20, 0], [20, 12], [0, 12]], word, 0.9]], None)
