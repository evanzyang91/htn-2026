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

import numpy as np
import pytest

from skillweaver import contracts
from skillweaver.contracts import Box, Element, ElementKind, ElementSource, Screenshot
from skillweaver.errors import PerceptionError
from skillweaver.perception import crop as cropping
from skillweaver.perception import screenshot as shots
from skillweaver.perception.elements import ElementIndex, normalize_text, overlap_ratio, stable_id
from skillweaver.perception.ocr import DEFAULT_MIN_CONFIDENCE, RapidOcrReader

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
# The line cache: the same pixels are never recognized twice
# --------------------------------------------------------------------------------------


class _StagedEngine:
    """A RapidOCR-shaped double exposing the three stages the fast path uses.

    Each "line" is a distinct little array, so the reader's content digest separates
    them the way it separates real crops. ``recognized`` counts the crops that reached
    the recognizer, which is the quantity the cache exists to reduce.
    """

    use_cls = True

    def __init__(self, lines: dict[tuple[int, int], str]) -> None:
        self._lines = lines
        self.recognized: list[int] = []
        self.detections = 0

    def text_det(self, image: object) -> tuple[list[object], float]:
        self.detections += 1
        boxes = [
            np.array([[x, y], [x + 40, y], [x + 40, y + 10], [x, y + 10]], dtype=np.float32)
            for (x, y) in self._lines
        ]
        return boxes, 0.0

    def get_crop_img_list(self, image: object, boxes: list[object]) -> list[object]:
        # One crop per line, its pixels derived from the text so that the same text at
        # the same place is byte-identical between reads and different text is not.
        return [
            np.full((10, 40, 3), abs(hash(text)) % 251, dtype=np.uint8)
            for text in self._lines.values()
        ]

    def text_rec(self, crops: list[object], *args: object) -> tuple[list[object], float]:
        self.recognized.append(len(crops))
        wanted = list(self._lines.values())
        answers = []
        for crop in crops:
            shade = int(crop[0, 0, 0])
            answers.append(
                next((t for t in wanted if abs(hash(t)) % 251 == shade), ("", 0.0)) or ""
            )
        return [[text, 0.9] for text in answers], 0.0


def test_the_same_screen_read_twice_recognizes_nothing_the_second_time() -> None:
    """The cache's whole purpose, asserted at the recognizer rather than on a clock.

    Detection still runs on every read - the lines could have moved - but no crop whose
    pixels have already been read is sent to the recognizer again.
    """
    engine = _StagedEngine({(10, 10): "Inbox", (10, 30): "Archived", (10, 50): "Sent"})
    reader = RapidOcrReader(engine=engine)
    shot = _load("login@1x")

    first = reader.read(shot)
    second = reader.read(shot)

    assert [e.text for e in first] == [e.text for e in second], "the same screen reads the same"
    assert engine.detections == 2, "detection is not cached: the lines may have moved"
    assert engine.recognized == [3], "three crops recognized once, and never again"
    assert reader.cache_info()["hits"] == 3


def test_only_the_lines_that_changed_are_recognized_again() -> None:
    """The case a real run is always in: one action changes part of one screen."""
    lines = {(10, 10): "Inbox", (10, 30): "Archived", (10, 50): "Sent"}
    engine = _StagedEngine(lines)
    reader = RapidOcrReader(engine=engine)
    shot = _load("login@1x")
    reader.read(shot)

    lines[(10, 50)] = "Drafts"  # one line of three now says something else
    reader.read(shot)

    assert engine.recognized == [3, 1], "only the changed line went back to the recognizer"


def test_a_reader_built_without_a_cache_recognizes_every_line_every_time() -> None:
    """``cache=False`` is what a benchmark measuring a cold read needs."""
    engine = _StagedEngine({(10, 10): "Inbox", (10, 30): "Archived"})
    reader = RapidOcrReader(engine=engine, cache=False)
    shot = _load("login@1x")
    reader.read(shot)
    reader.read(shot)
    assert engine.recognized == [2, 2]


def test_the_staged_path_and_the_plain_path_agree_on_the_real_engine(
    reader: RapidOcrReader,
) -> None:
    """The fast path must not change WHAT is read, only what it costs.

    The shipped reader uses detection plus a cached recognition; a reader handed the
    same engine wrapped so that the stages are hidden falls back to calling it whole.
    Both are run over a real screenshot and must agree on every line and every box.
    """

    class _Opaque:
        """The same engine with its stages hidden, so only ``__call__`` is reachable."""

        use_cls = True

        def __init__(self, inner: object) -> None:
            self._inner = inner

        def __call__(self, image: object, **kwargs: object) -> object:
            return self._inner(image, **kwargs)

    shot = _load("invoices@1x")
    staged = reader.read(shot)
    plain = RapidOcrReader(engine=_Opaque(reader._ensure_engine())).read(shot)  # noqa: SLF001

    assert [(e.text, e.box) for e in staged] == [(e.text, e.box) for e in plain]
