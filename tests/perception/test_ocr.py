"""The real OCR reader against the real committed screenshots.

Unlike the rest of the suite these tests do load a model - the bundled PP-OCRv4 ONNX
files that ship inside ``rapidocr-onnxruntime`` - but they still touch no network and no
API. That is the point: OCR has to be exercised on actual pixels, because the failures
that matter (text not recovered, boxes off by ``scale``) are invisible against a fake.

The fixtures and their expectations come from ``tests/fixtures/shots/generate.py``;
``expect_text`` lists strings OCR must recover, each with the ground-truth box of the
element that prints it, so a wrong coordinate fails just as loudly as wrong text.

The engine is loaded once for the whole module and every screenshot is read once -
and, by default, in a child process, so these tests exercise the worker pipe on every
single read as a side effect of testing anything else.

The last section is about ending a read that has gone wrong. What is tested there is
the ENFORCEMENT: a substituted child that does not answer is abandoned within its
budget and reported as a perception failure. Reproducing ONNX Runtime's own spin on
demand is not practical, so it is not attempted - see ``test_a_read_that_blocks...``
for how close the substitute gets, which is a child genuinely executing no Python.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from collections.abc import Iterator
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
    DEFAULT_OCR_THREADS,
    DEFAULT_READ_TIMEOUT_S,
    DEFAULT_REC_BATCH,
    CachingTextReader,
    OcrWorker,
    PerceptionCounters,
    PerceptionTimeout,
    RapidOcrReader,
    _worker_argv,
    _worker_env,
    content_key,
    ocr_threads,
    read_timeout_s,
    rec_batch,
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
def reader() -> Iterator[RapidOcrReader]:
    """One engine for the module: loading it is the expensive part, reading is not.

    Isolated, like every shipped reader, so the whole file reads through the worker.
    Closed at the end because an unclosed one leaves a child process behind, and a
    test suite that leaks engines is how a laptop ends up at 100% CPU.
    """
    reader = RapidOcrReader()
    yield reader
    reader.close()


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
    """Degrading to an empty list would let an agent click blindly, so it must raise.

    ``isolate=False`` because the message being pinned down is
    :func:`build_engine`'s, and a monkeypatch of this interpreter's ``sys.modules``
    says nothing about what a child process can import.
    """
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    with pytest.raises(PerceptionError, match="not installed"):
        RapidOcrReader(isolate=False).read(_load("login@1x"))


def test_an_unloadable_model_raises_perception_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("model file is missing")

    stub = types.ModuleType("rapidocr_onnxruntime")
    stub.RapidOCR = explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", stub)
    with pytest.raises(PerceptionError, match="could not be loaded"):
        RapidOcrReader(isolate=False).read(_load("login@1x"))


def test_a_worker_that_cannot_import_the_engine_reports_it_in_the_parent(
    tmp_path: Path,
) -> None:
    """The child's failure has to come back, not stay in the child.

    Made real rather than mocked: the worker is given a ``PYTHONPATH`` whose first
    entry shadows ``rapidocr_onnxruntime`` with a module that refuses to import, so
    the child genuinely takes :func:`build_engine`'s ImportError branch. Without the
    reply channel this would be a silent child and a parent waiting on it.
    """
    (tmp_path / "rapidocr_onnxruntime.py").write_text("raise ImportError('no models here')\n")
    env = _worker_env(1)
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{env['PYTHONPATH']}"
    reader = RapidOcrReader(worker=OcrWorker(argv=_worker_argv(), env=env), timeout_s=60)
    try:
        with pytest.raises(PerceptionError, match="not installed"):
            reader.read(_load("login@1x"))
    finally:
        reader.close()


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


# --------------------------------------------------------------------------------------
# Ending a read that has gone wrong
# --------------------------------------------------------------------------------------
#
# The defect: a 12-task evaluation stopped producing output after 49 runs and sat at
# 98.8% CPU for 36 minutes, every thread parked in onnxruntime's WorkerLoop. Nothing
# stopped it, because the skill clock is checked from Python and a native call executes
# none. What follows tests that a read which does not come back is abandoned anyway.
#
# The native spin itself is NOT reproduced here - provoking onnxruntime into it on
# demand is not practical. What is substituted for it is a child process that genuinely
# executes no Python (blocked in ``time.sleep``, which releases the GIL and sits in the
# OS) and a child that genuinely burns a core. Both are ended the same way the real one
# is, by killing the process, which is the only mechanism that works on either.

BLOCKED_CHILD = "import sys, time\nsys.stdin.buffer.read(4)\ntime.sleep(600)\n"
"""A worker that takes the request and then blocks in a native call forever."""

SPINNING_CHILD = "import sys\nsys.stdin.buffer.read(4)\nwhile True:\n    pass\n"
"""A worker that takes the request and then burns a core forever."""

DEAD_CHILD = "import sys\nraise SystemExit(3)\n"
"""A worker that is not there at all by the time the request arrives."""


def _substitute(source: str, timeout_s: float = 1.0) -> RapidOcrReader:
    """A reader whose engine process is ``source`` instead of the real worker."""
    return RapidOcrReader(
        worker=OcrWorker(argv=[sys.executable, "-c", source], env=dict(os.environ)),
        timeout_s=timeout_s,
    )


def _still_running(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], capture_output=True, check=False).returncode == 0


@pytest.mark.parametrize("child", [BLOCKED_CHILD, SPINNING_CHILD], ids=["native-block", "spin"])
def test_a_read_that_blocks_is_abandoned_within_its_budget_and_the_process_killed(
    child: str,
) -> None:
    """The whole point, for a call that no Python-level mechanism could have stopped.

    ``BLOCKED_CHILD`` sits in ``time.sleep``, which holds no GIL and runs no bytecode -
    the same shape as a thread parked in onnxruntime - and ``SPINNING_CHILD`` burns a
    core. A signal handler, a flag or a trace hook would reach neither. A kill reaches
    both, and the assertion that the pid is gone is the half that matters: abandoning
    the WAIT without abandoning the WORK would leave the 98.8% exactly where it was.
    """
    reader = _substitute(child, timeout_s=1.0)
    worker = reader._worker
    assert worker is not None
    worker.start()
    pid = worker.pid
    assert pid is not None

    started = time.perf_counter()
    with pytest.raises(PerceptionTimeout, match="did not return within 1s"):
        reader.read(_load("login@1x"))
    elapsed = time.perf_counter() - started

    assert elapsed < 10.0, f"abandoned after {elapsed:.1f}s, which is not a bound"
    assert elapsed >= 1.0, "it must actually wait out the budget, not give up early"
    assert not worker.alive
    assert not _still_running(pid), "the wait ended but the work did not: cores still burning"


def test_an_abandoned_read_is_a_perception_failure_and_says_which(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The verdict and the wording, which is what keeps a good skill from being demoted."""
    reader = _substitute(BLOCKED_CHILD, timeout_s=0.5)
    with pytest.raises(PerceptionTimeout) as caught:
        reader.read(_load("login@1x"))

    assert isinstance(caught.value, PerceptionError), "callers catch the general kind"
    message = str(caught.value)
    assert "abandoned" in message and "killed" in message
    assert "nothing has been learned" in message, "the message has to say whose fault it is not"


def test_the_worker_starts_a_fresh_process_after_one_is_abandoned() -> None:
    """A killed engine must not be the end of perception for the rest of the run."""
    worker = OcrWorker(argv=[sys.executable, "-c", BLOCKED_CHILD], env=dict(os.environ))
    worker.start()
    first = worker.pid

    with pytest.raises(PerceptionTimeout):
        worker.rows(_load("login@1x"), 0.5)
    assert not worker.alive

    worker.start()
    try:
        assert worker.alive and worker.pid != first
    finally:
        worker.abandon()


def test_a_real_reader_reads_again_after_one_of_its_reads_was_abandoned(
    reader: RapidOcrReader,
) -> None:
    """End to end: the shipped reader survives losing its engine mid-run.

    The abandoned read is faked by handing the reader a worker that never answers;
    the recovery is real, because the next read builds a true engine and reads the
    true fixture through it.
    """
    broken = OcrWorker(argv=[sys.executable, "-c", BLOCKED_CHILD], env=dict(os.environ))
    victim = RapidOcrReader(worker=broken, timeout_s=0.5)
    with pytest.raises(PerceptionTimeout):
        victim.read(_load("login@1x"))
    assert victim._worker is None, "the dead engine is dropped, not reused"

    # Back to a real budget: the next read starts a true worker, and starting one
    # costs the model load that the half-second above was never meant to cover.
    victim.timeout_s = DEFAULT_READ_TIMEOUT_S
    try:
        found = victim.read(_load("invoices@1x"))
    finally:
        victim.close()
    assert [e.text for e in found] == [e.text for e in reader.read(_load("invoices@1x"))]


def test_a_worker_whose_child_is_gone_reports_it_rather_than_waiting() -> None:
    """A dead engine is a failure, and not a wait for a reply that cannot come."""
    reader = _substitute(DEAD_CHILD, timeout_s=30.0)
    started = time.perf_counter()
    with pytest.raises(PerceptionError, match="without answering"):
        reader.read(_load("login@1x"))
    assert time.perf_counter() - started < 10.0, "it noticed the corpse instead of timing out"


def test_a_screenshot_bigger_than_a_pipe_does_not_block_the_caller() -> None:
    """The second unbounded wait, which the first implementation of this had.

    A pipe holds 64 KiB; the 2x fixture's PNG is larger, so writing the request to a
    child that has stopped reading blocks the writer. Doing that on the calling thread
    would mean the deadline below was never even reached - the same silent hang, one
    layer up. This test is the reason the send happens on its own thread.
    """
    assert len(_load("invoices@2x").png) > 64 * 1024, "the fixture must exceed a pipe buffer"
    reader = _substitute(BLOCKED_CHILD, timeout_s=1.0)
    started = time.perf_counter()
    with pytest.raises(PerceptionTimeout):
        reader.read(_load("invoices@2x"))
    assert time.perf_counter() - started < 10.0


def test_a_zero_budget_means_no_bound_and_only_a_test_should_ask_for_it() -> None:
    assert RapidOcrReader(timeout_s=0).timeout_s == 0.0


def test_an_abandoned_read_is_counted_and_never_remembered_as_an_answer() -> None:
    """A run's tally has to show that the eyes stopped, and the cache must not lie.

    ``ocr_timeouts`` is the honest report: a counts line that says an observation was
    abandoned is the difference between "this took a while" and "perception broke".
    """
    attempts: list[int] = []

    class Flaky:
        def read(self, _screenshot: Screenshot) -> list[Element]:
            attempts.append(1)
            if len(attempts) == 1:
                raise PerceptionTimeout("OCR did not return within 60s and was abandoned")
            return list(RapidOcrReader(engine=_one_word_engine("Recovered")).read(_screenshot))

    counters = PerceptionCounters()
    caching = CachingTextReader(Flaky(), counters=counters)

    with pytest.raises(PerceptionTimeout):
        caching.read(_load("login@1x"))
    assert (counters.ocr_reads, counters.ocr_timeouts) == (1, 1)
    assert "1 ABANDONED" in str(counters), "a report must not hide an abandoned read"

    assert [e.text for e in caching.read(_load("login@1x"))] == ["Recovered"]
    assert (counters.ocr_reads, counters.ocr_timeouts, counters.ocr_hits) == (2, 1, 0)


def test_an_ordinary_perception_failure_is_not_counted_as_an_abandonment() -> None:
    """``ocr_timeouts`` means the engine stopped answering, not that a read failed."""
    counters = PerceptionCounters()

    def explode(_image: object) -> object:
        raise RuntimeError("onnxruntime had a bad day")

    caching = CachingTextReader(RapidOcrReader(engine=explode), counters=counters)
    with pytest.raises(PerceptionError):
        caching.read(_load("login@1x"))
    assert (counters.ocr_reads, counters.ocr_timeouts) == (1, 0)


# --------------------------------------------------------------------------------------
# The thread pool, sized on purpose
# --------------------------------------------------------------------------------------


POOL_PROBE = """
import json
from skillweaver.perception.ocr import build_engine

engine = build_engine(2)
parts = (engine.text_det, engine.text_cls, engine.text_rec)
sessions = [getattr(part, "infer", None) for part in parts]
options = [s.session.get_session_options() for s in sessions if s is not None]
print(json.dumps({
    "threads": [[o.intra_op_num_threads, o.inter_op_num_threads] for o in options],
    "rec_batch": engine.text_rec.rec_batch_num,
}))
"""


def test_the_engine_is_built_with_the_thread_pool_it_was_given() -> None:
    """The knob has to reach the SESSION, not just the constructor.

    RapidOCR copies its global ``intra_op_num_threads`` into the detection,
    classification and recognition configs, and each builds its own onnxruntime
    session; the number that matters is the one those sessions ended up with. Left at
    the default it is 0, which onnxruntime reads as "one thread per core, and spin
    while they wait" - the behaviour measured at 6.11 CPU-seconds per wall second in
    this module's docstring.

    Probed in a CHILD, for the same reason the shipped reader reads in one: an
    onnxruntime session built and then torn down inside this interpreter takes its
    thread pool and its C++ statics with it, and doing that in a long-lived pytest
    process aborted the suite at exit about one run in three - ``libc++abi:
    recursive_mutex lock failed``, after every test had passed. Nothing else in this
    file loads the engine in-process, and this is where that stays true.
    """
    done = subprocess.run(
        [sys.executable, "-c", POOL_PROBE],
        capture_output=True,
        text=True,
        env=_worker_env(2),
        check=False,
    )
    assert done.returncode == 0, done.stderr
    probed = json.loads(done.stdout.strip().splitlines()[-1])
    options = probed["threads"]
    assert options, "rapidocr moved its session attribute; this test needs updating"
    for intra, inter in options:
        assert (intra, inter) == (2, 2)


def test_the_recognizer_is_given_one_line_per_call() -> None:
    """The batch has to reach the RECOGNIZER, not just the constructor.

    Same probe and the same reason for running it in a child as the pool above:
    ``rec_batch_num`` is read out of RapidOCR's ``Rec`` config into
    :class:`TextRecognizer`, and the only number that decides how much work one ONNX
    Runtime call is handed is the one that landed there. Ship the default six and OCR
    is ~1.5x slower on every real page; the module docstring has the table.
    """
    done = subprocess.run(
        [sys.executable, "-c", POOL_PROBE],
        capture_output=True,
        text=True,
        env=_worker_env(2),
        check=False,
    )
    assert done.returncode == 0, done.stderr
    probed = json.loads(done.stdout.strip().splitlines()[-1])
    assert probed["rec_batch"] == DEFAULT_REC_BATCH == 1


def test_the_default_pool_is_explicit_rather_than_the_core_count() -> None:
    assert DEFAULT_OCR_THREADS == 4
    assert ocr_threads() == DEFAULT_OCR_THREADS
    assert read_timeout_s() == DEFAULT_READ_TIMEOUT_S
    assert rec_batch() == DEFAULT_REC_BATCH


def test_the_environment_can_change_the_pool_and_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKILLWEAVER_OCR_THREADS", "2")
    monkeypatch.setenv("SKILLWEAVER_OCR_TIMEOUT_S", "12.5")
    monkeypatch.setenv("SKILLWEAVER_OCR_REC_BATCH", "6")
    assert ocr_threads() == 2
    assert read_timeout_s() == 12.5
    assert rec_batch() == 6
    assert RapidOcrReader().timeout_s == 12.5


@pytest.mark.parametrize("bad", ["", "   ", "not-a-number", "-4"])
def test_a_malformed_setting_falls_back_instead_of_refusing_to_see(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    """A misspelt variable must not be a reason for perception to have no bound."""
    monkeypatch.setenv("SKILLWEAVER_OCR_TIMEOUT_S", bad)
    monkeypatch.setenv("SKILLWEAVER_OCR_THREADS", bad)
    monkeypatch.setenv("SKILLWEAVER_OCR_REC_BATCH", bad)
    assert read_timeout_s() == DEFAULT_READ_TIMEOUT_S
    assert ocr_threads() == DEFAULT_OCR_THREADS
    assert rec_batch() == DEFAULT_REC_BATCH


def test_the_child_is_told_the_pool_size_by_every_name_that_reads_one() -> None:
    """OpenMP and the BLAS libraries read their variables AT IMPORT, before the child
    can pass anything to a session, so the only moment early enough is the environment
    handed to the process."""
    env = _worker_env(3)
    assert env["SKILLWEAVER_OCR_THREADS"] == "3"
    assert env["OMP_NUM_THREADS"] == "3"
    assert env["OPENBLAS_NUM_THREADS"] == "3"
    assert env["MKL_NUM_THREADS"] == "3"
    assert str(Path("src")) in env["PYTHONPATH"] or "skillweaver" in env["PYTHONPATH"]


def test_the_child_inherits_the_batch_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped reader reads in a CHILD, so a knob the child cannot see is no knob.

    ``rec_batch`` is read in the process that builds the engine, which on the shipped
    path is the worker - so the only thing that carries the setting across is the
    environment being copied rather than rebuilt.
    """
    monkeypatch.setenv("SKILLWEAVER_OCR_REC_BATCH", "6")
    assert _worker_env(2)["SKILLWEAVER_OCR_REC_BATCH"] == "6"


def test_the_worker_command_starts_this_interpreter_and_nothing_else() -> None:
    argv = _worker_argv()
    assert argv[0] == sys.executable and argv[1] == "-c"
    assert "_worker_main" in argv[2]
