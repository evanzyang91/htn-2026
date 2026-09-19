"""Fake perception: detector, text reader, fingerprinter, element index, perceiver.

``FakeDetector`` and ``FakeTextReader`` return whatever the fake state DECLARES for a
screenshot - they look the state up by the screenshot's PNG bytes, so they also work
on a screenshot captured earlier. Elements declared with ``source=ElementSource.ocr``
come from the reader; all others come from the detector.
"""

from __future__ import annotations

import difflib
import hashlib
import math
from collections.abc import Sequence

from skillweaver.contracts import (
    Controller,
    Detector,
    Element,
    ElementKind,
    ElementSource,
    Fingerprint,
    Fingerprinter,
    Observation,
    Point,
    Screenshot,
    TextReader,
    utcnow,
)
from skillweaver.errors import PerceptionError
from tests.fakes.controller import FakeController


def _declared(controller: FakeController, screenshot: Screenshot) -> tuple[Element, ...]:
    state = controller.state_for_png(screenshot.png)
    if state is None:
        raise PerceptionError("screenshot does not belong to any state of the fake controller")
    return state.elements


class FakeDetector:
    """A ``contracts.Detector``: the declared non-OCR elements of the screenshot's
    state. ``calls`` counts ``detect`` invocations."""

    def __init__(self, controller: FakeController) -> None:
        self._controller = controller
        self.calls = 0

    def detect(self, screenshot: Screenshot) -> list[Element]:
        self.calls += 1
        found = [
            e for e in _declared(self._controller, screenshot) if e.source != ElementSource.ocr
        ]
        return sorted(found, key=lambda e: -e.confidence)


class FakeTextReader:
    """A ``contracts.TextReader``: the declared ``source=ocr`` elements of the
    screenshot's state, in reading order. ``calls`` counts ``read`` invocations."""

    def __init__(self, controller: FakeController) -> None:
        self._controller = controller
        self.calls = 0

    def read(self, screenshot: Screenshot) -> list[Element]:
        self.calls += 1
        found = [
            e for e in _declared(self._controller, screenshot) if e.source == ElementSource.ocr
        ]
        return sorted(found, key=_reading_order)


def _h(*chunks: str) -> str:
    return hashlib.sha256("\x1f".join(chunks).encode()).hexdigest()[:16]


class FakeFingerprinter:
    """A ``contracts.Fingerprinter`` computed purely from its inputs (never from the
    pixels): ``parts`` holds ``url``, ``layout`` (kinds and boxes) and ``text``
    sub-hashes, and ``value`` hashes the three. Deterministic."""

    def __init__(self) -> None:
        self.calls = 0

    def fingerprint(
        self, screenshot: Screenshot, elements: Sequence[Element], url: str | None = None
    ) -> Fingerprint:
        self.calls += 1
        ordered = sorted(elements, key=_reading_order)
        parts = {
            "url": _h(url or ""),
            "layout": _h(*(f"{e.kind}:{e.box.x},{e.box.y},{e.box.w},{e.box.h}" for e in ordered)),
            "text": _h(*(e.text for e in ordered)),
        }
        return Fingerprint(_h(parts["url"], parts["layout"], parts["text"]), parts)


def _reading_order(element: Element) -> tuple[int, int]:
    return (element.box.y, element.box.x)


_KIND_WORDS: dict[ElementKind, frozenset[str]] = {
    ElementKind.button: frozenset({"button"}),
    ElementKind.text_field: frozenset({"field", "input", "textbox", "box"}),
    ElementKind.checkbox: frozenset({"checkbox"}),
    ElementKind.radio: frozenset({"radio"}),
    ElementKind.link: frozenset({"link"}),
    ElementKind.icon: frozenset({"icon"}),
    ElementKind.menu: frozenset({"menu"}),
    ElementKind.tab: frozenset({"tab"}),
    ElementKind.row: frozenset({"row", "item", "entry"}),
    ElementKind.image: frozenset({"image", "picture"}),
    ElementKind.text: frozenset({"text", "label", "heading"}),
    ElementKind.other: frozenset(),
}


class SimpleElementIndex:
    """A small, dependency-free ``contracts.ElementIndex`` over a fixed element list.

    Good enough for tests: exact and substring text matches rank above fuzzy ones
    (``difflib`` ratio of at least ``0.6``); ``best`` scores shared words plus a bonus
    when the description names the element's kind.
    """

    def __init__(self, elements: Sequence[Element]) -> None:
        self._elements = sorted(elements, key=_reading_order)

    def all(self) -> list[Element]:
        return list(self._elements)

    def by_kind(self, kind: ElementKind) -> list[Element]:
        return [e for e in self._elements if e.kind == kind]

    def find_text(
        self, query: str, kind: ElementKind | None = None, fuzzy: bool = True
    ) -> list[Element]:
        q = query.strip().lower()
        if not q:
            return []
        scored: list[tuple[float, Element]] = []
        for e in self._elements:
            if kind is not None and e.kind != kind:
                continue
            text = e.text.strip().lower()
            if not text:
                continue
            if text == q:
                score = 3.0
            elif q in text:
                score = 2.0 + len(q) / len(text)
            elif fuzzy:
                ratio = max(
                    difflib.SequenceMatcher(None, q, candidate).ratio()
                    for candidate in (text, *text.split())
                )
                score = ratio if ratio >= 0.6 else 0.0
            else:
                score = 0.0
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda pair: -pair[0])
        return [e for _, e in scored]

    def nearest(self, point: Point, kind: ElementKind | None = None) -> list[Element]:
        def distance(e: Element) -> float:
            dx = max(e.box.x - point.x, 0, point.x - (e.box.x + e.box.w - 1))
            dy = max(e.box.y - point.y, 0, point.y - (e.box.y + e.box.h - 1))
            return math.hypot(dx, dy)

        pool = [e for e in self._elements if kind is None or e.kind == kind]
        return sorted(pool, key=distance)

    def containing(self, point: Point) -> list[Element]:
        return sorted(
            (e for e in self._elements if e.box.contains(point)), key=lambda e: e.box.area
        )

    def best(self, description: str) -> list[Element]:
        words = set(description.lower().split())
        scored: list[tuple[float, Element]] = []
        for e in self._elements:
            score = float(len(words & set(e.text.lower().split())))
            if words & (_KIND_WORDS[e.kind] | {e.kind.value}):
                score += 1.5
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda pair: -pair[0])
        return [e for _, e in scored]


class FakePerceiver:
    """A ``contracts.Perceiver`` composing any detector, reader and fingerprinter -
    fake or real - with a :class:`SimpleElementIndex`. No merging of overlapping
    elements is attempted. ``calls`` counts ``observe`` invocations."""

    def __init__(
        self, detector: Detector, reader: TextReader, fingerprinter: Fingerprinter
    ) -> None:
        self.detector = detector
        self.reader = reader
        self.fingerprinter = fingerprinter
        self.calls = 0

    @classmethod
    def for_controller(cls, controller: FakeController) -> FakePerceiver:
        """The usual wiring: fake detector and reader bound to ``controller``."""
        return cls(FakeDetector(controller), FakeTextReader(controller), FakeFingerprinter())

    def observe(self, controller: Controller) -> Observation:
        self.calls += 1
        shot = controller.capture()
        url = controller.url()
        elements = sorted(
            [*self.detector.detect(shot), *self.reader.read(shot)], key=_reading_order
        )
        return Observation(
            screenshot=shot,
            elements=tuple(elements),
            index=SimpleElementIndex(elements),
            fingerprint=self.fingerprinter.fingerprint(shot, elements, url),
            url=url,
            taken_at=utcnow(),
        )
