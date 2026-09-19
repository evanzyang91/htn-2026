"""OCR: reading the text on a screenshot with RapidOCR (PP-OCRv4, ONNX Runtime, local).

:class:`RapidOcrReader` is the project's :class:`skillweaver.contracts.TextReader`. It
runs the bundled ONNX models on the CPU, with no network and no API key, which is why
it can be the default: perception must work in a test, in an eval loop and on a plane.

Two decisions worth knowing about before using it:

**The model loads on the first :meth:`RapidOcrReader.read`, not on import.**
    Importing this module costs nothing, so a CLI that lists skills does not pay for
    ONNX Runtime. Construct the reader wherever you like; the 1-2 second load happens
    when text is first actually needed, and only once per reader.

**Failure is loud.** If the engine cannot be imported or built, or inference raises,
    :meth:`read` raises :class:`~skillweaver.errors.PerceptionError`. It never returns
    an empty list to mean "OCR is broken", because an agent cannot tell that apart from
    "this screen has no text" and would go on to click blindly. An empty list means the
    engine ran and found nothing.

OCR runs on the PHYSICAL-resolution image (sharper - it is what the model wants) and the
boxes are divided by ``Screenshot.scale`` on the way out, so everything this module
returns is in LOGICAL pixels like the rest of the system.

Not reading the same pixels twice
---------------------------------

Reading is the expensive half of perception by an enormous margin. Profiled against live
pages at 1280x800, a full-page read costs 0.44s to over 7s while capture costs 0.02-0.05s
and detection 0.06s - **84% to 97% of all perception time**, on every real page measured.
Two ways of making the read itself cheaper were measured and both lost: cropping to the
detector's boxes took 53.8s against 7.4s for one full-page read of the same frame (the
engine's per-call overhead dwarfs the pixel saving), and downscaling showed no reliable
win. So the saving has to come from not reading at all.

:class:`CachingTextReader` is that saving, and :class:`PerceptionCounters` is how you know
it worked. Counts, unlike seconds, do not move when the machine is busy, so "this task
went from 14 reads to 3" is a claim that survives being measured on a loaded laptop.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from skillweaver.contracts import Box, Element, ElementKind, ElementSource, Screenshot
from skillweaver.errors import PerceptionError
from skillweaver.perception.elements import reading_order, stable_id

__all__ = [
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_MIN_CONFIDENCE",
    "CachingTextReader",
    "PerceptionCounters",
    "PerceptionCounts",
    "RapidOcrReader",
    "content_key",
]

#: Recognitions below this score are dropped: at that level RapidOCR is reporting
#: shapes it could not read, and a wrong label is worse for an agent than no label.
DEFAULT_MIN_CONFIDENCE = 0.5


class RapidOcrReader:
    """A :class:`~skillweaver.contracts.TextReader` backed by RapidOCR.

    One element of kind :attr:`~skillweaver.contracts.ElementKind.text` per recognized
    line, with ``source=ElementSource.ocr``, a real confidence from the recognizer, a
    box in LOGICAL pixels and a :func:`~skillweaver.perception.elements.stable_id`
    already filled in.

    Instances are safe to share between threads (the engine is built once under a lock)
    and are cheap to create.

    Args:
        min_confidence: Recognitions scoring below this are dropped.
        engine: An already-built RapidOCR callable, for tests or for reusing one engine
            across readers. When given, nothing is imported or loaded.
    """

    __slots__ = ("_engine", "_lock", "min_confidence")

    def __init__(
        self,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        engine: Any | None = None,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self._engine = engine
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """Whether the OCR engine has been built yet."""
        return self._engine is not None

    def _ensure_engine(self) -> Any:
        """Build the engine on first use.

        Raises:
            PerceptionError: if RapidOCR is not installed or its models cannot be
                loaded. The message names the cause, because "OCR silently found no
                text" is the single most expensive failure mode in this pipeline.
        """
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise PerceptionError(
                    "OCR is unavailable: rapidocr-onnxruntime is not installed "
                    f"({exc}). Install the project's dependencies with `make install`."
                ) from exc
            try:
                self._engine = RapidOCR()
            except Exception as exc:  # model files missing, ONNX Runtime broken, ...
                raise PerceptionError(f"OCR engine could not be loaded: {exc}") from exc
            return self._engine

    def read(self, screenshot: Screenshot) -> list[Element]:
        """Read the text on ``screenshot``.

        Returns one element per recognized line, in reading order. An empty list means
        the engine ran and found no text.

        Raises:
            PerceptionError: if the screenshot cannot be decoded, the engine cannot be
                loaded, or inference fails.
        """
        engine = self._ensure_engine()
        image = screenshot.to_array(logical=False)
        scale = screenshot.scale if screenshot.scale > 0 else 1.0
        try:
            raw, _elapsed = engine(image)
        except Exception as exc:
            raise PerceptionError(f"OCR inference failed: {exc}") from exc

        elements: list[Element] = []
        for entry in raw or ():
            parsed = _parse_entry(entry)
            if parsed is None:
                continue
            polygon, text, confidence = parsed
            if not text.strip() or confidence < self.min_confidence:
                continue
            box = _polygon_to_box(polygon, scale, screenshot.width, screenshot.height)
            if box is None:
                continue
            element = Element(
                box=box,
                kind=ElementKind.text,
                text=text.strip(),
                confidence=confidence,
                stable_id=None,
                source=ElementSource.ocr,
            )
            elements.append(dataclasses.replace(element, stable_id=stable_id(element)))
        return reading_order(elements)


def _parse_entry(entry: Any) -> tuple[Any, str, float] | None:
    """Pull ``(polygon, text, confidence)`` out of one RapidOCR result row.

    RapidOCR returns ``[polygon, text, score]`` rows, but the exact container types have
    moved between releases, so this stays shape-driven rather than trusting one version.
    """
    try:
        polygon, text, score = entry[0], entry[1], entry[2]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        confidence = float(score)
    except (TypeError, ValueError):
        confidence = 0.0
    return polygon, str(text), max(0.0, min(1.0, confidence))


def _polygon_to_box(polygon: Any, scale: float, width: int, height: int) -> Box | None:
    """Convert a physical-pixel quadrilateral to a clamped LOGICAL-pixel ``Box``.

    Dividing by ``scale`` here is the one conversion that keeps OCR honest: skip it on a
    Retina capture and every click derived from text lands twice as far down the screen.
    """
    import numpy as np

    try:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if points.size == 0 or not np.all(np.isfinite(points)):
        return None
    x0 = float(points[:, 0].min()) / scale
    y0 = float(points[:, 1].min()) / scale
    x1 = float(points[:, 0].max()) / scale
    y1 = float(points[:, 1].max()) / scale

    left = max(0, min(int(np.floor(x0)), width))
    top = max(0, min(int(np.floor(y0)), height))
    right = max(left, min(int(np.ceil(x1)), width))
    bottom = max(top, min(int(np.ceil(y1)), height))
    box = Box(left, top, max(right - left, 1), max(bottom - top, 1))
    return box if box.area > 0 else None


# --------------------------------------------------------------------------------------
# Counting what perception did
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerceptionCounts:
    """How much work perception did over some window, as COUNTS rather than seconds.

    Seconds are the number everybody wants and the number nobody can trust: the same
    frame read on a quiet machine and on one running four other workers differs by more
    than any optimization in this module could ever buy. Counts do not move, so this is
    the type a run reports alongside its model calls and its dollars.

    Attributes:
        observations: Completed :meth:`~skillweaver.contracts.Perceiver.observe` calls.
        captures: Frames taken from the controller.
        detections: Detector (YOLO) invocations.
        ocr_reads: Text reads that actually ran the OCR engine. **This is the number
            an optimization here has to move.**
        ocr_hits: Text reads answered from cache without touching the engine.
    """

    observations: int = 0
    captures: int = 0
    detections: int = 0
    ocr_reads: int = 0
    ocr_hits: int = 0

    @property
    def text_reads(self) -> int:
        """Every text read asked for, served by the engine or by the cache."""
        return self.ocr_reads + self.ocr_hits

    @property
    def hit_rate(self) -> float:
        """Fraction of text reads the cache answered, in ``0.0..1.0``.

        ``0.0`` when no text read was asked for, rather than a division by zero.
        """
        asked = self.text_reads
        return self.ocr_hits / asked if asked else 0.0

    def __add__(self, other: PerceptionCounts) -> PerceptionCounts:
        return PerceptionCounts(
            observations=self.observations + other.observations,
            captures=self.captures + other.captures,
            detections=self.detections + other.detections,
            ocr_reads=self.ocr_reads + other.ocr_reads,
            ocr_hits=self.ocr_hits + other.ocr_hits,
        )

    def __sub__(self, other: PerceptionCounts) -> PerceptionCounts:
        """The work done SINCE ``other``, which is how a per-attempt figure is taken.

        Counters only ever climb, so every field is clamped at zero rather than
        reporting a negative amount of work if the two came from different counters.
        """
        return PerceptionCounts(
            observations=max(self.observations - other.observations, 0),
            captures=max(self.captures - other.captures, 0),
            detections=max(self.detections - other.detections, 0),
            ocr_reads=max(self.ocr_reads - other.ocr_reads, 0),
            ocr_hits=max(self.ocr_hits - other.ocr_hits, 0),
        )

    def __bool__(self) -> bool:
        """Whether anything at all was counted, so a report can stay silent otherwise."""
        return bool(self.observations or self.captures or self.detections or self.text_reads)

    def __str__(self) -> str:
        return (
            f"{self.observations} observation(s), {self.ocr_reads} OCR read(s) "
            f"+ {self.ocr_hits} cached, {self.detections} detection(s)"
        )


@dataclass(slots=True)
class PerceptionCounters:
    """MUTABLE running tally of perception work, shared by everything that does some.

    Modelled on :class:`~skillweaver.contracts.Spend`: one instance is threaded through
    a perceiver and its reader, each of them charges its own work to it, and a caller
    takes :meth:`snapshot` before and after a stretch to get that stretch's
    :class:`PerceptionCounts`. Not thread-safe, like ``Spend``; the counts are
    diagnostics, and a lost increment under concurrency is not worth a lock on the
    hot path.
    """

    observations: int = 0
    captures: int = 0
    detections: int = 0
    ocr_reads: int = 0
    ocr_hits: int = 0

    def snapshot(self) -> PerceptionCounts:
        """An immutable copy of the tally as it stands."""
        return PerceptionCounts(
            observations=self.observations,
            captures=self.captures,
            detections=self.detections,
            ocr_reads=self.ocr_reads,
            ocr_hits=self.ocr_hits,
        )

    def since(self, mark: PerceptionCounts) -> PerceptionCounts:
        """The work done since ``mark`` was taken from this counter."""
        return self.snapshot() - mark

    def reset(self) -> None:
        """Zero every field, for a caller measuring one stretch in isolation."""
        self.observations = self.captures = self.detections = 0
        self.ocr_reads = self.ocr_hits = 0

    def __str__(self) -> str:
        return str(self.snapshot())


# --------------------------------------------------------------------------------------
# Not reading the same pixels twice
# --------------------------------------------------------------------------------------

DEFAULT_CACHE_SIZE = 32
"""Frames a :class:`CachingTextReader` remembers.

Small on purpose. The hit this cache exists to catch is the one the agent hands it
immediately - a check, a critic and the next loop iteration all looking at the screen
the last action left - and a run that comes back to a frame it last saw thirty frames
ago has almost certainly re-rendered it in the meantime. Each entry holds a frame's
text elements, not its pixels, so the bound is on entries rather than bytes.
"""


def content_key(screenshot: Screenshot) -> str:
    """The identity of the pixels a text read would be performed on.

    Two screenshots share a key exactly when reading them is guaranteed to produce the
    same elements: the same PNG bytes *and* the same declared geometry. The geometry
    belongs in the key because :attr:`~skillweaver.contracts.Screenshot.scale` is what
    divides OCR's physical coordinates down to logical ones - the same bytes declared at
    ``scale=2.0`` yield boxes at half the position of the same bytes at ``scale=1.0``,
    and serving one for the other is the doubled-coordinate bug this project warns about
    everywhere else.

    Hashing the whole PNG on every read is affordable by four orders of magnitude: 0.26ms
    median for a 334 KiB live Wikipedia frame, against the 440ms to 7.4s read it may save.
    """
    digest = hashlib.blake2b(screenshot.png, digest_size=16).hexdigest()
    return f"{digest}:{screenshot.width}x{screenshot.height}@{screenshot.scale:g}"


class CachingTextReader:
    """A :class:`~skillweaver.contracts.TextReader` that never reads the same pixels twice.

    Wraps another reader with a bounded LRU keyed on :func:`content_key`, so an
    observation of a screen nothing has changed reuses the previous read instead of
    paying for it again. Measured on live Wikipedia pages, four captures of an untouched
    page are byte-identical, so this is the common case and not a corner one.

    Why the key is exact, and not "the same state"
    ----------------------------------------------

    The obvious improvement is to serve a cached read whenever the new frame is the SAME
    STATE as the cached one, by
    :meth:`~skillweaver.contracts.Fingerprint.similarity` against
    :data:`~skillweaver.perception.fingerprint.SAME_STATE_THRESHOLD`. That judgment is
    the right one for its own question and the wrong one for this one, and the
    fingerprinter's own measured table says why: ``dense_text_scrolled_slightly`` scores
    0.750 and ``same_screen_clock_tick`` scores 1.000. Both are correctly the same state;
    in both the text has MOVED or CHANGED.

    That is not a theoretical worry. Taking the 12 frames one live Wikipedia exploration
    actually captured, grouping them by perceptual hash and normalized URL - a far
    STRICTER key than the 0.62 same-state cut, since it demands every pixel row and the
    URL agree - and reading all 12 for real: three frames would have been served another
    frame's text. The worst of them differed by twelve strings that were still on screen
    but at a different box, because the page had been scrolled. Those boxes are precisely
    what a skill clicks, and a skill cannot tell a stale read from a true one - it just
    clicks. A slow read costs seconds; a stale one costs a wrong click on a real site, so
    this cache only ever answers for pixels it has literally seen.

    Nothing is given up for that. The exact key caught every duplicate those 12 frames
    contained: 12 frames, 9 distinct, 3 reads saved, which is the whole of what was
    safely available.

    Args:
        inner: The reader that does the actual work on a miss.
        capacity: How many frames to remember. ``0`` disables caching while leaving the
            counting intact.
        counters: The tally to charge reads and hits to. A fresh one is made when not
            given; share one to count a whole perceiver's work together.
    """

    __slots__ = ("_cache", "_capacity", "_counters", "_inner", "_lock")

    def __init__(
        self,
        inner: Any,
        *,
        capacity: int = DEFAULT_CACHE_SIZE,
        counters: PerceptionCounters | None = None,
    ) -> None:
        self._inner = inner
        self._capacity = max(int(capacity), 0)
        self._counters = counters if counters is not None else PerceptionCounters()
        self._cache: OrderedDict[str, tuple[Element, ...]] = OrderedDict()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"CachingTextReader({self._inner!r}, capacity={self._capacity}, "
            f"cached={len(self._cache)})"
        )

    @property
    def inner(self) -> Any:
        """The wrapped reader."""
        return self._inner

    @property
    def counters(self) -> PerceptionCounters:
        """The tally this reader charges its work to."""
        return self._counters

    @property
    def capacity(self) -> int:
        """How many frames are remembered at most."""
        return self._capacity

    def clear(self) -> None:
        """Forget every remembered frame. The counters are left alone."""
        with self._lock:
            self._cache.clear()

    def read(self, screenshot: Screenshot) -> list[Element]:
        """The text on ``screenshot``, read once and remembered.

        The returned list is a fresh one every time, so a caller that sorts or trims it
        cannot corrupt what the next caller is served.

        Raises:
            PerceptionError: whatever ``inner`` raises. A failed read is NOT cached: an
                engine that could not load this time may load next time, and caching the
                failure would turn a transient fault into a permanent blind spot.
        """
        if self._capacity == 0:
            self._counters.ocr_reads += 1
            return self._inner.read(screenshot)

        key = content_key(screenshot)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
        if hit is not None:
            self._counters.ocr_hits += 1
            return list(hit)

        self._counters.ocr_reads += 1
        elements = tuple(self._inner.read(screenshot))
        with self._lock:
            self._cache[key] = elements
            self._cache.move_to_end(key)
            while len(self._cache) > self._capacity:
                self._cache.popitem(last=False)
        return list(elements)
