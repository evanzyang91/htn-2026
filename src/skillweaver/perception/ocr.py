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

Why this is not simply ``engine(image)``
----------------------------------------

Reading a screen is the single most expensive thing perception does - on a 1200x780
frame the stock call costs around 1.5 seconds, against 13ms for the element detector -
and an agent reads the screen after every action. Three things make that affordable,
and all three are measurable with ``scripts/bench_perception.py``:

* **Thread count.** ONNX Runtime's default is one thread per core, and the recognizer
  is a small model run over many tiny crops, so on a large machine it spends its time
  in synchronisation rather than arithmetic. Measured on a 20-core box: 1567ms at the
  default, 590ms at six threads. :data:`DEFAULT_THREADS` is therefore a modest number
  rather than "all of them", and it is the one setting here most worth re-measuring on
  new hardware.
* **No rotation classifier.** It exists to detect text that has been turned upside
  down, which happens in photographs of documents and does not happen in a screenshot.
  Skipping it is worth about a quarter of the remaining time and cannot cost accuracy
  on an axis-aligned rendering of a web page.
* **A recognition cache.** Detection is 130ms of that 1.5 seconds; recognising the
  ~100 resulting line crops is the rest. Between two consecutive screens of an
  application most lines are pixel-identical - the navigation bar, the folder list,
  the rows nothing touched - so each crop is hashed and its text remembered, and only
  the genuinely new lines are recognised. The saving grows with how little changed,
  which is exactly the case an agent taking one action at a time is always in.

The cache is keyed by the crop's own pixels, so it cannot return the wrong text for a
line: two crops that hash the same ARE the same image. Its only cost is memory, which
is bounded by :data:`CACHE_LINES`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import threading
from collections import OrderedDict
from typing import Any

from skillweaver.contracts import Box, Element, ElementKind, ElementSource, Screenshot
from skillweaver.errors import PerceptionError
from skillweaver.perception.arrays import array_of
from skillweaver.perception.elements import reading_order, stable_id

__all__ = [
    "CACHE_LINES",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_THREADS",
    "RapidOcrReader",
]

#: Recognitions below this score are dropped: at that level RapidOCR is reporting
#: shapes it could not read, and a wrong label is worse for an agent than no label.
DEFAULT_MIN_CONFIDENCE = 0.5

DEFAULT_THREADS = 6
"""ONNX Runtime threads for detection and recognition.

Not the core count. The recognizer runs a small network over many small crops, and
past a handful of threads the coordination costs more than the work: on a 20-core
machine the stock "one per core" setting measured 1567ms against 590ms at six.
Overridable with ``SKILLWEAVER_OCR_THREADS`` for a machine where that is wrong, and
capped by the actual core count so a small machine is never oversubscribed.
"""

CACHE_LINES = 4096
"""Recognized line crops remembered, keyed by their pixels.

A screen holds around a hundred lines, so this is tens of screens' worth - enough that
a run moving back and forth between two screens keeps hitting, and small enough that
the digests and short strings involved stay negligible.
"""


def _threads() -> int:
    """The thread count to build the engine with, honouring the environment."""
    raw = os.environ.get("SKILLWEAVER_OCR_THREADS", "").strip()
    wanted = DEFAULT_THREADS
    if raw:
        try:
            wanted = int(raw)
        except ValueError:
            wanted = DEFAULT_THREADS
    cores = os.cpu_count() or DEFAULT_THREADS
    return max(1, min(wanted, cores))


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
            across readers. When given, nothing is imported or loaded. A caller that
            supplies one is trusted to have configured it; the staged fast path is used
            only when the object exposes the stages, and a plain callable still works.
        cache: Whether to remember recognized line crops between reads. ``False`` for a
            benchmark measuring the cold cost of a single frame.
        threads: ONNX Runtime threads. ``None`` uses :func:`_threads`.
    """

    __slots__ = ("_cache", "_engine", "_hits", "_lock", "_misses", "_threads", "min_confidence")

    def __init__(
        self,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        engine: Any | None = None,
        cache: bool = True,
        threads: int | None = None,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self._engine = engine
        self._lock = threading.Lock()
        self._threads = _threads() if threads is None else max(1, int(threads))
        self._cache: OrderedDict[bytes, tuple[str, float]] | None = OrderedDict() if cache else None
        self._hits = 0
        self._misses = 0

    @property
    def loaded(self) -> bool:
        """Whether the OCR engine has been built yet."""
        return self._engine is not None

    def cache_info(self) -> dict[str, int]:
        """Line-crop cache hits, misses and size. For benchmarks and tests."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "lines": 0 if self._cache is None else len(self._cache),
        }

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
                self._engine = RapidOCR(
                    intra_op_num_threads=self._threads,
                    inter_op_num_threads=self._threads,
                )
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
        image = array_of(screenshot, logical=False)
        scale = screenshot.scale if screenshot.scale > 0 else 1.0
        try:
            raw = self._recognize(engine, image)
        except PerceptionError:
            raise
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

    # -- inference ---------------------------------------------------------------------

    def _recognize(self, engine: Any, image: Any) -> Any:
        """The engine's rows for one frame, through the staged path when it is available.

        The staged path is detection, then recognition of only those line crops whose
        pixels have not been read before. It needs three things from the engine -
        ``text_det``, ``get_crop_img_list`` and ``text_rec`` - which is what RapidOCR
        exposes; anything else (a test double, a future release that renames them) falls
        back to calling the engine, so this is an optimisation rather than a dependency.
        """
        staged = self._stages(engine)
        if staged is None:
            # ``use_cls`` is offered only to an engine that has the setting, so a plain
            # callable double - which is what a test supplies - is still called the way
            # its signature promises rather than with a keyword it never accepted.
            result = engine(image, use_cls=False) if hasattr(engine, "use_cls") else engine(image)
            return result[0] if isinstance(result, tuple) else result
        return self._staged_read(image, *staged)

    @staticmethod
    def _stages(engine: Any) -> tuple[Any, Any, Any] | None:
        """``(detect, crop, recognize)`` when the engine exposes them, else ``None``."""
        detect = getattr(engine, "text_det", None)
        crop = getattr(engine, "get_crop_img_list", None)
        recognize = getattr(engine, "text_rec", None)
        if detect is None or crop is None or recognize is None:
            return None
        return detect, crop, recognize

    def _staged_read(self, image: Any, detect: Any, crop: Any, recognize: Any) -> list[Any]:
        """Detect lines, recognise only the unseen ones, and rebuild the engine's rows.

        The boxes come straight from detection on the unmodified physical frame. That
        skips RapidOCR's own ``preprocess``/letterbox steps, which only act on images
        far from a screenshot's proportions - a frame taller than it is wide, or one
        larger than 2000px on its longest side - so for a capture from this project's
        controllers the geometry is the same and no coordinate has to be mapped back.
        """
        boxes, _elapsed = detect(image)
        if boxes is None or len(boxes) < 1:
            return []
        crops = crop(image, boxes)

        wanted: list[int] = []
        answers: list[tuple[str, float] | None] = []
        for index, piece in enumerate(crops):
            remembered = self._remembered(piece)
            answers.append(remembered)
            if remembered is None:
                wanted.append(index)

        if wanted:
            fresh, _rec_elapsed = recognize([crops[i] for i in wanted])
            for position, index in enumerate(wanted):
                answer = _rec_answer(fresh, position)
                answers[index] = answer
                self._remember(crops[index], answer)

        rows: list[Any] = []
        for box, answer in zip(boxes, answers, strict=False):
            if answer is None:
                continue
            text, score = answer
            rows.append([box, text, score])
        return rows

    # -- the line cache ------------------------------------------------------------------

    def _remembered(self, crop: Any) -> tuple[str, float] | None:
        """What this exact crop said last time, or ``None``."""
        if self._cache is None:
            return None
        key = _digest(crop)
        if key is None:
            return None
        found = self._cache.get(key)
        if found is None:
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return found

    def _remember(self, crop: Any, answer: tuple[str, float] | None) -> None:
        """Remember what a crop said, including that it said nothing readable."""
        if self._cache is None or answer is None:
            return
        key = _digest(crop)
        if key is None:
            return
        self._cache[key] = answer
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_LINES:
            self._cache.popitem(last=False)


def _digest(crop: Any) -> bytes | None:
    """A content digest of one line crop, or ``None`` for something unhashable.

    blake2b over the raw pixel buffer plus the shape. The shape is in the digest
    because two different crops can hold the same bytes at different dimensions, and
    reading one as the other would put text in the wrong place.
    """
    try:
        shape = repr(getattr(crop, "shape", None)).encode()
        buffer = crop.tobytes()
    except (AttributeError, ValueError, TypeError):
        return None
    return hashlib.blake2b(buffer, digest_size=16, person=b"ocr-line", salt=b"").digest() + shape


def _rec_answer(fresh: Any, position: int) -> tuple[str, float] | None:
    """``(text, score)`` for one recognized crop out of a recognizer's batch reply."""
    try:
        row = fresh[position]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        text, score = row[0], row[1]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        confidence = float(score)
    except (TypeError, ValueError):
        confidence = 0.0
    return str(text), max(0.0, min(1.0, confidence))


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
