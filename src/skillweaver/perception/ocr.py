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
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Any

from skillweaver.contracts import Box, Element, ElementKind, ElementSource, Screenshot
from skillweaver.errors import PerceptionError
from skillweaver.perception.elements import reading_order, stable_id

__all__ = ["DEFAULT_MIN_CONFIDENCE", "RapidOcrReader"]

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
