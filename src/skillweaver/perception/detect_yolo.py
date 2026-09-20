"""The YOLO element detector: a ``Detector`` over pixels, never over the DOM.

``Screenshot`` in, ``Element`` boxes out in LOGICAL pixels with real confidences and
``source=yolo``. The weights ship with the repository at ``data/models/ui_detector.pt`` -
the one file under ``data/`` that is not git-ignored - and there is currently no in-repo
path to regenerate them.

What they have SEEN is a local demo app and live public pages. An earlier version trained
on the demo app alone was very good at it and close to useless anywhere else, so a page
from a site the training set never held is still the weakest case.

Missing weights RAISE: an empty list is indistinguishable from a blank screen, and an
agent told it sees nothing on a page of buttons fails in a way nobody can debug. Loading
is lazy, so importing this module costs nothing.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from skillweaver.config import settings
from skillweaver.contracts import Box, Element, ElementSource, Screenshot
from skillweaver.errors import PerceptionError
from skillweaver.perception.labeling import CLASS_NAMES, kind_of

DEFAULT_WEIGHTS_NAME = "ui_detector.pt"
"""File name of the fine-tuned detector inside ``Settings.models_dir``."""

DEFAULT_CONFIDENCE = 0.25
"""Score below which a detection is dropped. Low enough that a faint control still
reaches the merger, which has OCR text and heuristics to settle it."""

DEFAULT_IOU = 0.5
"""Non-maximum-suppression threshold. UI controls sit in tight rows, so a lower
value would merge two adjacent buttons into one box."""

DEFAULT_MAX_DETECTIONS = 300
"""Ceiling on boxes per frame. It does NOT bind on a real page: across sixteen dense live
page/viewport pairs the detection count at this cap equalled the count at
``max_detections=3000`` every time, the largest being 135 on an MDN page whose DOM reports
402 elements. What loses elements on a dense page is RECALL, by a factor of three, and
the cap loses none. The only frame that
reaches it is a synthetic grid of 900 buttons (307 uncapped), where the boxes either side
of rank 300 score 0.269 to 0.250, so the cut can only bite inside the
:data:`DEFAULT_CONFIDENCE` floor band.

Both reasons previously recorded for not raising it are WRONG. It does not buy back OCR
time - the reader reads the whole frame and caches on pixels, and ``ocr_reads`` is
identical at 300 and 3000 - nor NMS time, which was flat at 69.5/68.1/69.4 ms for
300/1000/3000. What more detections cost is ``merge_elements``, which is quadratic: 20.9 ms
for 133 detections, 180.8 for 1064, 1057.6 for 3059, and paid on EVERY observation
including one the text cache serves free. So the cap bounds that quadratic for the frame
where the model melts down; raising it raises a quadratic. Making the merge cheap enough
that the ceiling stops mattering is the follow-up.

One side effect, since the cap sits in front of a cache: at the cap a 3-pixel scroll of
that synthetic grid swapped one element out of the surviving set and one in (Jaccard
0.992 against 1.000 uncapped). It did not reach state identity and no real page comes
within 2.2x of the cap."""

_BUILD_HINT = (
    "The detector ships with the repository at data/models/ui_detector.pt; there is "
    "currently no in-repo path to regenerate it, so restore that file from version control."
)


def default_weights_path() -> Path:
    """Where the detector expects its weights: ``<models_dir>/ui_detector.pt``."""
    return settings().models_dir / DEFAULT_WEIGHTS_NAME


class YoloDetector:
    """A ``contracts.Detector`` backed by an ultralytics YOLO model.

    Safe to share between threads: the one-time load is guarded by a lock.

    Args:
        weights: Path to a ``.pt`` file. ``None`` means :func:`default_weights_path`.
        confidence: Minimum score for a detection to be returned.
        iou: Non-maximum-suppression IoU threshold.
        max_detections: Hard ceiling on boxes returned for one frame.
        device: Torch device string; ``None`` lets ultralytics choose.

    Raises:
        PerceptionError: from :meth:`detect`, never ``__init__``, when the weights are
            missing or unloadable or inference fails.
    """

    def __init__(
        self,
        weights: Path | str | None = None,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        device: str | None = None,
    ) -> None:
        self._weights = Path(weights) if weights is not None else None
        self._confidence = float(confidence)
        self._iou = float(iou)
        self._max_detections = int(max_detections)
        self._device = device
        self._model: Any | None = None
        self._lock = threading.Lock()

    # -- Detector protocol -------------------------------------------------------------

    def detect(self, screenshot: Screenshot) -> list[Element]:
        """Detect UI elements, highest confidence first, boxes in LOGICAL pixels.

        Inference runs on native PHYSICAL pixels - the sharpest image available - so every
        box is divided by ``scale`` on the way out and clipped to the logical viewport: a
        model may predict a box off the edge of the frame, and a click target that is is a bug.

        Raises:
            PerceptionError: the weights are missing or inference failed.
        """
        model = self._load()
        # ultralytics reads a numpy array as BGR, and torch refuses the negative stride a
        # bare ``[..., ::-1]`` view would hand it.
        image = np.ascontiguousarray(screenshot.to_array(logical=False)[:, :, ::-1])
        try:
            results = model.predict(
                source=image,
                conf=self._confidence,
                iou=self._iou,
                max_det=self._max_detections,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - any torch failure is a perception failure
            raise PerceptionError(f"YOLO inference failed: {exc}") from exc

        viewport = Box(0, 0, screenshot.width, screenshot.height)
        elements: list[Element] = []
        for result in results:
            elements.extend(self._elements_of(result, screenshot.scale, viewport))
        elements.sort(key=lambda element: -element.confidence)
        return elements

    # -- internals ---------------------------------------------------------------------

    def weights_path(self) -> Path:
        """The weights file this detector will load (resolved, not checked)."""
        return self._weights if self._weights is not None else default_weights_path()

    def _load(self) -> Any:
        """Load the model once, on first use."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            path = self.weights_path()
            if not path.is_file():
                raise PerceptionError(
                    f"YOLO detector weights not found at {path}. "
                    f"Set SKILLWEAVER_DATA_DIR, or pass weights=..., if they live "
                    f"elsewhere. {_BUILD_HINT}"
                )
            try:
                YOLO = _import_yolo()
            except ImportError as exc:  # pragma: no cover - ultralytics is a hard dependency
                raise PerceptionError(f"ultralytics is not installed: {exc}") from exc
            try:
                model = YOLO(str(path))
            except Exception as exc:  # noqa: BLE001 - a corrupt checkpoint is a perception failure
                raise PerceptionError(f"could not load YOLO weights from {path}: {exc}") from exc
            self._check_classes(model, path)
            self._model = model
            return model

    def _check_classes(self, model: Any, path: Path) -> None:
        """Refuse weights whose class map is not ``labeling.CLASS_NAMES``.

        Class ids are POSITIONAL, so a checkpoint trained against a different map would
        report confident nonsense - every button relabelled a checkbox - and nothing
        downstream could tell.
        """
        names = getattr(model, "names", None)
        if not names:
            return
        ordered = tuple(str(names[index]) for index in sorted(names))
        if ordered != CLASS_NAMES:
            raise PerceptionError(
                f"YOLO weights at {path} were trained on classes {ordered}, but this "
                f"build expects {CLASS_NAMES}. {_BUILD_HINT}"
            )

    def _elements_of(self, result: Any, scale: float, viewport: Box) -> list[Element]:
        """Convert one ultralytics result into contract elements."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        try:
            xyxy = boxes.xyxy.cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
        except Exception as exc:  # noqa: BLE001 - a surprising result shape is a perception failure
            raise PerceptionError(f"could not read YOLO detections: {exc}") from exc

        elements: list[Element] = []
        for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes, strict=True):
            try:
                kind = kind_of(int(cls))
            except ValueError as exc:
                raise PerceptionError(str(exc)) from exc
            box = _clip(_to_logical(x1, y1, x2, y2, scale), viewport)
            if box.area == 0:
                continue
            elements.append(
                Element(
                    box=box,
                    kind=kind,
                    text="",
                    confidence=float(score),
                    source=ElementSource.yolo,
                )
            )
        return elements


def _import_yolo():  # noqa: ANN202 - the ultralytics class, imported lazily
    """Import ``YOLO`` and undo the one side effect ultralytics has on the process.

    It replaces ``PIL.Image.open`` with a wrapper that, on the first failed decode, tries
    to register a HEIF plugin - which runs ``pip install pi-heif``. Process-wide, that
    turns a corrupt PNG into ``ModuleNotFoundError`` instead of the
    ``UnidentifiedImageError`` ``Screenshot.to_array`` promises, and starts a package
    install from code that should never touch the network. Screenshots are PNGs, so the
    wrapper buys nothing here.
    """
    import PIL.Image
    from ultralytics import YOLO

    if getattr(PIL.Image.open, "__module__", "").startswith("ultralytics"):
        from ultralytics.utils import patches

        original = getattr(patches, "_image_open", None)
        if original is not None:
            PIL.Image.open = original
    return YOLO


def _to_logical(x1: float, y1: float, x2: float, y2: float, scale: float) -> Box:
    """A physical-pixel ``xyxy`` corner pair as a logical-pixel :class:`Box`.

    Both corners convert before the size is taken, so independent rounding of origin and
    width cannot grow or shrink the box by a pixel.
    """
    left, top = round(x1 / scale), round(y1 / scale)
    right, bottom = round(x2 / scale), round(y2 / scale)
    return Box(x=left, y=top, w=max(right - left, 0), h=max(bottom - top, 0))


def _clip(box: Box, viewport: Box) -> Box:
    """``box`` trimmed to the viewport; zero-size when it falls entirely outside."""
    left = max(box.x, viewport.x)
    top = max(box.y, viewport.y)
    right = min(box.x + box.w, viewport.x + viewport.w)
    bottom = min(box.y + box.h, viewport.y + viewport.h)
    return Box(x=left, y=top, w=max(right - left, 0), h=max(bottom - top, 0))
