"""The YOLO element detector: a :class:`~skillweaver.contracts.Detector` over pixels.

This is how the agent sees a button. It takes a :class:`Screenshot` and returns
:class:`Element` boxes with an :class:`ElementKind`, in LOGICAL pixels, with real
confidences and ``source=ElementSource.yolo``. It never reads the DOM - the whole
premise of the project is that it does not have one.

Weights ship with the repository at ``data/models/ui_detector.pt`` - the one file
under ``data/`` that is not git-ignored - so a clean clone detects immediately, with
no training step and no network. They are produced by :mod:`scripts.train_detector`,
which fine-tunes a small YOLO model on a dataset labelled by ``BrowserGroundTruth``
(see :mod:`scripts.build_ui_dataset`); that dataset stays ignored because it is
large and regenerable.

A detector still has to cope with them being gone, because ``SKILLWEAVER_DATA_DIR``
can point anywhere and a demo laptop is not a clean clone.

**Missing weights raise.** Returning an empty list would be indistinguishable from
a blank screen, and an agent told it sees nothing on a full page of buttons fails
in a way nobody can debug. :class:`PerceptionError` with the path and the command
that builds it is the honest answer.

Loading is lazy: constructing a ``YoloDetector`` touches no file and imports no
torch, so ``from skillweaver.perception.detect_yolo import YoloDetector`` stays
cheap for code that may never detect anything.

Typical use::

    detector = YoloDetector()                      # nothing loaded yet
    elements = detector.detect(controller.capture())
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from skillweaver.config import settings
from skillweaver.contracts import Box, Element, ElementSource, Screenshot
from skillweaver.errors import PerceptionError
from skillweaver.perception.arrays import array_of
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
"""Ceiling on boxes per frame. A dense table screen genuinely has a couple of
hundred elements; far more than that is a model melting down, not a screen."""

_BUILD_HINT = (
    "build it with:\n"
    "  uv run python scripts/build_ui_dataset.py --out data/models/ui-dataset\n"
    "  uv run python scripts/train_detector.py --data data/models/ui-dataset/data.yaml"
)


_loaded: dict[tuple[str, int, int, str | None], Any] = {}
"""Models already read from disk this process, keyed by checkpoint identity.

A detector is built per session and an evaluation opens one session per run, so
without this the same unchanged file is read from disk dozens of times in a suite.
See :meth:`YoloDetector._load` for why the key is what it is.
"""

_loaded_lock = threading.Lock()


def default_weights_path() -> Path:
    """Where the detector expects its weights: ``<models_dir>/ui_detector.pt``."""
    return settings().models_dir / DEFAULT_WEIGHTS_NAME


def forget_loaded_models() -> None:
    """Drop every process-cached model. For a test that must load one afresh."""
    with _loaded_lock:
        _loaded.clear()


class YoloDetector:
    """A ``contracts.Detector`` backed by an ultralytics YOLO model.

    Args:
        weights: Path to a ``.pt`` file. ``None`` means :func:`default_weights_path`.
        confidence: Minimum score for a detection to be returned.
        iou: Non-maximum-suppression IoU threshold.
        max_detections: Hard ceiling on boxes returned for one frame.
        device: Torch device string (``"cpu"``, ``"mps"``, ``"cuda:0"``). ``None``
            lets ultralytics choose.

    The instance is safe to share between threads: the one-time model load is
    guarded by a lock, and inference itself goes straight to ultralytics.

    Raises:
        PerceptionError: from :meth:`detect` (never from ``__init__``) when the
            weights are missing or unloadable, or when inference fails.
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

        Inference runs on the screenshot's native PHYSICAL pixels, because that is
        the sharpest image available and a Retina frame carries real extra detail.
        Every box is therefore divided by ``screenshot.scale`` on the way out, and
        clipped to the logical viewport - a model is free to predict a box that
        runs off the edge of the frame, and a click target that does is a bug.

        Raises:
            PerceptionError: if the weights are missing or inference fails.
        """
        model = self._load()
        # ultralytics reads a numpy array as BGR, and torch refuses the negative
        # stride a bare ``[..., ::-1]`` view would hand it.
        image = np.ascontiguousarray(array_of(screenshot, logical=False)[:, :, ::-1])
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
        """Load the model once, on first use, and once per process per checkpoint.

        The instance cache is what stops one detector reloading between frames. The
        process cache below it is what stops an evaluation reloading between RUNS:
        every session builds its own detector, so a suite of fourteen tasks at five
        runs each paid seventy loads of the same unchanged file - minutes of wall
        clock spent reading a checkpoint that was already in memory.

        Keyed by the path AND its modification time and size, so a checkpoint that
        was retrained under the same name is read again rather than silently served
        from before. Sharing is safe because a loaded model is used read-only here:
        inference takes the image as an argument and keeps nothing of it.

        Raises:
            PerceptionError: if the file is absent or ultralytics cannot read it.
        """
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            path = self.weights_path()
            if not path.is_file():
                raise PerceptionError(
                    f"YOLO detector weights not found at {path}. "
                    f"Set SKILLWEAVER_DATA_DIR or pass weights=..., or {_BUILD_HINT}"
                )
            stat = path.stat()
            key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, self._device)
            with _loaded_lock:
                shared = _loaded.get(key)
            if shared is not None:
                self._model = shared
                return shared
            try:
                YOLO = _import_yolo()
            except ImportError as exc:  # pragma: no cover - ultralytics is a hard dependency
                raise PerceptionError(f"ultralytics is not installed: {exc}") from exc
            try:
                model = YOLO(str(path))
            except Exception as exc:  # noqa: BLE001 - a corrupt checkpoint is a perception failure
                raise PerceptionError(f"could not load YOLO weights from {path}: {exc}") from exc
            self._check_classes(model, path)
            with _loaded_lock:
                _loaded[key] = model
            self._model = model
            return model

    def _check_classes(self, model: Any, path: Path) -> None:
        """Refuse weights whose class map is not the one in ``labeling.CLASS_NAMES``.

        Class ids are positional, so a checkpoint trained against a different map
        would report perfectly confident nonsense - every button relabelled as a
        checkbox - and nothing downstream could tell. Failing here is loud and
        cheap; the alternative is a week of debugging the agent.
        """
        names = getattr(model, "names", None)
        if not names:
            return
        ordered = tuple(str(names[index]) for index in sorted(names))
        if ordered != CLASS_NAMES:
            raise PerceptionError(
                f"YOLO weights at {path} were trained on classes {ordered}, but this "
                f"build expects {CLASS_NAMES}; retrain the detector - {_BUILD_HINT}"
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

    Importing ultralytics replaces ``PIL.Image.open`` with a wrapper that, the first
    time a decode fails, tries to register a HEIF plugin - and registering it runs
    ``pip install pi-heif``. Two things then go wrong for everybody else in the
    process, not just for this detector: a corrupt PNG raises ``ModuleNotFoundError``
    instead of the ``UnidentifiedImageError`` that ``Screenshot.to_array`` promises to
    turn into a ``PerceptionError``, and code that should never touch the network
    starts a package install. A test suite that imports this module once inherits both.

    skillweaver decodes screenshots, which are PNGs, so the wrapper buys nothing here.
    Putting the original back is the smallest fix that keeps the contract honest. If a
    future ultralytics stops exposing the original, the patch simply stays in place.
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

    Both corners are converted before the size is taken, so a box never grows or
    shrinks by a pixel through independent rounding of its origin and its width.
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
